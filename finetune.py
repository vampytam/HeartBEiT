# Finetuning code

import os
import time

import tqdm
import torch
import torchvision

import pandas as pd
import numpy as np

from PIL import Image
from sklearn import metrics, model_selection

from config import Config
from load_checkpoint import LoadCheckpoint
from dataset.dataset_folder import DatasetFolder, default_loader, IMG_EXTENSIONS

np.random.seed(Config.random_state)
torch.backends.cudnn.benchmark = True




class ECGDataset(torch.utils.data.Dataset):
    # Should really only be concerned with returning image tensors
    # when supplied with a path
    # Will be supplied with a transforms keyword arg (inherited from torchvision)
    # Use pandas fuckery to create the dataframe proper

    def __init__(self, df, model_identifier, return_image_path=False):
        self.df = df
        self.dtype = torch.long if Config.ce_loss else torch.float32
        self.return_image_path = return_image_path

        # Use imagenet mean/median by default
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if model_identifier == 'vit':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),  # 将较短边调整为224
            torchvision.transforms.CenterCrop(224),  # 从中心裁剪出224x224的正方形
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Expects 2 columns: PATH / OUTCOME
        image_path = self.df.iloc[index]['PATH']
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        label = self.df.iloc[index]['LABEL']
        label = torch.tensor(label, dtype=self.dtype)  # CE loss expects long

        if self.return_image_path:
            return image, label, image_path

        return image, label


class Finetune:
    def __init__(self, outcome, dataset):
        self.outcome = outcome

        # Separate into internal and external datasets
        # ext_val_site = "ST.LUKE'S-ROOSEVELT HOSPITAL (S)"
        # self.dataset = dataset.query('SITENAME != @ext_val_site')
        # self.ext_dataset = dataset.query('SITENAME == @ext_val_site')
        split_index = int(len(dataset) * 0.9)
        self.dataset = dataset.iloc[:split_index]
        self.ext_dataset = dataset.iloc[split_index:]

    @staticmethod
    def eval_model(dataloader, model):
        if len(dataloader) == 0:
            return 0, 0, 0, None

        # Evaluate the model for this epoch
        all_preds = []
        all_labels = []

        # CE or BCE loss
        criterion = torch.nn.CrossEntropyLoss() if Config.ce_loss else torch.nn.BCEWithLogitsLoss()

        model.eval()
        for images, labels in tqdm.tqdm(dataloader):
            testing_epoch_loss = 0

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    images = images.cuda()
                    labels = labels.cuda()

                    outputs = model(images)

                    # Calculate loss - probably not required
                    testing_loss = criterion(outputs.squeeze(), labels)
                    testing_epoch_loss += testing_loss.item() * outputs.shape[0]

                    # Put all outputs together
                    if Config.ce_loss:
                        # This allows for the usage of the rest of the pipeline as is without making any changes
                        normalized_preds = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                    else:
                        normalized_preds = torch.sigmoid(outputs).detach().cpu().numpy()

                    all_preds.extend(normalized_preds)
                    all_labels.extend(labels.cpu().numpy())

        # Overall epoch loss
        testing_loss = testing_epoch_loss / len(dataloader)

        df_pred = pd.DataFrame({
            'TRUE': np.array(all_labels).astype(int),
            'PRED': np.array(all_preds).astype(float)
        })
        
        valid_mask = df_pred['PRED'].notna()
        if not valid_mask.all():
            print(f"filtered {len(df_pred)-valid_mask.sum()} invalid number")
            df_pred = df_pred[valid_mask].copy()
        
        if len(df_pred) < 2:
            print(f"not enough valid number: ({len(df_pred)})")
            return float('nan'), float('nan'), testing_loss, df_pred
        
        unique_labels = df_pred['TRUE'].unique()
        if len(unique_labels) < 2:
            print(f"only one class of label: {dict(df_pred['TRUE'].value_counts())}")
            return float('nan'), float('nan'), testing_loss, df_pred
        
        try:
            auroc = metrics.roc_auc_score(df_pred['TRUE'], df_pred['PRED'])
        except ValueError as e:
            print(df_pred['PRED'].describe())
            return float('nan'), float('nan'), testing_loss, df_pred
        precision, recall, _ = metrics.precision_recall_curve(df_pred['TRUE'], df_pred['PRED'])
        aupr = metrics.auc(recall, precision)
        
        return auroc, aupr, testing_loss, df_pred

    def gaping_maw(self, train_dataloader, test_dataloader, ext_val_dataloader, model, model_identifier, percentage):
        print('Model identifier:', model_identifier)

        # 检查训练数据集大小
        if len(train_dataloader) == 0:
            print(f"Warning: Training dataloader is empty with {percentage*100}% data. Skipping this training.")
            return

        # Housekeeping
        result_dir = os.path.join(self.outcome, model_identifier)
        os.makedirs(result_dir, exist_ok=True)

        # Use this model as the base for training further
        # Expected to have proper architecture when it arrives here
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        
        # Create model params
        if model_identifier.startswith('vit'):
            # TODO: Sweep through these hyperparameters
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        # CE or BCE loss
        criterion = torch.nn.CrossEntropyLoss() if Config.ce_loss else torch.nn.BCEWithLogitsLoss()

        scaler = torch.cuda.amp.GradScaler()
        
        # 确保 steps_per_epoch 至少为 1
        steps_per_epoch = max(1, len(train_dataloader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=Config.ft_epochs,
            steps_per_epoch=steps_per_epoch)

        # Keep track of performance
        all_results = []

        # Start the loop
        for epoch in range(Config.ft_epochs):
            epoch_loss = 0

            model.train()
            for images, labels in tqdm.tqdm(train_dataloader):

                with torch.cuda.amp.autocast():
                    images = images.cuda()
                    labels = labels.cuda()

                    # Same as optim.zero_grad()
                    for param in model.parameters():
                        param.grad = None

                    # Forward pass
                    outputs = model(images)

                    # Calculate loss
                    loss = criterion(outputs.squeeze(), labels)
                    epoch_loss += loss.item() * outputs.shape[0]

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Continue training for 5 epochs more
                    if epoch < Config.ft_epochs:
                        scheduler.step()

            # Overall epoch loss
            training_loss = epoch_loss / len(train_dataloader)

            # Get learning rate from scheduler
            lr = scheduler.get_last_lr()[0]

            # Other things about the training
            training_samples = len(train_dataloader.dataset)
            testing_samples = len(test_dataloader.dataset)
            ext_val_samples = len(ext_val_dataloader.dataset)

            # Evalulate model on testing and external validation sets
            auroc, aupr, testing_loss, df_pred = Finetune.eval_model(test_dataloader, model)
            ext_auroc, ext_aupr, ext_testing_loss, ext_df_pred = Finetune.eval_model(ext_val_dataloader, model)

            # Results for this model for this epoch
            all_results.append([
                model_identifier, percentage, epoch, auroc, aupr, lr, training_loss, testing_loss,
                ext_auroc, ext_aupr, ext_testing_loss, training_samples, testing_samples, ext_val_samples])
            df_results = pd.DataFrame(all_results)
            df_results.columns = [
                'MODEL', 'PERCENTAGE', 'EPOCH', 'AUROC', 'AUPR', 'LR', 'TRAINING_LOSS', 'TESTING_LOSS',
                'EXT_AUROC', 'EXT_AUPR', 'EXT_TESTING_LOSS', 'TRAINING_SAMPLES', 'TESTING_SAMPLES', 'EXT_VAL_SAMPLES']

            print('Model', model_identifier, 'Epoch:', epoch, 'LR:', lr)
            print('AUROC:', auroc, 'AUPR:', aupr, 'Training loss:', training_loss, 'Testing loss:', testing_loss)
            print('Ext AUROC:', ext_auroc, 'Ext AUPR:', ext_aupr, 'Ext Testing loss:', ext_testing_loss)

            outfile_name = os.path.join(result_dir, f'results_{percentage}.csv')
            df_results.to_csv(outfile_name)

            # Prediction probabilities for this model for this epoch
            outfile_name = os.path.join(result_dir, f'prob_{percentage}_{epoch}.csv')
            df_pred.to_csv(outfile_name)

            # Prediction probabilities for this model for this epoch
            outfile_name = os.path.join(result_dir, f'ext_prob_{percentage}_{epoch}.csv')
            if ext_df_pred is not None:
                ext_df_pred.to_csv(outfile_name)

            # Save the model at the end of each epoch
            model_out_dir = os.path.join(result_dir, 'models')
            os.makedirs(model_out_dir, exist_ok=True)
            outfile_name = os.path.join(model_out_dir, f'model_{percentage}_{epoch}.pt')
            torch.save(model.state_dict(), outfile_name)

    def create_model(self, torchvision_identifier, identifier):
        # Identifier is native to this project
        # torchvision_identifier is the identifier used by torchvision
        
        if identifier == 'vit':
            model = LoadCheckpoint(checkpoint_path=Config.checkpoint_path).hammer_time()

        elif identifier == 'vit_imagenet':
            model = LoadCheckpoint(checkpoint_path=Config.imagenet_checkpoint_path).hammer_time()

        else:
            weight_identifier = Config.models[identifier]['weights']
            weights = None
            if weight_identifier is not None:
                weights = eval(f'torchvision.models.{weight_identifier}.IMAGENET1K_V1')
            model = eval(f'torchvision.models.{torchvision_identifier}(weights=weights)')

            # Rename the identifier by splitting away the -
            identifier = identifier.split('-')[0]

            # CE or BCE loss
            num_classes = 2 if Config.ce_loss else 1

            # Classification head
            if identifier == 'ImageNetVit':  # This is from the torchvision library - doesn't work
                model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
            if identifier == 'efficientnet_b4':
                model.classifier[1] = torch.nn.Linear(1792, num_classes)
            if identifier == 'densenet201':
                model.classifier = torch.nn.Linear(1920, num_classes)
            if identifier.startswith('resnet'):
                model.fc = torch.nn.Linear(2048, num_classes)

        return model

    def create_dataloaders(self, model_identifier, percentage, return_dataframes=False):
        train_test_split = model_selection.train_test_split(
            self.dataset,
            test_size=0.2,  # 80%训练集, 20%测试集
            stratify=self.dataset['LABEL'],  # 按标签分层采样
            random_state=Config.random_state
        )
        
        df_train, df_test = train_test_split

        if return_dataframes:
            return df_train, df_test, self.ext_dataset

        # Instead of sampling to reduce dataset size, do a train_test split and stratify by LABEL
        if percentage < 1:  # Raises an error if percentage is 100%
            df_train, _ = model_selection.train_test_split(
                df_train, train_size=percentage,
                stratify=df_train['LABEL'],
                shuffle=True,
                random_state=Config.random_state)
        print(f'Restricting to {percentage} of training data: {len(df_train)}')

        # Create datasets with these data splits
        train_dataset = ECGDataset(df_train, model_identifier)
        test_dataset = ECGDataset(df_test, model_identifier)
        ext_val_dataset = ECGDataset(self.ext_dataset, model_identifier)
        
        # Print outcome prevalence
        print('Outcome:', self.outcome)
        print('Training prevalence:', (df_train['LABEL'].sum() / df_train['LABEL'].count()) * 100, '%')
        print('Testing prevalence:', (df_test['LABEL'].sum() / df_test['LABEL'].count()) * 100, '%')
        print('External validation prevalence:', (self.ext_dataset['LABEL'].sum() / self.ext_dataset['LABEL'].count()) * 100, '%')

        # 动态调整 batch_size，确保至少有一个 batch
        train_size = len(train_dataset)
        batch_size = min(Config.batch_size, max(1, train_size // 2))
        print(f'Using batch_size: {batch_size} for training size: {train_size}')

        # Create data loaders with these datasets
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=40,
            drop_last=True,
            pin_memory=True)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=40,
            pin_memory=True)
        
        ext_val_dataloader = torch.utils.data.DataLoader(
            ext_val_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=40,
            pin_memory=True)

        return train_dataloader, test_dataloader, ext_val_dataloader

    def hammer_time(self):
        errorslog = os.path.join(self.outcome, 'Errors.log')
        with open(errorslog, 'a') as outfile:
            outfile.write(f'{time.ctime()} New run\n')

        for percentage in Config.finetuning_percentage_iter:

            for model_identifier in Config.models:
                try:
                    model = self.create_model(
                        Config.models[model_identifier]['internal_identifier'], model_identifier)

                    train_dataloader, test_dataloader, ext_val_dataloader = \
                        self.create_dataloaders(model_identifier, percentage)

                    self.gaping_maw(
                        train_dataloader, test_dataloader, ext_val_dataloader,
                        model, model_identifier, percentage)

                except Exception as e:
                    with open(errorslog, 'a') as outfile:
                        outfile.write(f'{time.ctime()} {self.outcome} {model_identifier} {percentage} {e}\n')

def main():
    torch.manual_seed(Config.random_state)
    torch.cuda.manual_seed(Config.random_state)
    
    ds_folder = DatasetFolder(Config.dir_ecg_plots, loader=default_loader, extensions=IMG_EXTENSIONS)
    dataset = pd.DataFrame(ds_folder.samples, columns=['PATH', 'LABEL'])
    
    # 1 创建微调类
    finetune = Finetune(Config.outcome, dataset) # outcome: 输出目录, dataset: 数据集pandas dataframe['PATH', 'LABEL', 'MRN', 'SITENAME']
    
    for model_identifier in Config.models:
        print(f"\ntraining begin: {model_identifier}")
        
        try:
            # 2 创建对应模型，加载预训练权重
            model = finetune.create_model(
                Config.models[model_identifier]['internal_identifier'],
                model_identifier
            )
            
            for percentage in Config.finetuning_percentage_iter:
                print(f"\n使用 {percentage*100}% 的训练数据")
                
                # 3 数据加载器
                train_loader, test_loader, ext_val_loader = finetune.create_dataloaders(
                    model_identifier=model_identifier,
                    percentage=percentage
                )
                
                # 4 微调和评估
                finetune.gaping_maw(
                    train_dataloader=train_loader,
                    test_dataloader=test_loader, 
                    ext_val_dataloader=ext_val_loader,
                    model=model,
                    model_identifier=model_identifier,
                    percentage=percentage
                )
                
        except Exception as e:
            import logging
            logging.exception(e)
            print(f"when training {model_identifier} {percentage}%, an error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()