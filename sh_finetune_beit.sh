export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=4 --master_port=29502 runs/run_class_finetuning.py \
    --model "beit_base_patch16_224" `# 要微调的模型名称` \
    --data_path "data/imagewoof2-320/val" `# 数据集路径` \
    --nb_classes 10 `# 分类类别数量` \
    --data_set "image_folder" `# 数据集类型` \
    --disable_eval_during_finetuning `# 在微调过程中禁用评估` \
    --finetune "output/pretrain/checkpoint-19.pth" `# 预训练模型权重路径` \
    --output_dir "output/finetune" `# 保存模型的输出路径` \
    --save_ckpt_freq 5 `# 保存模型的频率` \
    --batch_size 16 `# 每个批次的样本数量` \
    --epochs 20 `# 训练的总轮数` \
    --lr 3e-3 `# 学习率` \
    --update_freq 1 `# 参数更新频率` \
    --warmup_epochs 5 `# 预热学习率的轮数` \
    --layer_decay 0.65 `# 层衰减率，控制不同层的学习率` \
    --drop_path 0.2 `# 路径丢弃率(Drop Path)` \
    --weight_decay 0.05 `# 权重衰减率，用于正则化` \
    --layer_scale_init_value 0.1 `# 层缩放初始化值` \
    --clip_grad 3.0 `# 梯度裁剪阈值，防止梯度爆炸`