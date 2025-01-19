export PYTHONPATH=$PYTHONPATH:$(pwd)
OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=4 runs/run_beit_pretraining.py \
    --data_path "data/imagewoof2-320/train" `# 预训练数据集路径` \
    --output_dir "output/pretrain/" `# 保存模型、日志的路径` \
    --model "beit_base_patch16_224_8k_vocab" `# 要训练的模型名称` \
    --discrete_vae_type "dall-e" `# VAE的类型` \
    --discrete_vae_weight_path "data/dalle" `# VAE权重的路径` \
    --num_mask_patches 75 `# 需要掩蔽的视觉标记/补丁数量` \
    --batch_size 16 `# 每个批次的样本数量` \
    --lr 5e-4 `# 学习率` \
    --warmup_epochs 5 `# 预热学习率的轮数` \
    --epochs 20 `# 训练的总轮数` \
    --clip_grad 3.0 `# 梯度裁剪阈值, 防止梯度爆炸。` \
    --save_ckpt_freq 20 `# 保存检查点的频率(每20个epoch保存一次)` \
    --drop_path 0.1 `# 路径丢弃率(Drop Path)` \
    --layer_scale_init_value 0.1 `# 层缩放初始化值` \
    --imagenet_default_mean_and_std `# 使用ImageNet的默认均值和标准差`