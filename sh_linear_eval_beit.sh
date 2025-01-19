export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 runs/run_linear_eval.py \
    --model "beit_base_patch16_224" `# 模型名称` \
    --data_path "data/imagewoof2-320" `# 数据集路径` \
    --output_dir "output/linear_eval" `# 输出目录` \
    --pretrained_weights "output/finetune/checkpoint-19.pth" `# 预训练权重路径` \
    --checkpoint_key "model" `# 检查点中的模型键名` \
    --num_labels 10 `# 分类标签数量` \
    --batch_size_per_gpu 32 `# 每个GPU的批次大小` \
    --epochs 20 `# 训练轮数` \
    --lr 0.001 `# 学习率` \
    --num_workers 4 `# 数据加载的工作进程数` \
    --val_freq 5 `# 验证频率` \
    --optimizer "adamw" `# 优化器类型` \
    --layer_scale_init_value 0.1 `# 层缩放初始化值` \
    --imagenet_default_mean_and_std true`# 使用ImageNet默认的均值和标准差` \
