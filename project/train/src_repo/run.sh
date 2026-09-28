#!/usr/bin/env bash
# =============================================================================
# 极市平台训练命令脚本
# 平台通过: bash /project/train/src_repo/run.sh 发起训练
# -----------------------------------------------------------------------------
# 目录规范:
#   - 代码目录:   /project/train/src_repo/
#   - 模型保存:   /project/train/models/
#   - 日志保存:   /project/train/log/log.txt
#   - 训练结果图: /project/train/result-graphs/
#   - 数据集:     /home/data/
# =============================================================================
set -e

# 切换到代码目录
cd /project/train/src_repo

# 创建必要目录
mkdir -p /project/train/models
mkdir -p /project/train/log
mkdir -p /project/train/result-graphs

# 依次微调 SDK 使用的两个模型。
python /project/train/src_repo/finetune_facexformer.py \
    --data_dir /home/data \
    --epochs 5 \
    --batch_size 16 \
    --lr 1e-5 \
    --num_workers 4 \
    --device cuda \
    --checkpoint /project/ev_sdk/model/facexformer/model_original.pt \
    --output /project/train/models/facexformer_finetuned.pt

python /project/train/src_repo/finetune_swinface.py \
    --data_dir /home/data \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-5 \
    --num_workers 4 \
    --device cuda \
    --checkpoint /project/ev_sdk/model/swinface/checkpoint_original.pt \
    --output /project/train/models/swinface_finetuned.pt

echo "训练完成，临时微调权重已保存至 /project/train/models/"
