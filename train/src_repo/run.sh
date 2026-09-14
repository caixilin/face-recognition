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

# 启动训练（使用绝对路径）
python /project/train/src_repo/train.py \
    --data_dir /home/data \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --input_size 112 \
    --device cuda

echo "训练完成，模型已保存至 /project/train/models/"
