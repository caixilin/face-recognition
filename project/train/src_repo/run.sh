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

# -----------------------------------------------------------------------------
# 训练规模参数
#   平台「执行命令」填写 bash /project/train/src_repo/run.sh。
#   使用本入口时在这里改训练规模，改完重新上传本文件。
#
#   MAX_SAMPLES 是 12 小时上限的核心阀门：多个数据集同时挂载时 /home/data 下会有
#   多份标注被合并成一份，目标数可达 30 万以上，全量跑必然超时。
#   脚本会按数据来源等量分层抽样，所以抽出来的子集覆盖每一份数据，
#   而不是只取路径最靠前的那一份。0 = 全部。
#
#   怎么定这个值：在平台终端跑一次单 epoch 计时，算出每秒能处理多少样本，
#   再让 MAX_SAMPLES * 0.8 * EPOCHS / 每秒样本数 留出 12 小时内的余量。
#   注意两个脚本是串行执行（set -e），前一个没跑完后一个不会开始，
#   给 SwinFace 也要留时间。
# -----------------------------------------------------------------------------
EPOCHS=5
MAX_SAMPLES=100000
BATCH_SIZE_FACEXFORMER=16
BATCH_SIZE_SWINFACE=32

# 创建必要目录
mkdir -p /project/train/models
mkdir -p /project/train/log
mkdir -p /project/train/result-graphs

# 在长时间训练前检查 SwinFace 依赖，避免 FaceXFormer 跑完才发现缺包。
# 使用当前训练解释器安装，保证依赖落在同一个 Python 环境。
if ! python -c 'from timm.models.layers import DropPath, to_2tuple, trunc_normal_'; then
    echo "检查未通过，安装 SwinFace 依赖 timm==0.6.13 ..."
    python -m pip install "timm==0.6.13"
fi
python -c 'from timm.models.layers import DropPath, to_2tuple, trunc_normal_'

# --help 会导入两套模型源码，但不会加载数据或启动训练。
python /project/train/src_repo/finetune_facexformer.py --help > /dev/null
python /project/train/src_repo/finetune_swinface.py --help > /dev/null

# 依次微调 SDK 使用的两个模型。
python /project/train/src_repo/finetune_facexformer.py \
    --data_dir /home/data \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE_FACEXFORMER" \
    --max_samples "$MAX_SAMPLES" \
    --lr 1e-5 \
    --num_workers 4 \
    --device cuda \
    --checkpoint /project/ev_sdk/model/facexformer/model_original.pt \
    --output /project/train/models/facexformer_finetuned.pt

python /project/train/src_repo/finetune_swinface.py \
    --data_dir /home/data \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE_SWINFACE" \
    --max_samples "$MAX_SAMPLES" \
    --lr 1e-5 \
    --num_workers 4 \
    --device cuda \
    --checkpoint /project/ev_sdk/model/swinface/checkpoint_original.pt \
    --output /project/train/models/swinface_finetuned.pt

echo "训练完成，临时微调权重已保存至 /project/train/models/"
