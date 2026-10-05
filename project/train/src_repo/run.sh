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
set -euo pipefail

# 切换到代码目录
cd /project/train/src_repo

# -----------------------------------------------------------------------------
# 训练规模参数
#   平台「执行命令」填写 bash /project/train/src_repo/run.sh。
#   使用本入口时在这里改训练规模，改完重新上传本文件。
#
#   MAX_SAMPLES 是 12 小时上限的核心阀门：多个数据集同时挂载时 /home/data 下会有
#   多份标注被合并成一份，目标数可达 30 万以上，全量可能超时。
#   脚本会按数据来源等量分层抽样，所以抽出来的子集覆盖每一份数据，
#   而不是只取路径最靠前的那一份。0 = 全部。
#
#   怎么定这个值：在平台终端跑一次单 epoch 计时，算出每秒能处理多少样本，
#   再让 MAX_SAMPLES * 0.8 * EPOCHS / 每秒样本数 留出 12 小时内的余量。
#   注意三个脚本串行执行，前一个没跑完后一个不会开始，
#   给 SwinFace 也要留时间。
# -----------------------------------------------------------------------------
EPOCHS=5
MAX_SAMPLES=100000
# 验证样本上限不是平台限制；设 0 可评估所选训练子集的完整验证部分。
MAX_VAL_SAMPLES=2000
BATCH_SIZE_FACEXFORMER=16
BATCH_SIZE_SWINFACE=32
DETECTOR_EPOCHS=5
# 0 = 使用挂载数据中全部符合条件的原图，仍按原图留出验证集。
DETECTOR_MAX_IMAGES=0
DETECTOR_MAX_VAL_IMAGES=500
DETECTOR_BATCH_SIZE=2

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

# --help 会导入三套训练源码，但不会加载数据或启动训练。
python /project/train/src_repo/finetune_facexformer.py --help > /dev/null
python /project/train/src_repo/finetune_swinface.py --help > /dev/null
python /project/train/src_repo/train_head_detector.py --help > /dev/null
python -c 'from torchvision.ops import nms; import torch; nms(torch.tensor([[0., 0., 2., 2.]]), torch.tensor([1.]), 0.5)'

# 明确选择三套初始权重，优先原始属性备份，缺失时使用现有 SDK 文件。
FACEXFORMER_CHECKPOINT=/project/ev_sdk/model/facexformer/model_original.pt
if [ ! -f "$FACEXFORMER_CHECKPOINT" ]; then
    FACEXFORMER_CHECKPOINT=/project/ev_sdk/model/facexformer/model.pt
fi
SWINFACE_CHECKPOINT=/project/ev_sdk/model/swinface/checkpoint_original.pt
if [ ! -f "$SWINFACE_CHECKPOINT" ]; then
    SWINFACE_CHECKPOINT=/project/ev_sdk/model/swinface/checkpoint_step_79999_gpu_0.pt
fi
for checkpoint in "$FACEXFORMER_CHECKPOINT" "$SWINFACE_CHECKPOINT"; do
    if [ ! -f "$checkpoint" ]; then
        echo "缺少属性初始权重: $checkpoint" >&2
        exit 1
    fi
done

# 可以显式指定已有头肩或离线 COCO 文件；默认复用用户已有的头肩模型。
DETECTOR_INIT_ARGS=()
if [ -n "${HEAD_CHECKPOINT:-}" ]; then
    DETECTOR_INIT_ARGS=(--checkpoint "$HEAD_CHECKPOINT")
elif [ -n "${COCO_CHECKPOINT:-}" ]; then
    DETECTOR_INIT_ARGS=(--coco_checkpoint "$COCO_CHECKPOINT")
elif [ -f /project/ev_sdk/model/headshoulder/model.pt ]; then
    DETECTOR_INIT_ARGS=(--checkpoint /project/ev_sdk/model/headshoulder/model.pt)
elif [ -f /project/ev_sdk/model/headshoulder/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth ]; then
    DETECTOR_INIT_ARGS=(--coco_checkpoint /project/ev_sdk/model/headshoulder/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth)
else
    echo "缺少头肩权重和离线 COCO 初始化文件，请先上传；本入口不等待联网下载。" >&2
    exit 1
fi

HEAD_ARGS=(--data_dir /home/data --epochs "$DETECTOR_EPOCHS" --batch_size "$DETECTOR_BATCH_SIZE"
    --max_images "$DETECTOR_MAX_IMAGES" --max_val_images "$DETECTOR_MAX_VAL_IMAGES"
    --num_workers 4 --device cuda --keep_epoch_checkpoints
    --output /project/train/models/headshoulder_finetuned.pt "${DETECTOR_INIT_ARGS[@]}")
FACEXFORMER_ARGS=(--data_dir /home/data --epochs "$EPOCHS" --batch_size "$BATCH_SIZE_FACEXFORMER"
    --max_samples "$MAX_SAMPLES" --max_val_samples "$MAX_VAL_SAMPLES" --lr 1e-5
    --num_workers 4 --device cuda --checkpoint "$FACEXFORMER_CHECKPOINT"
    --output /project/train/models/facexformer_finetuned.pt)
SWINFACE_ARGS=(--data_dir /home/data --epochs "$EPOCHS" --batch_size "$BATCH_SIZE_SWINFACE"
    --max_samples "$MAX_SAMPLES" --max_val_samples "$MAX_VAL_SAMPLES" --lr 1e-5
    --num_workers 4 --device cuda --keep_epoch_checkpoints --checkpoint "$SWINFACE_CHECKPOINT"
    --output /project/train/models/swinface_finetuned.pt)

# 开始任何训练前检查全部模型的数据/权重；检查不训练、不保存模型。
echo "开始三套模型的训练准备检查。"
python /project/train/src_repo/finetune_swinface.py "${SWINFACE_ARGS[@]}" --check_only
python /project/train/src_repo/finetune_facexformer.py "${FACEXFORMER_ARGS[@]}" --check_only
python /project/train/src_repo/train_head_detector.py "${HEAD_ARGS[@]}" --check_only

echo "[1/3] 开始头肩检测训练。"
python /project/train/src_repo/train_head_detector.py "${HEAD_ARGS[@]}"
echo "[2/3] 开始 FaceXFormer 训练。"
python /project/train/src_repo/finetune_facexformer.py "${FACEXFORMER_ARGS[@]}"
echo "[3/3] 开始 SwinFace 训练。"
python /project/train/src_repo/finetune_swinface.py "${SWINFACE_ARGS[@]}"

for name in headshoulder_finetuned.pt facexformer_finetuned.pt swinface_finetuned.pt; do
    if [ ! -s "/project/train/models/$name" ]; then
        echo "训练完成检查失败，未生成模型: $name" >&2
        exit 1
    fi
done

echo "训练完成，头肩检测与两套属性权重已保存至 /project/train/models/"
