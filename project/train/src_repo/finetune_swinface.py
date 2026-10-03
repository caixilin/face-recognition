"""Fine-tune the SDK SwinFace checkpoint on the platform VOC dataset."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

TRAIN_SRC = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_SRC.parents[1]
SWINFACE_SRC = PROJECT_ROOT / "ev_sdk" / "src" / "face_attr" / "models" / "swinface"
if str(SWINFACE_SRC) not in sys.path:
    sys.path.insert(0, str(SWINFACE_SRC))

from dataset import FaceAttributeDataset, MISSING_LABEL, get_data_dir  # noqa: E402
from model import build_model  # noqa: E402


class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size = 3
    fam_in_chans = 2112
    fam_conv_shared = False
    fam_conv_mode = "split"
    fam_channel_attention = "CBAM"
    fam_spatial_attention = None
    fam_pooling = "max"
    fam_la_num_list = [2 for _ in range(11)]
    fam_feature = "all"
    embedding_size = 512


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device)
    images = F.interpolate(images, size=(112, 112), mode="bilinear", align_corners=False)
    return (images - 0.5) / 0.5


def masked_mean(loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return loss.sum() * 0.0
    return loss[valid].mean()


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return zero for a task with no valid labels instead of NaN."""
    valid = targets != MISSING_LABEL
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid])


# 验证集最多评这么多样本，避免评估本身吃掉太多时间（它只用来观察收敛趋势）。
MAX_VAL_SAMPLES = 2000

# 训练和评估使用同一映射：榜单 0/1/2 -> SwinFace 5(angry)/3(happy)/6(neutral)。
EMOTION_TO_SWINFACE = (5, 3, 6)


def emotion_targets(labels: torch.Tensor) -> torch.Tensor:
    """转换有效表情标签，保留 -1；不修改原始标签。"""
    targets = labels.clone()
    valid = targets != MISSING_LABEL
    mapping = targets.new_tensor(EMOTION_TO_SWINFACE)
    targets[valid] = mapping[targets[valid]]
    return targets


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """在验证集上统计年龄 MAE 与各属性准确率，用于观察收敛与过拟合。"""
    model.eval()
    age_error = 0.0
    age_count = 0
    correct = {"glasses": 0.0, "hat": 0.0, "whiskers": 0.0, "emotion": 0.0}
    total = {"glasses": 0, "hat": 0, "whiskers": 0, "emotion": 0}
    for images, labels in loader:
        images = prepare_images(images, device)
        labels = {key: value.to(device) for key, value in labels.items()}
        outputs = model(images)

        valid = labels["age"] != MISSING_LABEL
        if valid.any():
            age_error += float((outputs["Age"].reshape(-1)[valid] - labels["age"][valid]).abs().sum())
            age_count += int(valid.sum())

        # 榜单眼镜有 0/1/2 三类，SwinFace 只有「是否戴眼镜」，训练时把 1/2 合并成正类，
        # 评估必须用同样的合并方式，否则算出来的准确率没有意义。
        glasses_target = labels["glasses"].clone()
        glasses_target[glasses_target > 0] = 1
        binary_tasks = (
            ("glasses", outputs["Eyeglasses"], glasses_target),
            ("hat", outputs["Wearing Hat"], labels["hat"]),
            ("whiskers", outputs["Mustache"], labels["whiskers"]),
        )
        for key, logits, target in binary_tasks:
            valid = target != MISSING_LABEL
            if valid.any():
                correct[key] += float((logits[valid].argmax(dim=1) == target[valid]).sum())
                total[key] += int(valid.sum())

        valid = labels["emotion"] != MISSING_LABEL
        if valid.any():
            predicted = outputs["Expression"][valid].argmax(dim=1)
            expected = emotion_targets(labels["emotion"][valid])
            correct["emotion"] += float((predicted == expected).sum())
            total["emotion"] += int(valid.sum())

    metrics = {key: (correct[key] / total[key] if total[key] else float("nan")) for key in correct}
    metrics["age_mae"] = age_error / age_count if age_count else float("nan")
    return metrics


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = prepare_images(images, device)
        labels = {key: value.to(device) for key, value in labels.items()}
        outputs = model(images)

        valid_age = labels["age"] != MISSING_LABEL
        age_loss = masked_mean((outputs["Age"].reshape(-1) - labels["age"]) ** 2, valid_age)

        # SwinFace 的属性头是 2 分类输出（维度为 2），必须用交叉熵而不是 BCE。
        # 榜单眼镜有 0/1/2 三类，而 SwinFace 只有「是否戴眼镜」，这里把 1/2 都当作正类。
        glasses_target = labels["glasses"].clone()
        glasses_target[glasses_target > 0] = 1
        glasses_loss = masked_cross_entropy(outputs["Eyeglasses"], glasses_target)
        hat_loss = masked_cross_entropy(outputs["Wearing Hat"], labels["hat"])
        whiskers_loss = masked_cross_entropy(outputs["Mustache"], labels["whiskers"])

        # Dataset labels: 0=frown, 1=smile, 2=calm. Other (-1) is ignored.
        emotion_target = emotion_targets(labels["emotion"])
        emotion_loss = masked_cross_entropy(outputs["Expression"], emotion_target)

        loss = age_loss + glasses_loss + hat_loss + whiskers_loss + emotion_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="总量上限，0 表示全部。多份数据集会按来源等量分层抽样后合并，"
        "保证每份数据都被覆盖；平台有训练时长上限时用它控制耗时。",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "ev_sdk" / "model" / "swinface" / "checkpoint_original.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "train" / "models" / "swinface_finetuned.pt"),
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = FaceAttributeDataset(args.data_dir or get_data_dir(), input_size=112)

    # 抽子集必须在 random_split 之前，否则训练/验证的 8:2 比例会被打乱。
    # select_subset 会先按来源等量分层、再整体打乱，理由同 finetune_facexformer.py。
    if args.max_samples:
        dataset.select_subset(args.max_samples)

    val_size = max(1, int(len(dataset) * 0.2))
    train_set, val_set = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_set = val_set
    if len(val_set) > MAX_VAL_SAMPLES:
        eval_set = Subset(val_set, range(MAX_VAL_SAMPLES))
    val_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"训练样本 {len(train_set)}，验证样本 {len(eval_set)}", flush=True)

    model = build_model(SwinFaceCfg()).to(device)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    print(f"加载初始权重: {checkpoint_path}（存在={checkpoint_path.exists()}）", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.backbone.load_state_dict(checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(checkpoint["state_dict_fam"])
    model.tss.load_state_dict(checkpoint["state_dict_tss"])
    model.om.load_state_dict(checkpoint["state_dict_om"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        print(
            f"SwinFace epoch {epoch + 1}/{args.epochs} loss={loss:.5f} "
            f"age_mae={metrics['age_mae']:.3f} glasses_acc={metrics['glasses']:.4f} "
            f"hat_acc={metrics['hat']:.4f} whiskers_acc={metrics['whiskers']:.4f} "
            f"emotion_acc={metrics['emotion']:.4f}",
            flush=True,
        )
        # 每个 epoch 结束就存一次，理由同 finetune_facexformer.py：
        # 被平台时长上限砍掉时，已完成的 epoch 不会被丢掉。
        torch.save(
            {
                "state_dict_backbone": model.backbone.state_dict(),
                "state_dict_fam": model.fam.state_dict(),
                "state_dict_tss": model.tss.state_dict(),
                "state_dict_om": model.om.state_dict(),
            },
            output_path,
        )
        print(f"  checkpoint 已保存 -> {output_path}", flush=True)

    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
