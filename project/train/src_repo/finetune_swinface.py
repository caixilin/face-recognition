"""Fine-tune the SDK SwinFace checkpoint on the platform VOC dataset."""

from __future__ import annotations

import argparse
import math
import random
import shutil
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
from validation_metrics import ClassificationCounts, write_validation_report, require_training_classes  # noqa: E402


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
    images = images.to(device, non_blocking=device.type == "cuda")
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
    """按赛道标签统计裁剪图诊断；不包含检测、朝向门控或 FPS。"""
    model.eval()
    age_error = 0.0
    age_count = 0
    age_ignored_count = 0
    counts = {
        "glasses": ClassificationCounts(["no_glasses", "glasses", "sunglasses"]),
        "glasses_binary": ClassificationCounts(["no_glasses", "any_glasses"]),
        "hat": ClassificationCounts(["no_hat", "hat"]),
        "whiskers": ClassificationCounts(["no_whiskers", "whiskers"]),
        "emotion": ClassificationCounts(["frown", "smile", "calm"]),
        "mask": ClassificationCounts(["no_mask", "mask"]),
    }
    for images, labels in loader:
        images = prepare_images(images, device)
        labels = {key: value.to(device) for key, value in labels.items()}
        outputs = model(images)

        valid = labels["age"] != MISSING_LABEL
        age_ignored_count += int((~valid).sum())
        if valid.any():
            age_error += float((outputs["Age"].reshape(-1)[valid] - labels["age"][valid]).abs().sum())
            age_count += int(valid.sum())

        glasses_target = labels["glasses"].clone()
        glasses_target[glasses_target > 0] = 1
        eye_logits = outputs["Eyeglasses"]
        eye_prediction = (eye_logits.argmax(1) if eye_logits.shape[1] == 3 else
                          (eye_logits.softmax(1)[:, 1] >= 0.5).long())
        counts["glasses"].update(labels["glasses"].tolist(), eye_prediction.tolist())
        counts["glasses_binary"].update(glasses_target.tolist(), (eye_prediction > 0).long().tolist())
        classification_tasks = (
            ("hat", outputs["Wearing Hat"], labels["hat"]),
            ("whiskers", outputs["Mustache"], labels["whiskers"]),
        )
        for key, logits, target in classification_tasks:
            # SDK 用正类概率 >= 0.5，包括恰好 0.5 的情况。
            predicted = (logits.softmax(dim=1)[:, 1] >= 0.5).long()
            counts[key].update(target.tolist(), predicted.tolist())
        mask_target = labels.get("mask", torch.full_like(labels["hat"], MISSING_LABEL))
        mask_prediction = ((outputs["Mask"].softmax(1)[:, 1] >= 0.5).long() if "Mask" in outputs
                           else torch.full_like(mask_target, MISSING_LABEL))
        counts["mask"].update(mask_target.tolist(), mask_prediction.tolist())

        predicted = outputs["Expression"].argmax(dim=1)
        # 与 SDK 一致：SwinFace 其它表情输出为 -1，不能算作三类中的任意一类。
        decoded = torch.full_like(predicted, MISSING_LABEL)
        for platform_class, swinface_class in enumerate(EMOTION_TO_SWINFACE):
            decoded[predicted == swinface_class] = platform_class
        counts["emotion"].update(labels["emotion"].tolist(), decoded.tolist())

    details = {key: value.report() for key, value in counts.items()}
    metrics = {key: (value["accuracy"] if value["accuracy"] is not None else float("nan"))
               for key, value in details.items()}
    metrics["age_mae"] = age_error / age_count if age_count else float("nan")
    metrics["age_labeled_count"] = age_count
    metrics["age_ignored_count"] = age_ignored_count
    metrics["classification"] = details
    return metrics


def train_epoch(model, loader, optimizer, device, loss_stats=None):
    model.train()
    totals = torch.zeros(6, device=device)
    loss_names = ("age_mse", "glasses", "mask", "hat", "whiskers", "emotion")
    for step, (images, labels) in enumerate(loader, 1):
        images = prepare_images(images, device)
        labels = {key: value.to(device, non_blocking=device.type == "cuda")
                  for key, value in labels.items()}
        outputs = model(images)

        valid_age = labels["age"] != MISSING_LABEL
        age_loss = masked_mean((outputs["Age"].reshape(-1) - labels["age"]) ** 2, valid_age)

        if outputs["Eyeglasses"].shape[1] != 3 or "Mask" not in outputs:
            raise ValueError("训练必须启用眼镜三分类和口罩分类头")
        glasses_loss = masked_cross_entropy(outputs["Eyeglasses"], labels["glasses"])
        mask_loss = masked_cross_entropy(outputs["Mask"], labels["mask"])
        hat_loss = masked_cross_entropy(outputs["Wearing Hat"], labels["hat"])
        whiskers_loss = masked_cross_entropy(outputs["Mustache"], labels["whiskers"])

        # Dataset labels: 0=frown, 1=smile, 2=calm. Other (-1) is ignored.
        emotion_target = emotion_targets(labels["emotion"])
        emotion_loss = masked_cross_entropy(outputs["Expression"], emotion_target)

        loss = age_loss + glasses_loss + mask_loss + hat_loss + whiskers_loss + emotion_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        totals += torch.stack((age_loss, glasses_loss, mask_loss, hat_loss,
                               whiskers_loss, emotion_loss)).detach()
        if step % 100 == 0:
            print(f"SwinFace step={step}/{len(loader)}", flush=True)
    averages = (totals / max(1, len(loader))).cpu().tolist()
    if not all(math.isfinite(value) for value in averages):
        raise ValueError(f"训练出现非有限损失，停止并保留上一轮已保存权重: {dict(zip(loss_names, averages))}")
    if loss_stats is not None:
        loss_stats.update(zip(loss_names, averages))
    return sum(averages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_val_samples", type=int, default=MAX_VAL_SAMPLES,
                        help="验证目标数上限，0 为全部；不是平台测试集限制。")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="跳过初始权重对照；跳过后无法据此判断微调改善。")
    parser.add_argument("--keep_epoch_checkpoints", action="store_true",
                        help="额外保留每轮权重，方便比较中间轮次，避免重新训练；会增加存储占用。")
    parser.add_argument("--check_only", action="store_true",
                        help="仅检查标注类别和初始权重加载，不训练、不保存模型。")
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
    if args.max_val_samples < 0:
        parser.error("--max_val_samples 必须 >= 0")
    if args.batch_size < 2 or args.epochs < 1 or args.num_workers < 0 or args.max_samples < 0 or args.lr <= 0:
        parser.error("batch_size 必须 >= 2，epochs/lr 必须 > 0，num_workers/max_samples 必须 >= 0")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"初始权重不存在: {checkpoint_path}")

    set_seed(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = FaceAttributeDataset(args.data_dir or get_data_dir(), input_size=112)

    # 抽子集必须在 random_split 之前，否则训练/验证的 8:2 比例会被打乱。
    # select_subset 会先按来源等量分层、再整体打乱，理由同 finetune_facexformer.py。
    if args.max_samples:
        dataset.select_subset(args.max_samples)
    if len(dataset) < 3:
        raise ValueError("至少需要 3 个有效目标，才能保留验证样本并训练 BatchNorm")

    val_size = max(1, int(len(dataset) * 0.2))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    training_support = require_training_classes(dataset.samples, train_set.indices, {"glasses": 3, "mask": 2})
    for key, class_count in (("hat", 2), ("whiskers", 2), ("emotion", 3)):
        counts = [0] * class_count
        for index in train_set.indices:
            value = int(dataset.samples[index][1].get(key, MISSING_LABEL))
            if 0 <= value < class_count:
                counts[value] += 1
        training_support[key] = counts
        if not all(counts):
            print(f"提示: {key} 有未覆盖的类别，训练不能证明这些类别的识别能力。", flush=True)
    print(f"训练类别分布: {training_support}", flush=True)
    # SwinFace 的 BatchNorm1d 不能用单目标尾批训练；仅在余数为 1 时丢弃尾批。
    drop_singleton = len(train_set) % args.batch_size == 1
    if drop_singleton:
        print("训练尾批只有 1 个目标，将跳过该尾批，避免 BatchNorm 在轮末报错。", flush=True)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=drop_singleton,
    )
    eval_set = val_set
    if args.max_val_samples and len(val_set) > args.max_val_samples:
        eval_set = Subset(val_set, range(args.max_val_samples))
    val_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print(f"训练样本 {len(train_set)}，验证样本 {len(eval_set)}", flush=True)

    model = build_model(SwinFaceCfg()).to(device)
    output_path = Path(args.output)
    print(f"加载初始权重: {checkpoint_path}（存在={checkpoint_path.exists()}）", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("platform_heads") == 1:
        if checkpoint.get("glasses_classes") != ["no_glasses", "glasses", "sunglasses"] or checkpoint.get("mask_classes") != ["no_mask", "mask"]:
            raise ValueError("初始权重的平台眼镜/口罩类别顺序与 SDK 不一致")
        model.om.enable_platform_heads()
    model.backbone.load_state_dict(checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(checkpoint["state_dict_fam"])
    model.tss.load_state_dict(checkpoint["state_dict_tss"])
    model.om.load_state_dict(checkpoint["state_dict_om"])
    del checkpoint  # 参数已复制到模型，释放初始 checkpoint 占用的设备内存。
    if args.check_only:
        model.om.enable_platform_heads()
        print("SwinFace 训练准备检查通过：数据类别和权重加载正常，未启动训练。", flush=True)
        return

    report = {
        "scope": "local_crop_diagnostics_not_platform_score",
        "checkpoint": str(checkpoint_path), "split_seed": 42, "split": "target",
        "training_class_support": training_support,
        "validation_total": len(val_set), "validation_evaluated": len(eval_set),
        "max_val_samples": args.max_val_samples, "epochs": [],
        "keep_epoch_checkpoints": args.keep_epoch_checkpoints,
        "training_drop_singleton_batch": drop_singleton,
        "limitations": ["no_detection_or_toward_gate_or_fps"],
    }
    report["baseline"] = None if args.skip_baseline else evaluate(model, val_loader, device)
    if report["baseline"] is not None:
        print(f"初始权重验证: {report['baseline']}", flush=True)
    write_validation_report(output_path, report)

    # Load old heads first, then preserve their weights while extending the model.
    model.om.enable_platform_heads()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss_stats = {}
        loss = train_epoch(model, loader, optimizer, device, loss_stats=loss_stats)
        metrics = evaluate(model, val_loader, device)
        print(
            f"SwinFace epoch {epoch + 1}/{args.epochs} loss={loss:.5f} "
            f"age_mae={metrics['age_mae']:.3f} glasses_3class_acc={metrics['glasses']:.4f} "
            f"glasses_binary_acc={metrics['glasses_binary']:.4f} "
            f"hat_acc={metrics['hat']:.4f} whiskers_acc={metrics['whiskers']:.4f} "
            f"emotion_acc={metrics['emotion']:.4f} mask_acc={metrics['mask']:.4f}",
            flush=True,
        )
        print(f"  分项损失（各批均值）: {loss_stats}", flush=True)
        # 每个 epoch 结束就存一次，理由同 finetune_facexformer.py：
        # 被平台时长上限砍掉时，已完成的 epoch 不会被丢掉。
        torch.save(
            {
                "state_dict_backbone": model.backbone.state_dict(),
                "state_dict_fam": model.fam.state_dict(),
                "state_dict_tss": model.tss.state_dict(),
                "state_dict_om": model.om.state_dict(),
                "platform_heads": 1,
                "glasses_classes": ["no_glasses", "glasses", "sunglasses"],
                "mask_classes": ["no_mask", "mask"],
                "training_class_support": training_support,
                "epoch": epoch + 1,
            },
            output_path,
        )
        print(f"  checkpoint 已保存 -> {output_path}", flush=True)
        epoch_path = None
        if args.keep_epoch_checkpoints:
            epoch_path = output_path.with_name(f"{output_path.stem}.epoch_{epoch + 1:02d}{output_path.suffix}")
            # 复制刚保存的完整权重，保证该轮快照不会随下一轮参数更新而变化。
            shutil.copyfile(str(output_path), str(epoch_path))
            print(f"  本轮权重已保留 -> {epoch_path}", flush=True)
        report["epochs"].append({"epoch": epoch + 1, "loss": loss, "loss_components": loss_stats,
                                 "checkpoint": str(epoch_path) if epoch_path else None, "metrics": metrics})
        report_path = write_validation_report(output_path, report)
        print(f"  验证详情 -> {report_path}", flush=True)

    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
