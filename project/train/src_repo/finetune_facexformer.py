"""Fine-tune the SDK FaceXFormer checkpoint on the platform VOC dataset."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

TRAIN_SRC = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_SRC.parents[1]
FACEXFORMER_SRC = PROJECT_ROOT / "ev_sdk" / "src" / "face_attr" / "models" / "facexformer"
if str(FACEXFORMER_SRC) not in sys.path:
    sys.path.insert(0, str(FACEXFORMER_SRC))

from dataset import FaceAttributeDataset, MISSING_LABEL, get_data_dir  # noqa: E402
from network import FaceXFormer  # noqa: E402
from validation_metrics import ClassificationCounts, write_validation_report, require_training_classes  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_batch(images: torch.Tensor, labels: dict, device: torch.device):
    images = images.to(device)
    images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    images = (images - images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / images.new_tensor(
        [0.229, 0.224, 0.225]
    )[None, :, None, None]
    labels = {key: value.to(device) for key, value in labels.items()}
    return images, labels


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return zero for a task with no valid labels instead of NaN."""
    valid = targets != MISSING_LABEL
    if not valid.any():
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[valid], targets[valid])


# 验证集最多评这么多样本，避免评估本身吃掉太多时间（它只用来观察收敛趋势）。
MAX_VAL_SAMPLES = 2000

@torch.no_grad()
def evaluate(model, loader, device, gender_classes=("female", "male")) -> dict:
    """在验证集上统计 toward / gender / race 准确率。

    注意这是「按本脚本训练目标」的准确率，不是榜单 ACC（榜单里 toward 判对才继续
    判属性、back/other 判对即得分）。它的用途是看收敛和过拟合，别拿它当最终分数。
    """
    model.eval()
    if tuple(gender_classes) not in (("female", "male"), ("male", "female")):
        raise ValueError("无效 gender_classes")
    counts = {
        "toward": ClassificationCounts(["front", "back", "other"]),
        "gender": ClassificationCounts(["female", "male"]),
        "race": ClassificationCounts(["asian", "white", "black", "indian"]),
    }
    for images, labels in loader:
        images, labels = prepare_batch(images, labels, device)
        _, pose_logits, _, _, _, gender_logits, race_logits, _ = model(images, None, None)

        # 与 analyzer._headpose_to_orientation 一致；角度规则仍只是双眼可见定义的近似。
        pitch, yaw = pose_logits[:, 0].abs(), pose_logits[:, 1].abs()
        predicted = torch.full_like(labels["toward"], 2)
        predicted[(pitch < np.pi / 6) & (yaw < np.pi / 6)] = 0
        predicted[yaw >= np.pi / 2] = 1
        if hasattr(model, "toward_classifier"):
            predicted = model.toward_classifier(pose_logits).argmax(1)
        counts["toward"].update(labels["toward"].tolist(), predicted.tolist())

        for key, logits in (("gender", gender_logits), ("race", race_logits)):
            predicted = logits.argmax(dim=1)
            if key == "gender" and tuple(gender_classes) == ("male", "female"):
                predicted = 1 - predicted
            counts[key].update(labels[key].tolist(), predicted.tolist())

    details = {key: value.report() for key, value in counts.items()}
    metrics = {key: (value["accuracy"] if value["accuracy"] is not None else float("nan"))
               for key, value in details.items()}
    metrics["classification"] = details
    return metrics


def run_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for images, labels in loader:
        images, labels = prepare_batch(images, labels, device)

        # tasks=None → 一次前向就把 8 个头全部拿到。
        # 原来这里按 task=4 调一次、再按 task=2 调一次，但 FaceXFormer.forward
        # 无论如何都会把整个 face_decoder 算一遍，task 只对返回值做切片，
        # 所以第二次调用等于把解码器重算了一遍（约 2 倍算力、2 倍显存峰值）。
        # 另外 labels 参数在 forward 内部从未被使用，传 None 即可。
        _, pose_logits, _, _, _, gender_logits, race_logits, _ = model(images, None, None)

        # 注意：这里把数据集的 gender/race 标签直接当作分类目标训练，
        # 因此训练完成后的权重必须与 model_api.py 的解码保持一致：
        #   gender 0=女、1=男（榜单定义）；race 0-3（榜单定义，头是 5 类，多出的一类不会被预测到）。
        # 保存时写入 gender_classes，SDK 加载后按该顺序解码；旧的原始权重仍按
        # 0=男、1=女解码。元数据与文件名无关，复制替换 model.pt 后也能保留。
        gender_loss = masked_cross_entropy(gender_logits, labels["gender"])
        race_loss = masked_cross_entropy(race_logits, labels["race"])
        loss = gender_loss + race_loss

        # Supervise actual platform classes, rather than inventing Euler-angle labels.
        loss = loss + masked_cross_entropy(model.toward_classifier(pose_logits), labels["toward"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_val_samples", type=int, default=MAX_VAL_SAMPLES,
                        help="验证目标数上限，0 为全部；不是平台测试集限制。")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="跳过初始权重对照；跳过后无法据此判断微调改善。")
    parser.add_argument("--check_only", action="store_true",
                        help="仅检查训练标签和初始权重加载，不训练、不保存模型")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="总量上限，0 表示全部。多份数据集会按来源等量分层抽样后合并，"
        "保证每份数据都被覆盖；平台有训练时长上限时用它控制耗时。",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "ev_sdk" / "model" / "facexformer" / "model_original.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "train" / "models" / "facexformer_finetuned.pt"),
    )
    args = parser.parse_args()
    if args.max_val_samples < 0:
        parser.error("--max_val_samples 必须 >= 0")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 0 or args.lr <= 0:
        parser.error("epochs/batch_size/lr 必须 > 0，num_workers/max_samples 必须 >= 0")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"初始权重不存在: {checkpoint_path}")

    set_seed(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = args.data_dir or get_data_dir()
    dataset = FaceAttributeDataset(data_dir, input_size=224)

    # 抽子集必须在 random_split 之前，否则训练/验证的 8:2 比例会被打乱。
    # select_subset 会先按来源等量分层、再整体打乱，所以抽出来的是各份数据的混合，
    # 而不是「按路径排序取前 N 个」那种只用到其中一份的退化结果。
    if args.max_samples:
        dataset.select_subset(args.max_samples)
    if len(dataset) < 2:
        raise ValueError("至少需要两个有效目标才能划分训练和验证集")

    val_size = max(1, int(len(dataset) * 0.2))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    training_support = require_training_classes(dataset.samples, train_set.indices, {"toward": 3})
    for key, count in (("gender", 2), ("race", 4)):
        counts = [0] * count
        for index in train_set.indices:
            label = int(dataset.samples[index][1].get(key, MISSING_LABEL))
            if 0 <= label < count:
                counts[label] += 1
        training_support[key] = counts
        if not any(counts):
            print(f"提示: {key} 没有训练标签，该任务不会获得监督；完整属性训练需要挂载数据 B。", flush=True)
    print(f"朝向训练类别分布: {training_support}", flush=True)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_set = val_set
    if args.max_val_samples and len(val_set) > args.max_val_samples:
        eval_set = Subset(val_set, range(args.max_val_samples))
    val_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"训练样本 {len(train_set)}，验证样本 {len(eval_set)}", flush=True)

    model = FaceXFormer().to(device)
    output_path = Path(args.output)
    print(f"加载初始权重: {checkpoint_path}（存在={checkpoint_path.exists()}）", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "toward_classes" in checkpoint:
        if checkpoint["toward_classes"] != ["front", "back", "other"]:
            raise ValueError("无效 toward_classes")
        model.toward_classifier = nn.Linear(3, 3).to(device)
    model.load_state_dict(checkpoint["state_dict_backbone"])
    if args.check_only:
        if not hasattr(model, "toward_classifier"):
            model.toward_classifier = nn.Linear(3, 3).to(device)
        print(f"FaceXFormer 训练准备检查通过: 类别分布={training_support}，未启动训练。", flush=True)
        return
    report = {
        "scope": "local_crop_diagnostics_not_platform_score",
        "checkpoint": str(checkpoint_path), "split_seed": 42, "split": "target",
        "training_class_support": training_support,
        "validation_total": len(val_set), "validation_evaluated": len(eval_set),
        "max_val_samples": args.max_val_samples, "epochs": [],
        "limitations": ["no_detection_or_toward_gate_or_fps"],
    }
    baseline_classes = checkpoint.get("gender_classes", ["male", "female"])
    report["baseline_gender_classes"] = baseline_classes
    report["baseline"] = (None if args.skip_baseline else
                          evaluate(model, val_loader, device, gender_classes=baseline_classes))
    if report["baseline"] is not None:
        print(f"初始权重验证: {report['baseline']}", flush=True)
    write_validation_report(output_path, report)
    if not hasattr(model, "toward_classifier"):
        model.toward_classifier = nn.Linear(3, 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = run_epoch(model, loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        print(
            f"FaceXFormer epoch {epoch + 1}/{args.epochs} loss={loss:.5f} "
            f"toward_acc={metrics['toward']:.4f} gender_acc={metrics['gender']:.4f} "
            f"race_acc={metrics['race']:.4f}",
            flush=True,
        )
        # 每个 epoch 结束就存一次。平台有 12 小时上限，
        # 原来只在循环结束后存一次，一旦被砍就是零产物、全部进度丢失。
        torch.save(
            {"state_dict_backbone": model.state_dict(), "gender_classes": ["female", "male"],
             "toward_classes": ["front", "back", "other"], "training_class_support": training_support},
            output_path,
        )
        print(f"  checkpoint 已保存 -> {output_path}", flush=True)
        report["epochs"].append({"epoch": epoch + 1, "metrics": metrics})
        report_path = write_validation_report(output_path, report)
        print(f"  验证详情 -> {report_path}", flush=True)

    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
