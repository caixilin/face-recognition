"""Fine-tune the SDK FaceXFormer checkpoint on the platform VOC dataset."""

from __future__ import annotations

import argparse
import os
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

# toward 的训练目标：front→(pitch 0, yaw 0)、back→yaw 3.14、other→(pitch 0.8, yaw 0.8)。
# 评估时反过来用「预测的欧拉角离哪个目标最近」判朝向，与训练目标严格对应。
TOWARD_POSE_TARGETS = ((0.0, 0.0), (0.0, 3.14), (0.8, 0.8))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """在验证集上统计 toward / gender / race 准确率。

    注意这是「按本脚本训练目标」的准确率，不是榜单 ACC（榜单里 toward 判对才继续
    判属性、back/other 判对即得分）。它的用途是看收敛和过拟合，别拿它当最终分数。
    """
    model.eval()
    pose_targets = torch.tensor(TOWARD_POSE_TARGETS, device=device, dtype=torch.float32)
    correct = {"toward": 0.0, "gender": 0.0, "race": 0.0}
    total = {"toward": 0, "gender": 0, "race": 0}
    for images, labels in loader:
        images, labels = prepare_batch(images, labels, device)
        _, pose_logits, _, _, _, gender_logits, race_logits, _ = model(images, None, None)

        valid = labels["toward"] != MISSING_LABEL
        if valid.any():
            # 只取 pitch/yaw 两维，与训练目标的编码保持一致。
            distances = torch.cdist(pose_logits[valid][:, :2], pose_targets)
            correct["toward"] += float((distances.argmin(dim=1) == labels["toward"][valid]).sum())
            total["toward"] += int(valid.sum())

        for key, logits in (("gender", gender_logits), ("race", race_logits)):
            valid = labels[key] != MISSING_LABEL
            if valid.any():
                correct[key] += float((logits[valid].argmax(dim=1) == labels[key][valid]).sum())
                total[key] += int(valid.sum())

    return {
        key: (correct[key] / total[key] if total[key] else float("nan")) for key in correct
    }


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
        # toward 的欧拉角目标同样必须与 face_attr/analyzer._headpose_to_orientation 的阈值一致：
        #   front→(0, 0)、back→yaw≈3.14、other→yaw≈0.8 且 pitch≈0.8。
        gender_loss = masked_cross_entropy(gender_logits, labels["gender"])
        race_loss = masked_cross_entropy(race_logits, labels["race"])
        loss = gender_loss + race_loss

        valid_toward = labels["toward"] != MISSING_LABEL
        if valid_toward.any():
            # 不要用 torch.where(cond, 3.14, torch.where(cond2, 0.8, 0.0))。
            # torch 1.11 上 "标量 + 张量" 混用不会把标量转成张量的 dtype，
            # 会直接报 "expected scalar type double but found float"
            # （已实测：torch.where(t == 1, 3.14, 0.0) 正常，
            #   但 torch.where(t == 1, 3.14, 某个 float32 张量) 必报错）。
            # masked_fill 把填充值直接写进目标张量，全程只有张量，没有这个歧义。
            toward = labels["toward"]
            headpose = torch.zeros(images.shape[0], 3, device=device, dtype=torch.float32)
            headpose[:, 1] = headpose[:, 1].masked_fill(toward == 1, 3.14)
            headpose[:, 1] = headpose[:, 1].masked_fill(toward == 2, 0.8)
            headpose[:, 0] = headpose[:, 0].masked_fill(toward == 2, 0.8)
            loss = loss + 0.5 * nn.functional.smooth_l1_loss(pose_logits[valid_toward], headpose[valid_toward])

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

    set_seed(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = args.data_dir or get_data_dir()
    dataset = FaceAttributeDataset(data_dir, input_size=224)

    # 抽子集必须在 random_split 之前，否则训练/验证的 8:2 比例会被打乱。
    # select_subset 会先按来源等量分层、再整体打乱，所以抽出来的是各份数据的混合，
    # 而不是「按路径排序取前 N 个」那种只用到其中一份的退化结果。
    if args.max_samples:
        dataset.select_subset(args.max_samples)

    val_size = max(1, int(len(dataset) * 0.2))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_set = val_set
    if len(val_set) > MAX_VAL_SAMPLES:
        eval_set = Subset(val_set, range(MAX_VAL_SAMPLES))
    val_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"训练样本 {len(train_set)}，验证样本 {len(eval_set)}", flush=True)

    model = FaceXFormer().to(device)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    print(f"加载初始权重: {checkpoint_path}（存在={checkpoint_path.exists()}）", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict_backbone"])
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
            {"state_dict_backbone": model.state_dict(), "gender_classes": ["female", "male"]},
            output_path,
        )
        print(f"  checkpoint 已保存 -> {output_path}", flush=True)

    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
