"""人脸属性分析 - 训练脚本。

遵循极市平台规范:
    - 代码目录:   /project/train/src_repo/
    - 模型保存:   /project/train/models/
    - 日志保存:   /project/train/log/log.txt
    - 训练结果图: /project/train/result-graphs/
    - 数据集:     /home/data/

用法:
    python train.py
"""

from __future__ import annotations

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from dataset import ATTRIBUTE_NAMES, FaceAttributeDataset, get_data_dir
from model import build_model

# 极市平台固定目录
TRAIN_DIR = "/project/train"
SRC_REPO_DIR = os.path.join(TRAIN_DIR, "src_repo")
MODELS_DIR = os.path.join(TRAIN_DIR, "models")
LOG_DIR = os.path.join(TRAIN_DIR, "log")
RESULT_GRAPHS_DIR = os.path.join(TRAIN_DIR, "result-graphs")
LOG_FILE = os.path.join(LOG_DIR, "log.txt")


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg: str) -> None:
    """同时输出到控制台与日志文件。"""
    print(msg, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, criterion: nn.CrossEntropyLoss
) -> torch.Tensor:
    """对有效标签计算交叉熵；整批标签缺失时返回零损失。"""
    valid = targets >= 0
    if not valid.any():
        return logits.sum() * 0.0
    return criterion(logits, targets)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    total_age_loss = 0.0
    attribute_losses = {name: 0.0 for name in ATTRIBUTE_NAMES if name != "age"}
    num_batches = 0

    ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    toward_loss_weight = 2.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        outputs = model(images)

        age_mask = labels["age"] >= 0
        age_loss = (
            ((outputs["age"] - labels["age"]) ** 2)[age_mask].mean()
            if age_mask.any()
            else outputs["age"].sum() * 0.0
        )
        loss = age_loss
        for name in ATTRIBUTE_NAMES:
            if name == "age":
                continue
            attribute_loss = masked_cross_entropy(outputs[name], labels[name], ce_criterion)
            if name == "toward":
                attribute_loss = toward_loss_weight * attribute_loss
            attribute_losses[name] += attribute_loss.item()
            loss = loss + attribute_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_age_loss += age_loss.item()
        num_batches += 1

    return {"loss": total_loss / num_batches, "age_loss": total_age_loss / num_batches, **{
        f"{name}_loss": value / num_batches for name, value in attribute_losses.items()
    }}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """在验证集上评估模型。"""
    model.eval()
    total_loss = 0.0
    correct = {name: 0 for name in ATTRIBUTE_NAMES if name not in {"age"}}
    total = 0

    ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    toward_loss_weight = 2.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        outputs = model(images)

        age_mask = labels["age"] >= 0
        age_loss = (
            ((outputs["age"] - labels["age"]) ** 2)[age_mask].mean()
            if age_mask.any()
            else outputs["age"].sum() * 0.0
        )
        loss = age_loss
        for name in ATTRIBUTE_NAMES:
            if name == "age":
                continue
            attribute_loss = masked_cross_entropy(outputs[name], labels[name], ce_criterion)
            if name == "toward":
                attribute_loss = toward_loss_weight * attribute_loss
            loss = loss + attribute_loss

        total_loss += loss.item()
        total += images.size(0)

        for name in correct:
            valid = labels[name] >= 0
            correct[name] += ((outputs[name].argmax(1) == labels[name]) & valid).sum().item()

    num_batches = max(1, len(dataloader))
    return {
        "loss": total_loss / num_batches,
        **{f"{name}_acc": value / total for name, value in correct.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="人脸属性分析训练")
    parser.add_argument("--data_dir", type=str, default=None, help="数据集目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input_size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda")
    parser.add_argument("--max_samples", type=int, default=None, help="限制未预划分数据集的样本数")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作进程数")
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples < 2:
        raise ValueError("--max_samples 至少为 2")
    if args.num_workers < 0:
        raise ValueError("--num_workers 必须大于等于 0")

    set_seed(args.seed)

    # 设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log(f"使用设备: {device}")

    # 数据集目录
    data_dir = args.data_dir or get_data_dir()
    log(f"数据集目录: {data_dir}")

    # 数据集支持 train/val 目录，也支持极市实际的编号目录布局。
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_dataset = FaceAttributeDataset(train_dir, input_size=args.input_size)
        val_dataset = FaceAttributeDataset(val_dir, input_size=args.input_size)
    else:
        full_dataset = FaceAttributeDataset(data_dir, input_size=args.input_size)
        if args.max_samples is not None:
            sample_count = min(args.max_samples, len(full_dataset))
            full_dataset = Subset(full_dataset, range(sample_count))
        val_size = max(1, int(len(full_dataset) * 0.2))
        train_size = len(full_dataset) - val_size
        if train_size < 1:
            raise RuntimeError("数据集样本数不足，至少需要 2 个有效标注样本")
        split_generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=split_generator
        )
    log(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # 模型
    model = build_model(input_size=args.input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        log(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"gender_acc={val_metrics['gender_acc']:.4f} "
            f"emotion_acc={val_metrics['emotion_acc']:.4f} "
            f"toward_acc={val_metrics['toward_acc']:.4f}"
        )

        # 保存最优模型到 /project/train/models/
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "input_size": args.input_size,
                },
                model_path,
            )
            log(f"保存最优模型: {model_path}")

    os.makedirs(RESULT_GRAPHS_DIR, exist_ok=True)
    log("训练完成！")


if __name__ == "__main__":
    main()
