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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    EXPRESSION_CLASSES,
    GENDER_CLASSES,
    ORIENTATION_CLASSES,
    FaceAttributeDataset,
    get_data_dir,
)
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
    total_gender_loss = 0.0
    total_expr_loss = 0.0
    total_orient_loss = 0.0
    num_batches = 0

    age_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        outputs = model(images)

        age_loss = age_criterion(outputs["age"], labels["age"])
        gender_loss = ce_criterion(outputs["gender"], labels["gender"])
        expr_loss = ce_criterion(outputs["expression"], labels["expression"])
        orient_loss = ce_criterion(outputs["orientation"], labels["orientation"])

        loss = age_loss + gender_loss + expr_loss + orient_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_age_loss += age_loss.item()
        total_gender_loss += gender_loss.item()
        total_expr_loss += expr_loss.item()
        total_orient_loss += orient_loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "age_loss": total_age_loss / num_batches,
        "gender_loss": total_gender_loss / num_batches,
        "expr_loss": total_expr_loss / num_batches,
        "orient_loss": total_orient_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """在验证集上评估模型。"""
    model.eval()
    total_loss = 0.0
    correct_gender = 0
    correct_expr = 0
    correct_orient = 0
    total = 0

    age_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        outputs = model(images)

        age_loss = age_criterion(outputs["age"], labels["age"])
        gender_loss = ce_criterion(outputs["gender"], labels["gender"])
        expr_loss = ce_criterion(outputs["expression"], labels["expression"])
        orient_loss = ce_criterion(outputs["orientation"], labels["orientation"])
        loss = age_loss + gender_loss + expr_loss + orient_loss

        total_loss += loss.item()
        total += images.size(0)

        correct_gender += (
            (outputs["gender"].argmax(1) == labels["gender"]).sum().item()
        )
        correct_expr += (
            (outputs["expression"].argmax(1) == labels["expression"]).sum().item()
        )
        correct_orient += (
            (outputs["orientation"].argmax(1) == labels["orientation"]).sum().item()
        )

    num_batches = max(1, len(dataloader))
    return {
        "loss": total_loss / num_batches,
        "gender_acc": correct_gender / total,
        "expr_acc": correct_expr / total,
        "orient_acc": correct_orient / total,
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
    args = parser.parse_args()

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

    # 数据集
    train_dataset = FaceAttributeDataset(
        os.path.join(data_dir, "train"), input_size=args.input_size
    )
    val_dataset = FaceAttributeDataset(
        os.path.join(data_dir, "val"), input_size=args.input_size
    )
    log(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
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
            f"expr_acc={val_metrics['expr_acc']:.4f} "
            f"orient_acc={val_metrics['orient_acc']:.4f}"
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
