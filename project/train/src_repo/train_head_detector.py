"""Train a genuine person/head-shoulder detector on platform dataset A."""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SDK_SRC = Path(__file__).resolve().parents[2] / "ev_sdk" / "src"
sys.path.insert(0, str(SDK_SRC))
from head_detector import ARCHITECTURE, DETECTOR_CLASSES, build_head_detector, load_head_detector
from check_detection_boxes import box_iou
from detection_dataset import HeadShoulderDataset, collate_detection_batch
from validation_metrics import write_validation_report, split_image_indices


@torch.no_grad()
def evaluate(model, loader, device, score_threshold=0.5):
    """One-to-one matching at IoU .5 for diagnosis; not official platform ACC."""
    model.eval()
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in DETECTOR_CLASSES[1:]}
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for output, target in zip(outputs, targets):
            for label, name in enumerate(DETECTOR_CLASSES[1:], 1):
                gt = target["boxes"][target["labels"] == label].tolist()
                valid = (output["labels"] == label) & (output["scores"] >= score_threshold)
                boxes, scores = output["boxes"][valid].cpu().tolist(), output["scores"][valid].cpu().tolist()
                used = set()
                for index in sorted(range(len(boxes)), key=lambda index: -scores[index]):
                    candidates = [(box_iou(boxes[index], box), number)
                                  for number, box in enumerate(gt) if number not in used]
                    iou, match = max(candidates, default=(0, -1))
                    if iou >= 0.5:
                        used.add(match)
                        counts[name]["tp"] += 1
                    else:
                        counts[name]["fp"] += 1
                counts[name]["fn"] += len(gt) - len(used)
    for row in counts.values():
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else None
        row["recall"] = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else None
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/home/data")
    parser.add_argument("--output", default="/project/train/models/headshoulder_finetuned.pt")
    parser.add_argument("--checkpoint", help="继续训练本项目头肩权重")
    parser.add_argument("--coco_checkpoint", help="离线 COCO 初始化 .pth（不是头肩成品权重）")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=20000)
    parser.add_argument("--max_val_images", type=int, default=500)
    parser.add_argument("--min_size", type=int, default=640)
    parser.add_argument("--max_size", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep_epoch_checkpoints", action="store_true",
                        help="额外保留每轮完整检测权重，避免中间轮次被覆盖")
    parser.add_argument("--check_only", action="store_true",
                        help="仅检查数据划分和初始权重加载，不训练、不保存模型")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.min_size) < 1 or args.max_size < args.min_size:
        parser.error("epochs/batch_size/min_size 必须 > 0，max_size 必须 >= min_size")
    if min(args.max_images, args.max_val_images, args.num_workers) < 0:
        parser.error("样本上限和 num_workers 必须 >= 0")
    if args.lr <= 0:
        parser.error("lr 必须 > 0")
    for path in (args.checkpoint, args.coco_checkpoint):
        if path and not Path(path).is_file():
            parser.error(f"初始化权重不存在: {path}")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = HeadShoulderDataset(args.data_dir, args.max_images)
    train_ids, val_ids = split_image_indices(dataset.samples)
    validation_total = len(val_ids)
    if args.max_val_images:
        val_ids = val_ids[:args.max_val_images]
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       collate_fn=collate_detection_batch, pin_memory=device.type == "cuda",
                       persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(Subset(dataset, train_ids), shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(dataset, val_ids), shuffle=False, **loader_args)
    if args.checkpoint:
        print(f"继续训练已有头肩权重: {args.checkpoint}", flush=True)
        model = load_head_detector(args.checkpoint, device)
        args.min_size, args.max_size = model.transform.min_size[0], model.transform.max_size
    else:
        model = build_head_detector(args.min_size, args.max_size, pretrained=True,
                                    coco_path=args.coco_checkpoint).to(device)
    print(f"头肩数据划分: 训练={len(train_ids)} 验证={validation_total} 实际评估={len(val_ids)} max_images={args.max_images}", flush=True)
    if args.check_only:
        print("头肩训练准备检查通过：数据和权重加载正常，未启动训练。", flush=True)
        return
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "detection_diagnostics_not_platform_score", "split": "image", "seed": 42,
              "train_images": len(train_ids), "validation_images": len(val_ids),
              "validation_total": validation_total, "max_images": args.max_images,
              "initial_checkpoint": args.checkpoint or args.coco_checkpoint,
              "keep_epoch_checkpoints": args.keep_epoch_checkpoints,
              "baseline": evaluate(model, val_loader, device), "epochs": []}
    write_validation_report(output, report)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, (images, targets) in enumerate(train_loader, 1):
            losses = model([image.to(device, non_blocking=device.type == "cuda") for image in images],
                           [{key: value.to(device, non_blocking=device.type == "cuda")
                             for key, value in target.items()} for target in targets])
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"检测训练 loss 非有限值: {losses}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            total_loss += float(loss.detach())
            if step % 50 == 0:
                print(f"HeadShoulder epoch={epoch + 1} step={step}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        torch.save({"architecture": ARCHITECTURE, "classes": DETECTOR_CLASSES,
                    "min_size": args.min_size, "max_size": args.max_size,
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict()}, output)
        epoch_path = None
        if args.keep_epoch_checkpoints:
            epoch_path = output.with_name(f"{output.stem}.epoch_{epoch + 1:02d}{output.suffix}")
            shutil.copyfile(str(output), str(epoch_path))
        metrics = evaluate(model, val_loader, device)
        report["epochs"].append({"epoch": epoch + 1, "loss": total_loss / len(train_loader),
                                 "checkpoint": str(epoch_path) if epoch_path else None, "metrics": metrics})
        write_validation_report(output, report)
        print(f"HeadShoulder epoch {epoch + 1}/{args.epochs}: {metrics} saved={output}", flush=True)


if __name__ == "__main__":
    main()
