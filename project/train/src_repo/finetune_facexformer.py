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
from torch.utils.data import DataLoader, random_split

TRAIN_SRC = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_SRC.parents[1]
FACEXFORMER_SRC = PROJECT_ROOT / "ev_sdk" / "src" / "face_attr" / "models" / "facexformer"
if str(FACEXFORMER_SRC) not in sys.path:
    sys.path.insert(0, str(FACEXFORMER_SRC))

from dataset import FaceAttributeDataset, MISSING_LABEL, get_data_dir  # noqa: E402
from network import FaceXFormer  # noqa: E402

TASK_HEADPOSE = 2


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


def labels_for_model(labels: dict, device: torch.device) -> dict:
    batch_size = labels["age"].shape[0]
    model_labels = {
        "segmentation": torch.zeros(batch_size, 224, 224, device=device),
        "lnm_seg": torch.zeros(batch_size, 5, 2, device=device),
        "landmark": torch.zeros(batch_size, 68, 2, device=device),
        "headpose": torch.zeros(batch_size, 3, device=device),
        "attribute": torch.zeros(batch_size, 40, device=device),
        "a_g_e": torch.zeros(batch_size, 3, device=device),
        "visibility": torch.zeros(batch_size, 29, device=device),
    }
    return model_labels


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return zero for a task with no valid labels instead of NaN."""
    valid = targets != MISSING_LABEL
    if not valid.any():
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[valid], targets[valid])


def run_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for images, labels in loader:
        images, labels = prepare_batch(images, labels, device)
        model_labels = labels_for_model(labels, device)
        task = torch.full((images.shape[0],), 4, device=device, dtype=torch.long)
        _, _, _, _, _, gender_logits, race_logits, _ = model(images, model_labels, task)

        gender_loss = masked_cross_entropy(gender_logits, labels["gender"])
        race_loss = masked_cross_entropy(race_logits, labels["race"])
        loss = gender_loss + race_loss

        valid_toward = labels["toward"] != MISSING_LABEL
        if valid_toward.any():
            headpose = torch.zeros(images.shape[0], 3, device=device)
            headpose[:, 1] = torch.where(labels["toward"] == 1, 3.14, torch.where(labels["toward"] == 2, 0.8, 0.0))
            headpose[:, 0] = torch.where(labels["toward"] == 2, 0.8, 0.0)
            headpose_targets = {key: value for key, value in model_labels.items()}
            headpose_targets["headpose"] = headpose
            _, pose_logits, _, _, _, _, _, _ = model(images, headpose_targets, torch.full_like(task, TASK_HEADPOSE))
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
    val_size = max(1, int(len(dataset) * 0.2))
    train_set, _ = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = FaceXFormer().to(device)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict_backbone"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        print(f"FaceXFormer epoch {epoch + 1}/{args.epochs} loss={run_epoch(model, loader, optimizer, device):.5f}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict_backbone": model.state_dict()}, output_path)
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
