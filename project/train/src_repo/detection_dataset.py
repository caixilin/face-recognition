"""Full-image VOC supervision for person and head-shoulder detection (dataset A)."""

from collections import defaultdict
from pathlib import Path
import math
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _toward(obj):
    values = [node.text.strip().lower() for node in obj.findall("toward") if node.text]
    values.extend((attr.findtext("value") or "").strip().lower()
                  for attr in obj.findall("attributes/attribute")
                  if (attr.findtext("name") or "").strip().lower() == "toward")
    return values[0] if len(values) == 1 and values[0] in {"front", "back", "other"} else None


def parse_detection_objects(root):
    records = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in {"person", "head"}:
            continue
        try:
            box = tuple(float(obj.findtext("bndbox/" + key)) for key in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
            continue
        records.append({"bbox": box, "label": 1 if name == "person" else 2, "toward": _toward(obj)})
    return records


class HeadShoulderDataset(Dataset):
    def __init__(self, data_dir, max_images=0, seed=42):
        self.data_root = Path(data_dir)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"检测训练数据目录不存在: {self.data_root}")
        # Share the source-aware diagnostic image resolver, not the attribute crop loader.
        from check_detection_boxes import find_image
        index = defaultdict(list)
        for path in self.data_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                index[path.name.lower()].append(path)
        self.records, skipped = [], 0
        seen = set()
        for path in sorted(self.data_root.rglob("*.xml")):
            try:
                root = ET.parse(path).getroot()
                objects = parse_detection_objects(root)
                # Dataset B's full-crop head annotations must not teach the detector
                # to return the full camera image. Accept A's person or toward labels.
                if not any(item["label"] == 1 or item["toward"] is not None for item in objects):
                    skipped += 1
                    continue
                image_path = find_image(path, root, self.data_root, index)
                if image_path is None or str(image_path) in seen:
                    skipped += 1
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                boxes, labels, towards = [], [], []
                for item in objects:
                    x1, y1, x2, y2 = item["bbox"]
                    box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
                    if box[2] > box[0] and box[3] > box[1]:
                        boxes.append(box)
                        labels.append(item["label"])
                        towards.append(item["toward"])
                if 2 not in labels:
                    skipped += 1
                    continue
                self.records.append((str(image_path), boxes, labels, towards))
                seen.add(str(image_path))
            except (ET.ParseError, OSError, ValueError):
                skipped += 1
        random.Random(seed).shuffle(self.records)
        if max_images:
            self.records = self.records[:max_images]
        if len(self.records) < 2:
            raise RuntimeError("头肩检测至少需要两张数据 A 原图；不能只挂载数据 B 头肩裁剪图。")
        self.samples = [(row[0], {}) for row in self.records]
        counts = {"person": 0, "head": 0}
        for _, _, labels, _ in self.records:
            counts["person"] += labels.count(1)
            counts["head"] += labels.count(2)
        if not counts["person"]:
            raise ValueError("检测训练缺少 person 标注，请挂载同时含 person/head 的数据 A")
        print(f"检测训练: {len(self.records)} 张原图，标签={counts}，跳过 XML={skipped}", flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, boxes, labels, _ = self.records[index]
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div(255)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        return tensor, {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64),
                        "image_id": torch.tensor([index]),
                        "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                        "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)}


def collate_detection_batch(batch):
    return tuple(zip(*batch))
