"""人脸属性分析 - 数据集加载。

支持从极市平台数据集目录 /home/data 加载人脸属性数据。
数据集目录结构（示例）:
    /home/data/xxx/
        images/xxx.jpg
        labels/xxx.txt   # 每行: age gender expression orientation
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# 属性类别映射
GENDER_CLASSES = ["male", "female"]
EXPRESSION_CLASSES = [
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "angry",
]
ORIENTATION_CLASSES = ["front", "left", "right", "up", "down"]


class FaceAttributeDataset(Dataset):
    """人脸属性数据集。

    Args:
        data_dir: 数据集根目录，包含 images/ 与 labels/。
        input_size: 输入图像尺寸。
        transform: 数据增强/预处理函数。
    """

    def __init__(
        self,
        data_dir: str,
        input_size: int = 112,
        transform=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.transform = transform

        self.image_dir = self.data_dir / "images"
        self.label_dir = self.data_dir / "labels"

        self.samples: list[tuple[str, dict]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """扫描数据集，加载样本列表。"""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")

        for img_path in sorted(self.image_dir.glob("*.jpg")) + sorted(
            self.image_dir.glob("*.png")
        ):
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            label = self._parse_label(label_path)
            self.samples.append((str(img_path), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"数据集为空: {self.data_dir}")

    def _parse_label(self, label_path: Path) -> dict:
        """解析标签文件。"""
        with open(label_path, "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
        # 格式: age gender expression orientation
        age = float(parts[0])
        gender = int(parts[1])
        expression = int(parts[2])
        orientation = int(parts[3])
        return {
            "age": age,
            "gender": gender,
            "expression": expression,
            "orientation": orientation,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size))

        if self.transform is not None:
            image = self.transform(image)

        # HWC -> CHW, uint8 -> float
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        label_tensor = {
            "age": torch.tensor(label["age"], dtype=torch.float32),
            "gender": torch.tensor(label["gender"], dtype=torch.long),
            "expression": torch.tensor(label["expression"], dtype=torch.long),
            "orientation": torch.tensor(label["orientation"], dtype=torch.long),
        }
        return image_tensor, label_tensor


def get_data_dir() -> str:
    """获取极市平台数据集目录。

    优先使用环境变量 DATA_DIR，否则使用默认 /home/data。
    """
    return os.environ.get("DATA_DIR", "/home/data")
