"""模型榜 VOC 数据集加载。

支持数据集 A 的大图 VOC 标注，以及数据集 B 的头肩小图 VOC 标注。
每个 XML 中的 object 应包含 name、xmin/ymin/xmax/ymax；属性支持赛题实际使用的
attributes/attribute/name+value 结构，也兼容直接属性节点。
"""

from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# 属性类别映射
ATTRIBUTE_NAMES = (
    "toward",
    "glasses",
    "gender",
    "age",
    "race",
    "emotion",
    "mask",
    "hat",
    "whiskers",
)
CLASS_COUNTS = {
    "toward": 3,
    "glasses": 3,
    "gender": 2,
    "race": 4,
    "emotion": 3,
    "mask": 2,
    "hat": 2,
    "whiskers": 2,
}
TOWARD_TO_INDEX = {"front": 0, "back": 1, "other": 2}
MISSING_LABEL = -1


class FaceAttributeDataset(Dataset):
    """人脸属性数据集。

    Args:
    data_dir: 数据集根目录；支持 images/labels 子目录，也支持图片和 XML 同目录。
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
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.data_dir}")

        xml_paths = sorted(self.label_dir.rglob("*.xml")) if self.label_dir.exists() else []
        if not xml_paths:
            xml_paths = sorted(self.data_dir.rglob("*.xml"))
        for xml_path in xml_paths:
            image_path = self._image_path(xml_path)
            if image_path is None:
                continue
            for label in self._parse_xml(xml_path):
                self.samples.append((str(image_path), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"数据集为空: {self.data_dir}")

    def _image_path(self, xml_path: Path) -> Path | None:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")
        candidates = [
            self.image_dir / filename if filename else xml_path.with_suffix(".jpg"),
            self.data_dir / filename if filename else xml_path.with_suffix(".jpg"),
            xml_path.parent / filename if filename else xml_path.with_suffix(".jpg"),
        ]
        if filename:
            candidates.extend(self.data_dir.rglob(filename))
        for extension in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            candidates.append(xml_path.with_suffix(extension))
        return next((path for path in candidates if path.exists()), None)

    @staticmethod
    def _value(root: ET.Element, name: str) -> str | None:
        for attribute in root.findall("./attributes/attribute"):
            if (attribute.findtext("name") or "").strip().lower() == name:
                value = attribute.findtext("value")
                if value is not None:
                    return value.strip()
        node = root.find(name)
        if node is None:
            node = root.find(f".//{name}")
        return node.text.strip() if node is not None and node.text else None

    def _parse_xml(self, label_path: Path) -> list[dict]:
        """解析 VOC XML，并跳过无法用于监督学习的脏标注。"""
        root = ET.parse(label_path).getroot()
        results = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name not in {"head", "face", "head_shoulder"}:
                continue
            values = {
                key: self._value(obj, key) or self._value(root, key)
                for key in ATTRIBUTE_NAMES
            }
            try:
                age = values["age"]
                values["age"] = float(age) if age is not None else float(MISSING_LABEL)
                if values["age"] != MISSING_LABEL and not 0 <= values["age"] <= 100:
                    continue
                for key in ("gender", "race", "glasses", "emotion", "mask", "hat", "whiskers"):
                    values[key] = int(values[key]) if values[key] is not None else MISSING_LABEL
                toward = values["toward"]
                values["toward"] = (
                    TOWARD_TO_INDEX.get(str(toward).lower(), MISSING_LABEL)
                    if toward is not None
                    else MISSING_LABEL
                )
            except (TypeError, ValueError):
                continue
            if any(
                values[key] != MISSING_LABEL and not 0 <= values[key] < count
                for key, count in CLASS_COUNTS.items()
            ):
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            try:
                values["bbox"] = tuple(
                    int(float(bndbox.findtext(key)))
                    for key in ("xmin", "ymin", "xmax", "ymax")
                )
            except (TypeError, ValueError):
                continue
            results.append(values)
        return results

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x_min, y_min, x_max, y_max = label["bbox"]
        image = image[max(0, y_min):max(y_min + 1, y_max), max(0, x_min):max(x_min + 1, x_max)]
        image = cv2.resize(image, (self.input_size, self.input_size))

        if self.transform is not None:
            image = self.transform(image)

        # HWC -> CHW, uint8 -> float
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        label_tensor = {"age": torch.tensor(label["age"], dtype=torch.float32)}
        label_tensor.update(
            {
                key: torch.tensor(label[key], dtype=torch.long)
                for key in ATTRIBUTE_NAMES
                if key != "age"
            }
        )
        return image_tensor, label_tensor


def get_data_dir() -> str:
    """获取极市平台数据集目录。

    优先使用环境变量 DATA_DIR，否则使用默认 /home/data。
    """
    return os.environ.get("DATA_DIR", "/home/data")
