"""模型榜 VOC 数据集加载。

支持数据集 A 的大图 VOC 标注，以及数据集 B 的头肩小图 VOC 标注。
每个 XML 中的 object 应包含 name、xmin/ymin/xmax/ymax；属性支持赛题实际使用的
attributes/attribute/name+value 结构，也兼容直接属性节点。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

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
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.transform = transform
        self.logger = logger or (lambda message: print(message, flush=True))

        self.image_dir = self.data_dir / "images"
        self.label_dir = self.data_dir / "labels"

        # 文件名 -> 路径 的索引，懒加载，避免对每个 XML 都遍历一次整个目录树。
        self._image_index: Optional[dict] = None
        self._bad_image_count = 0

        self.samples: list[tuple[str, dict]] = []
        self._load_samples()

    def _build_image_index(self) -> dict:
        """遍历目录树一次，建立 ``小写文件名 -> 路径`` 索引。"""
        index: dict = {}
        for root, _dirs, files in os.walk(self.data_dir):
            for file_name in files:
                if file_name.lower().endswith(IMAGE_EXTENSIONS):
                    index.setdefault(file_name.lower(), Path(root) / file_name)
        return index

    @property
    def image_index(self) -> dict:
        if self._image_index is None:
            self._image_index = self._build_image_index()
            self.logger(f"已建立图片索引: {len(self._image_index)} 个文件")
        return self._image_index

    def _load_samples(self) -> None:
        """扫描数据集，加载样本列表。"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.data_dir}")

        xml_paths = sorted(self.label_dir.rglob("*.xml")) if self.label_dir.exists() else []
        if not xml_paths:
            xml_paths = sorted(self.data_dir.rglob("*.xml"))
        self.logger(f"发现标注文件: {len(xml_paths)} 个，开始解析 ...")

        skipped = 0
        total_xml = len(xml_paths)
        for position, xml_path in enumerate(xml_paths, start=1):
            if position % 1000 == 0 or position == total_xml:
                self.logger(
                    f"解析标注进度: {position}/{total_xml}，"
                    f"有效目标 {len(self.samples)}，跳过 {skipped}"
                )
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                skipped += 1
                continue
            image_path = self._image_path(xml_path, root)
            if image_path is None:
                skipped += 1
                continue
            for label in self._parse_xml(root):
                self.samples.append((str(image_path), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"数据集为空: {self.data_dir}")

    def _image_path(self, xml_path: Path, root: ET.Element) -> Optional[Path]:
        """根据 XML 中的 filename 定位图片，找不到时回退到同名文件。"""
        filename = (root.findtext("filename") or "").strip()

        if filename:
            name = Path(filename).name
            for candidate in (
                self.image_dir / name,
                self.data_dir / name,
                xml_path.parent / name,
                self.data_dir / filename,
            ):
                if candidate.exists():
                    return candidate
            # 图片目录结构与标注不一致时，查一次索引即可，不再对每个样本做 rglob。
            return self.image_index.get(name.lower())

        for extension in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png"):
            candidate = xml_path.with_suffix(extension)
            if candidate.exists():
                return candidate
        return None

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

    def _parse_xml(self, root: ET.Element) -> list[dict]:
        """解析已读取的 VOC XML 根节点，并跳过无法用于监督学习的脏标注。"""
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
        if image is None:
            # 平台数据集偶发的损坏图片：返回零图 + 全缺失标签，
            # 该样本在 age 与各分类分支上都会被 mask 掉，不会污染训练。
            self._bad_image_count += 1
            if self._bad_image_count <= 10:
                self.logger(f"警告: 无法读取图片，已忽略该样本: {img_path}")
            image_tensor = torch.zeros(3, self.input_size, self.input_size, dtype=torch.float32)
            label_tensor = {"age": torch.tensor(float(MISSING_LABEL), dtype=torch.float32)}
            label_tensor.update(
                {
                    key: torch.tensor(MISSING_LABEL, dtype=torch.long)
                    for key in ATTRIBUTE_NAMES
                    if key != "age"
                }
            )
            return image_tensor, label_tensor

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x_min, y_min, x_max, y_max = label["bbox"]
        crop = image[max(0, y_min):max(y_min + 1, y_max), max(0, x_min):max(x_min + 1, x_max)]
        if crop.size == 0:
            crop = image
        image = cv2.resize(crop, (self.input_size, self.input_size))

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
