"""人脸检测模块。

基于 MTCNN 检测图像中的人脸，返回人脸边界框。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from facenet_pytorch import MTCNN


@dataclass
class FaceBox:
    """单张人脸检测结果。"""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


class FaceDetector:
    """使用 MTCNN 的人脸检测器。"""

    def __init__(self, device: str = "cpu", margin: float = 0.5) -> None:
        self.margin = margin
        self._mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """检测图像中的人脸。

        Args:
            image: BGR 格式的 numpy 图像。

        Returns:
            人脸边界框列表，按置信度降序排列。
        """
        boxes, probs = self._mtcnn.detect(image)
        if boxes is None:
            return []

        results: list[FaceBox] = []
        height, width = image.shape[:2]
        for box, prob in zip(boxes, probs):
            x_min, y_min, x_max, y_max = box
            # 按 margin 比例放大边界框，保留更多上下文
            bw, bh = x_max - x_min, y_max - y_min
            x_min = max(0, x_min - bw * self.margin / 2)
            y_min = max(0, y_min - bh * self.margin / 2)
            x_max = min(width, x_max + bw * self.margin / 2)
            y_max = min(height, y_max + bh * self.margin / 2)
            results.append(FaceBox(x_min, y_min, x_max, y_max, float(prob)))

        results.sort(key=lambda f: f.confidence, reverse=True)
        return results

    def crop(self, image: np.ndarray, box: FaceBox) -> np.ndarray:
        """按边界框裁剪人脸区域。"""
        x_min, y_min = int(box.x_min), int(box.y_min)
        x_max, y_max = int(box.x_max), int(box.y_max)
        return image[y_min:y_max, x_min:x_max]
