"""人脸属性分析模块。

提供年龄、性别、表情、朝向等属性的分析接口。
当前为可扩展的骨架实现，后续可接入 FaceXFormer / SwinFace 等模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceAttributes:
    """单张人脸的属性分析结果。"""

    age: float | None = None
    gender: str | None = None
    expression: str | None = None
    orientation: str | None = None
    extra: dict[str, float] = field(default_factory=dict)


class AttributeAnalyzer:
    """人脸属性分析器。

    当前为占位实现，返回空属性。后续可在此接入具体模型：
    - FaceXFormer: 朝向、年龄、性别、种族、可见性
    - SwinFace: 表情、眼镜、帽子、胡须等
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def analyze(self, face_crop: np.ndarray) -> FaceAttributes:
        """分析单张人脸裁剪图。

        Args:
            face_crop: 人脸区域图像（BGR）。

        Returns:
            属性分析结果。
        """
        del face_crop  # 占位实现，暂不使用输入
        return FaceAttributes()
