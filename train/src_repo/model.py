"""人脸属性分析 - 模型定义。

基于 PyTorch 1.11 的轻量级人脸属性分类模型。
输入: 头肩裁剪图 (3, H, W)
输出: 模型榜要求的九项属性
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FaceAttributeNet(nn.Module):
    """轻量级人脸属性分析网络。

    使用简单的卷积骨干网络，分别输出年龄、性别、表情、朝向。
    """

    def __init__(
        self,
        num_expression: int = 3,
        num_orientation: int = 3,
        input_size: int = 112,
    ) -> None:
        super().__init__()
        self.input_size = input_size

        # 卷积骨干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 256

        # 各属性分支。类别数与模型榜标签定义保持一致。
        self.age_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )
        self.gender_head = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
        self.expression_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_expression),
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_orientation),
        )
        self.glasses_head = nn.Linear(feat_dim, 3)
        self.race_head = nn.Linear(feat_dim, 4)
        self.emotion_head = nn.Linear(feat_dim, 3)
        self.mask_head = nn.Linear(feat_dim, 2)
        self.hat_head = nn.Linear(feat_dim, 2)
        self.whiskers_head = nn.Linear(feat_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x: 输入图像张量 (B, 3, H, W)。

        Returns:
            各属性分支的输出字典。
        """
        feat = self.pool(self.backbone(x)).flatten(1)

        return {
            "age": self.age_head(feat).squeeze(-1),
            "gender": self.gender_head(feat),
            "toward": self.orientation_head(feat),
            "glasses": self.glasses_head(feat),
            "race": self.race_head(feat),
            "emotion": self.emotion_head(feat),
            "mask": self.mask_head(feat),
            "hat": self.hat_head(feat),
            "whiskers": self.whiskers_head(feat),
        }


def build_model(
    num_expression: int = 3,
    num_orientation: int = 3,
    input_size: int = 112,
) -> FaceAttributeNet:
    """构建模型。"""
    return FaceAttributeNet(
        num_expression=num_expression,
        num_orientation=num_orientation,
        input_size=input_size,
    )
