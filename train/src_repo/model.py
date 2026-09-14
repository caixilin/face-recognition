"""人脸属性分析 - 模型定义。

基于 PyTorch 1.11 的轻量级人脸属性分类模型。
输入: 人脸裁剪图 (3, H, W)
输出: 年龄(回归)、性别(二分类)、表情(多分类)、朝向(多分类)
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
        num_expression: int = 7,
        num_orientation: int = 5,
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
            nn.MaxPool2d(2),  # 7
        )

        feat_dim = 256 * 7 * 7

        # 各属性分支
        self.age_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 2)
        )
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x: 输入图像张量 (B, 3, H, W)。

        Returns:
            各属性分支的输出字典。
        """
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)

        return {
            "age": self.age_head(feat).squeeze(-1),
            "gender": self.gender_head(feat),
            "expression": self.expression_head(feat),
            "orientation": self.orientation_head(feat),
        }


def build_model(
    num_expression: int = 7,
    num_orientation: int = 5,
    input_size: int = 112,
) -> FaceAttributeNet:
    """构建模型。"""
    return FaceAttributeNet(
        num_expression=num_expression,
        num_orientation=num_orientation,
        input_size=input_size,
    )
