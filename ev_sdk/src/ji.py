"""极市平台自动测试脚本 ji.py。

放置在 /project/ev_sdk/src/ 目录下。
平台使用测试数据集评估模型性能，模型挂载路径为 /project/ev_sdk/model/。

注意: 请确保模型加载路径与发起测试时选择的模型文件路径完全一致。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# 将 src_repo 加入路径，复用训练代码
SRC_REPO_DIR = "/project/train/src_repo"
if SRC_REPO_DIR not in sys.path:
    sys.path.insert(0, SRC_REPO_DIR)

from model import build_model  # noqa: E402

# 模型挂载路径（与发起测试时选择的模型路径保持一致）
MODEL_PATH = "/project/ev_sdk/model/best_model.pth"

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


def load_model(device: torch.device) -> torch.nn.Module:
    """加载训练好的模型。"""
    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess(image: np.ndarray, input_size: int = 112) -> torch.Tensor:
    """图像预处理。"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> dict:
    """对单张图像进行人脸属性预测。"""
    input_tensor = preprocess(image).to(device)
    outputs = model(input_tensor)

    age = float(outputs["age"].item())
    gender_idx = int(outputs["gender"].argmax(1).item())
    expr_idx = int(outputs["expression"].argmax(1).item())
    orient_idx = int(outputs["orientation"].argmax(1).item())

    return {
        "age": round(age, 1),
        "gender": GENDER_CLASSES[gender_idx],
        "expression": EXPRESSION_CLASSES[expr_idx],
        "orientation": ORIENTATION_CLASSES[orient_idx],
    }


def main() -> None:
    """极市平台自动测试入口。

    平台会调用本脚本，读取测试图像并输出 JSON 结果。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    # 测试图像目录（平台挂载）
    test_dir = "/home/data"
    results = []

    for img_path in sorted(Path(test_dir).rglob("*.jpg")) + sorted(
        Path(test_dir).rglob("*.png")
    ):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        attrs = predict(model, image, device)
        results.append({"image": str(img_path), "attributes": attrs})

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
