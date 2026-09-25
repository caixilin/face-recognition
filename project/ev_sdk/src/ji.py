"""极市平台模型榜 Python SDK 入口。

平台先调用 ``init()``，再将 BGR 图片逐张传给 ``process_image()``。
模型权重统一从 ``/project/ev_sdk/model/`` 加载。
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from model_api import (
    FaceDetection,
    FaceAttributeRuntime,
    FaceXFormerAdapter,
    SwinFaceAdapter,
)


def _find_weight(name: str) -> str:
    for root in (Path("/project/ev_sdk/model"), Path(__file__).resolve().parents[1] / "model"):
        path = root / name
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"模型权重不存在: {name}")


def init():
    """初始化检测器和属性模型，返回平台后续复用的句柄。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from facenet_pytorch import MTCNN

    detector_model = MTCNN(keep_all=True, device=device)

    def detector(image):
        image_rgb = image[:, :, ::-1]
        boxes, probabilities = detector_model.detect(image_rgb)
        if boxes is None:
            return []
        detections = []
        for box, probability in zip(boxes, probabilities):
            x_min, y_min, x_max, y_max = [int(value) for value in box]
            detections.append(
                FaceDetection(
                    head_bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    person_bbox=(
                        max(0, x_min - (x_max - x_min) // 2),
                        max(0, y_min - (y_max - y_min)),
                        min(image.shape[1], x_max + (x_max - x_min) // 2)
                        - max(0, x_min - (x_max - x_min) // 2),
                        min(image.shape[0], y_max + (y_max - y_min) * 2)
                        - max(0, y_min - (y_max - y_min)),
                    ),
                    confidence=float(probability),
                )
            )
        return detections

    return FaceAttributeRuntime(
        detector=detector,
        facexformer=FaceXFormerAdapter(
            _find_weight("facexformer/model.pt"), device
        ),
        swinface=SwinFaceAdapter(
            _find_weight("swinface/checkpoint_step_79999_gpu_0.pt"), device
        ),
    )


@torch.no_grad()
def process_image(handle=None, input_image=None, args=None, **kwargs):
    """处理一张 BGR 图片并返回极市 JSON 协议字符串。"""
    del args, kwargs
    runtime = handle if isinstance(handle, FaceAttributeRuntime) else None
    objects = []
    if runtime is not None and isinstance(input_image, np.ndarray):
        objects = runtime.process(input_image)

    heads = [item for item in objects if item.get("name") == "head"]
    result = {
        "algorithm_data": {
            "is_alert": bool(heads),
            "target_count": len(heads),
            "target_info": heads,
        },
        "model_data": {"objects": objects},
    }
    return json.dumps(result, ensure_ascii=False)
