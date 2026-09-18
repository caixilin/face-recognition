"""极市平台模型榜 Python SDK 入口。

平台先调用 ``init()``，再将 BGR 图片逐张传给 ``process_image()``。
模型权重统一从 ``/project/ev_sdk/model/`` 加载。
"""

from __future__ import annotations

import json
import numpy as np
import torch

from model_api import FaceAttributeRuntime


def init():
    """初始化检测器和属性模型，返回平台后续复用的句柄。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    del device

    # 接入模型时在这里构造检测器、FaceXFormerAdapter 和 SwinFaceAdapter。
    # 权重路径必须位于 /project/ev_sdk/model/，不要写入 ev_sdk 源码目录。
    return FaceAttributeRuntime()


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
