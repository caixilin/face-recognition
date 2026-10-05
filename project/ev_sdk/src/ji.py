"""极市平台模型榜 Python SDK 入口。

平台先调用 ``init()``，再将 BGR 图片逐张传给 ``process_image()``。
优先加载平台挂载在 ``/project/train/models/`` 的微调权重，
未挂载时回退到 ``/project/ev_sdk/model/`` 的 SDK 权重。
"""

from __future__ import annotations

import json
import os
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
    finetuned_names = {
        "facexformer/model.pt": "facexformer_finetuned.pt",
        "swinface/checkpoint_step_79999_gpu_0.pt": "swinface_finetuned.pt",
        "headshoulder/model.pt": "headshoulder_finetuned.pt",
    }
    candidates = []
    if name in finetuned_names:
        candidates.append(Path("/project/train/models") / finetuned_names[name])
    for root in (Path("/project/ev_sdk/model"), Path(__file__).resolve().parents[1] / "model"):
        candidates.append(root / name)
    for path in candidates:
        if path.is_file():
            print(f"SDK 加载权重: {path}", flush=True)
            return str(path)
    raise FileNotFoundError(f"模型权重不存在: {name}；已检查: {', '.join(map(str, candidates))}")


def init():
    """初始化检测器和属性模型，返回平台后续复用的句柄。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from head_detector import HeadShoulderDetector, build_legacy_mtcnn_detector

    try:
        detector_path = _find_weight("headshoulder/model.pt")
    except FileNotFoundError:
        print("SDK 未挂载新头肩权重，继续使用已验证的原 MTCNN 人脸检测流程", flush=True)
        detector = build_legacy_mtcnn_detector(device)
    else:
        detector = HeadShoulderDetector(detector_path, device)
    facexformer = FaceXFormerAdapter(_find_weight("facexformer/model.pt"), device)
    swinface = SwinFaceAdapter(_find_weight("swinface/checkpoint_step_79999_gpu_0.pt"), device)
    temporal_reuse = os.environ.get("SDK_TEMPORAL_REUSE", "0") == "1"
    temporal_max_reuse = int(os.environ.get("SDK_TEMPORAL_MAX_REUSE", "2"))
    print(f"SDK 连续帧复用配置: enabled={temporal_reuse} max_reuse={temporal_max_reuse} "
          "每帧检测，位置和图像相似才复用属性", flush=True)
    return FaceAttributeRuntime(
        detector=detector, facexformer=facexformer, swinface=swinface,
        temporal_reuse=temporal_reuse, temporal_max_reuse=temporal_max_reuse)


@torch.no_grad()
def process_image(handle=None, input_image=None, args=None, **kwargs):
    """处理一张 BGR 图片并返回极市 JSON 协议字符串。"""
    # The model leaderboard does not document folder-boundary metadata. If a
    # caller explicitly supplies sequence_id, use it; otherwise use image gates.
    sequence_id = kwargs.get("sequence_id")
    if sequence_id is None:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = None
        if isinstance(args, dict):
            sequence_id = args.get("sequence_id")
    if not isinstance(sequence_id, (str, int)):
        sequence_id = None
    runtime = handle if isinstance(handle, FaceAttributeRuntime) else None
    objects = []
    if runtime is not None and isinstance(input_image, np.ndarray):
        objects = runtime.process(input_image, sequence_id=sequence_id)
    elif runtime is not None:
        runtime.reset_temporal_cache()

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
