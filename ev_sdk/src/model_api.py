"""FaceXFormer/SwinFace 接入极市 SDK 的公共接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import sys

import numpy as np


UNKNOWN_ATTRIBUTE = "-1"
BBox = Tuple[int, int, int, int]
Detector = Callable[[np.ndarray], List["FaceDetection"]]


@dataclass
class FaceDetection:
    """检测器输出，坐标格式为 ``(x, y, width, height)``。"""

    head_bbox: BBox
    person_bbox: Optional[BBox] = None
    confidence: float = 1.0
    target_id: str = "1"


@dataclass
class AttributePrediction:
    """两个属性模型共享的内部预测格式。"""

    toward: str = UNKNOWN_ATTRIBUTE
    glasses: str = UNKNOWN_ATTRIBUTE
    gender: str = UNKNOWN_ATTRIBUTE
    age: str = UNKNOWN_ATTRIBUTE
    race: str = UNKNOWN_ATTRIBUTE
    emotion: str = UNKNOWN_ATTRIBUTE
    mask: str = UNKNOWN_ATTRIBUTE
    hat: str = UNKNOWN_ATTRIBUTE
    whiskers: str = UNKNOWN_ATTRIBUTE
    # SwinFace 原始标签可能包含 expression；当前赛题正式输出使用 emotion。
    def update_known(self, other: "AttributePrediction") -> None:
        """使用另一个模型的非未知结果补充或覆盖当前结果。"""
        for field_name in self.__dataclass_fields__:
            value = getattr(other, field_name)
            if value != UNKNOWN_ATTRIBUTE:
                setattr(self, field_name, value)


class AttributeModelAdapter(ABC):
    """FaceXFormer 和 SwinFace 适配器必须实现的接口。"""

    @abstractmethod
    def predict(self, face_crop: np.ndarray) -> AttributePrediction:
        """分析一张 BGR 头肩裁剪图。"""
        raise NotImplementedError


class FaceXFormerAdapter(AttributeModelAdapter):
    """FaceXFormer 接口位置：负责朝向、年龄、性别、人种等字段。"""

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._analyzer = _build_analyzer(model_path, None, device)

    def predict(self, face_crop: np.ndarray) -> AttributePrediction:
        self._analyzer._ensure_facexformer()
        result = self._analyzer._run_facexformer(face_crop)
        return AttributePrediction(
            toward=_toward_value(result.get("orientation")),
            gender=_gender_value(result.get("gender")),
            race=_string_value(result.get("race")),
        )


class SwinFaceAdapter(AttributeModelAdapter):
    """SwinFace 接口位置：负责表情、眼镜、帽子、胡须等字段。"""

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._analyzer = _build_analyzer(None, model_path, device)

    def predict(self, face_crop: np.ndarray) -> AttributePrediction:
        self._analyzer._ensure_swinface()
        result = self._analyzer._run_swinface(face_crop)
        return AttributePrediction(
            glasses=_binary_value(result.get("eyeglasses"), positive="1"),
            emotion=_emotion_value(result.get("expression")),
            hat=_binary_value(result.get("wearing_hat"), positive="1"),
            whiskers=_mustache_value(result),
            age=_age_value(result.get("age")),
        )


class FaceAttributeRuntime:
    """检测、两个属性模型及结果融合的 SDK 运行时。"""

    def __init__(
        self,
        detector: Optional[Detector] = None,
        facexformer: Optional[AttributeModelAdapter] = None,
        swinface: Optional[AttributeModelAdapter] = None,
    ) -> None:
        self.detector = detector
        self.facexformer = facexformer
        self.swinface = swinface

    def process(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """处理一帧，返回符合 ``model_data.objects`` 的目标列表。"""
        if self.detector is None:
            return []

        image_height, image_width = image.shape[:2]
        objects = []  # type: List[Dict[str, Any]]
        for detection in self.detector(image):
            x, y, width, height = detection.head_bbox
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(image_width, x + width)
            y_max = min(image_height, y + height)
            face_crop = image[y_min:y_max, x_min:x_max]

            prediction = AttributePrediction()
            if self.facexformer is not None:
                prediction.update_known(self.facexformer.predict(face_crop))
            if self.swinface is not None:
                prediction.update_known(self.swinface.predict(face_crop))

            if detection.person_bbox is not None:
                objects.append(
                    _bbox_object(detection.person_bbox, detection.target_id, "person")
                )
            head = _bbox_object(detection.head_bbox, detection.target_id, "head")
            head.update(
                {
                    "toward": prediction.toward,
                    "glasses": prediction.glasses,
                    "gender": prediction.gender,
                    "age": prediction.age,
                    "race": prediction.race,
                    "emotion": prediction.emotion,
                    "mask": prediction.mask,
                    "hat": prediction.hat,
                    "whiskers": prediction.whiskers,
                }
            )
            objects.append(head)
        return objects


def _bbox_object(bbox: BBox, target_id: str, name: str) -> Dict[str, Any]:
    x, y, width, height = bbox
    return {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
        "id": str(target_id),
        "name": name,
    }


def _build_analyzer(facexformer_path: Optional[str], swinface_path: Optional[str], device: str):
    """加载仓库内的通用分析器；平台部署时需将 src/ 一并放入 SDK 包。"""
    root = Path(__file__).resolve().parents[2]
    project_root = root.parent
    candidates = [root / "src", Path("/project/ev_sdk/src"), Path("/project/src")]
    for candidate in candidates:
        if (candidate / "face_attr").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from face_attr.analyzer import AttributeAnalyzer

    if facexformer_path is None:
        facexformer_path = _optional_weight(
            "facexformer/model.pt", project_root / "models" / "facexformer" / "model.pt"
        )
    if swinface_path is None:
        swinface_path = _optional_weight(
            "swinface/checkpoint_step_79999_gpu_0.pt",
            project_root / "models" / "swinface" / "checkpoint_step_79999_gpu_0.pt",
        )
    return AttributeAnalyzer(
        device=device,
        facexformer_weights=facexformer_path,
        swinface_weights=swinface_path,
    )


def _optional_weight(relative_name: str, local_path: Path) -> str:
    platform_path = Path("/project/ev_sdk/model") / relative_name
    if platform_path.exists():
        return str(platform_path)
    return str(local_path)


def _string_value(value: Any) -> str:
    return UNKNOWN_ATTRIBUTE if value is None else str(value)


def _age_value(value: Any) -> str:
    if value is None:
        return UNKNOWN_ATTRIBUTE
    return str(max(0, min(100, round(float(value)))))


def _gender_value(value: Any) -> str:
    if value in {"male", "female"}:
        return value
    return UNKNOWN_ATTRIBUTE


def _toward_value(value: Any) -> str:
    if value == "front":
        return "front"
    if value in {"back", "left", "right", "up", "down"}:
        return "back" if value == "back" else "other"
    return UNKNOWN_ATTRIBUTE


def _emotion_value(value: Any) -> str:
    mapping = {"angry": "0", "happy": "1", "neutral": "2"}
    return mapping.get(str(value), UNKNOWN_ATTRIBUTE)


def _binary_value(value: Any, positive: str) -> str:
    if value is None:
        return UNKNOWN_ATTRIBUTE
    return positive if float(value) >= 0.5 else "0"


def _mustache_value(result: Dict[str, Any]) -> str:
    value = result.get("mustache")
    return UNKNOWN_ATTRIBUTE if value is None else ("1" if float(value) >= 0.5 else "0")