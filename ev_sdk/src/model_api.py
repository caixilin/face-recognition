"""FaceXFormer/SwinFace 接入极市 SDK 的公共接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    expression: str = UNKNOWN_ATTRIBUTE

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
        # 在这里创建 FaceXFormer 并加载 /project/ev_sdk/model/ 下的权重。

    def predict(self, face_crop: np.ndarray) -> AttributePrediction:
        del face_crop
        raise NotImplementedError("FaceXFormer 推理尚未接入")


class SwinFaceAdapter(AttributeModelAdapter):
    """SwinFace 接口位置：负责表情、眼镜、帽子、胡须等字段。"""

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        # 在这里创建 SwinFace 并加载 /project/ev_sdk/model/ 下的权重。

    def predict(self, face_crop: np.ndarray) -> AttributePrediction:
        del face_crop
        raise NotImplementedError("SwinFace 推理尚未接入")


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