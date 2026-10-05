"""FaceXFormer/SwinFace 接入极市 SDK 的公共接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import os
import sys

import numpy as np


UNKNOWN_ATTRIBUTE = "-1"
BBox = Tuple[int, int, int, int]
Detector = Callable[[np.ndarray], List["FaceDetection"]]


@dataclass
class FaceDetection:
    """检测器输出，坐标格式为 ``(x, y, width, height)``。"""

    head_bbox: Optional[BBox]
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

    def predict_batch(self, face_crops: List[np.ndarray]) -> List[AttributePrediction]:
        """兼容只实现单目标推理的适配器。"""
        return [self.predict(crop) for crop in face_crops]


class FaceXFormerAdapter(AttributeModelAdapter):
    """FaceXFormer 接口位置：负责朝向、年龄、性别、人种等字段。"""

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._analyzer = _build_analyzer(model_path, None, device)

    def predict(self, face_crop: np.ndarray) -> AttributePrediction:
        self._analyzer._ensure_facexformer()
        result = self._analyzer._run_facexformer(face_crop)
        return self._decode(result)

    def predict_batch(self, face_crops: List[np.ndarray]) -> List[AttributePrediction]:
        if not face_crops:
            return []
        self._analyzer._ensure_facexformer()
        return [self._decode(result) for result in
                self._analyzer._run_facexformer_batch(face_crops)]

    @staticmethod
    def _decode(result: Dict[str, Any]) -> AttributePrediction:
        return AttributePrediction(
            toward=_toward_value(result.get("orientation")),
            gender=_gender_value(result.get("gender")),
            race=_race_value(result.get("race")),
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
        return self._decode(result)

    def predict_batch(self, face_crops: List[np.ndarray]) -> List[AttributePrediction]:
        if not face_crops:
            return []
        self._analyzer._ensure_swinface()
        return [self._decode(result) for result in
                self._analyzer._run_swinface_batch(face_crops)]

    @staticmethod
    def _decode(result: Dict[str, Any]) -> AttributePrediction:
        return AttributePrediction(
            glasses=(_class_value(result["glasses_class"], 3) if "glasses_class" in result else
                     _binary_value(result.get("eyeglasses"), positive="1")),
            emotion=_emotion_value(result.get("expression")),
            hat=_binary_value(result.get("wearing_hat"), positive="1"),
            whiskers=_mustache_value(result),
            age=_age_value(result.get("age")),
            mask=_binary_value(result.get("mask"), positive="1"),
        )


@dataclass
class _TemporalEntry:
    bbox: BBox
    anchor_bbox: BBox
    signature: np.ndarray
    prediction: AttributePrediction
    reused: int = 0


def _box_iou(first: BBox, second: BBox) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _image_signature(image: np.ndarray) -> np.ndarray:
    # Average four regularly spaced samples per cell. Only 64x64 pixels are
    # gathered, without converting a full 1080P frame to float or adding a dependency.
    height, width = image.shape[:2]
    rows = np.minimum(((np.arange(64) + 0.5) * height / 64).astype(np.intp), height - 1)
    columns = np.minimum(((np.arange(64) + 0.5) * width / 64).astype(np.intp), width - 1)
    samples = image[rows[:, None], columns[None, :]].astype(np.float32)
    return samples.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3))


def _similar_crop(current: np.ndarray, anchor: np.ndarray) -> bool:
    # Uniform/very blurred crops cannot distinguish people reliably.
    if min(current.std(axis=(0, 1)).mean(), anchor.std(axis=(0, 1)).mean()) < 8:
        return False
    difference = np.abs(current - anchor)
    # Also check the upper region separately so shoulder/background pixels do
    # not hide a change in the face, orientation, glasses or expression.
    return bool(difference.mean() <= 3 and difference[:20].mean() <= 3
                and np.percentile(difference[:20], 90) <= 10)


class FaceAttributeRuntime:
    """检测、两个属性模型及结果融合的 SDK 运行时。"""

    def __init__(
        self,
        detector: Optional[Detector] = None,
        facexformer: Optional[AttributeModelAdapter] = None,
        swinface: Optional[AttributeModelAdapter] = None,
        attribute_batch_size: Optional[int] = None,
        temporal_reuse: bool = False,
        temporal_max_reuse: int = 2,
    ) -> None:
        self.detector = detector
        self.facexformer = facexformer
        self.swinface = swinface
        self.attribute_batch_size = int(
            os.environ.get("SDK_ATTRIBUTE_BATCH_SIZE", "4")
            if attribute_batch_size is None else attribute_batch_size
        )
        if self.attribute_batch_size < 1:
            raise ValueError("SDK_ATTRIBUTE_BATCH_SIZE 必须为正整数")
        self.temporal_reuse = temporal_reuse
        self.temporal_max_reuse = int(temporal_max_reuse)
        if self.temporal_max_reuse < 1:
            raise ValueError("SDK_TEMPORAL_MAX_REUSE 必须为正整数")
        self.temporal_stats = {"frames": 0, "targets": 0, "reused": 0, "fresh": 0}
        self._temporal_entries = []  # type: List[_TemporalEntry]
        self._temporal_scene = None
        self._temporal_shape = None
        self._temporal_sequence = None

    def reset_temporal_cache(self) -> None:
        """已知序列切换时清空缓存；不改变模型和累计统计。"""
        self._temporal_entries = []
        self._temporal_scene = None
        self._temporal_shape = None
        self._temporal_sequence = None

    def _match_temporal(self, image, boxes, signatures, sequence_id):
        scene = _image_signature(image)
        sequence_changed = sequence_id != self._temporal_sequence
        if (sequence_changed or image.shape != self._temporal_shape
                or self._temporal_scene is None
                or np.abs(scene - self._temporal_scene).mean() > 12):
            self.reset_temporal_cache()
        self._temporal_shape = image.shape
        self._temporal_scene = scene
        self._temporal_sequence = sequence_id

        previous = self._temporal_entries
        # Inference exceptions must not leave usable state from an older frame.
        self._temporal_entries = []
        overlaps = [[_box_iou(box, entry.bbox) for entry in previous] for box in boxes]
        matches = {}
        for index, row in enumerate(overlaps):
            candidates = [j for j, overlap in enumerate(row) if overlap >= 0.3]
            if len(candidates) != 1:
                continue
            j = candidates[0]
            # Both directions must be unambiguous. Detector IDs are per-frame
            # numbers and intentionally play no role in this association.
            if sum(other[j] >= 0.3 for other in overlaps) != 1:
                continue
            entry = previous[j]
            if (row[j] >= 0.9 and _box_iou(boxes[index], entry.anchor_bbox) >= 0.9
                    and entry.reused < self.temporal_max_reuse
                    and entry.prediction.toward in {"front", "back", "other"}
                    and _similar_crop(signatures[index], entry.signature)):
                matches[index] = entry
        return matches

    def _predict_crops(self, adapter, crops):
        predictions = []
        for start in range(0, len(crops), self.attribute_batch_size):
            chunk = crops[start:start + self.attribute_batch_size]
            batch_predict = getattr(adapter, "predict_batch", None)
            if len(chunk) == 1 or batch_predict is None:
                results = [adapter.predict(crop) for crop in chunk]
            else:
                results = batch_predict(chunk)
            if len(results) != len(chunk):
                raise ValueError("属性模型返回数量与输入目标数量不一致")
            predictions.extend(results)
        return predictions

    def process(self, image: np.ndarray, sequence_id=None) -> List[Dict[str, Any]]:
        """处理一帧，返回符合 ``model_data.objects`` 的目标列表。"""
        if self.detector is None:
            return []

        image_height, image_width = image.shape[:2]
        detections = self.detector(image)
        crops = []
        boxes = []
        crop_indices = {}
        for index, detection in enumerate(detections):
            if detection.head_bbox is None:
                continue
            x, y, width, height = detection.head_bbox
            x_min, y_min = max(0, x), max(0, y)
            x_max, y_max = min(image_width, x + width), min(image_height, y + height)
            if x_max <= x_min or y_max <= y_min:
                continue
            crop_indices[index] = len(crops)
            crops.append(image[y_min:y_max, x_min:x_max])
            boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

        predictions = [AttributePrediction() for _ in crops]
        matches = {}
        signatures = []
        if self.temporal_reuse:
            signatures = [_image_signature(crop) for crop in crops]
            matches = self._match_temporal(image, boxes, signatures, sequence_id)
            for i, entry in matches.items():
                predictions[i] = replace(entry.prediction)
        fresh_indices = [i for i in range(len(crops)) if i not in matches]
        if self.facexformer is not None:
            results = self._predict_crops(self.facexformer, [crops[i] for i in fresh_indices])
            for i, result in zip(fresh_indices, results):
                predictions[i].update_known(result)
        front_indices = [i for i in fresh_indices if predictions[i].toward == "front"]
        if self.swinface is not None:
            front_results = self._predict_crops(self.swinface, [crops[i] for i in front_indices])
            for i, result in zip(front_indices, front_results):
                predictions[i].update_known(result)

        if self.temporal_reuse:
            for i, prediction in enumerate(predictions):
                entry = matches.get(i)
                self._temporal_entries.append(_TemporalEntry(
                    bbox=boxes[i], anchor_bbox=entry.anchor_bbox if entry else boxes[i],
                    signature=entry.signature if entry else signatures[i],
                    prediction=replace(prediction), reused=entry.reused + 1 if entry else 0))
            stats = self.temporal_stats
            stats["frames"] += 1
            stats["targets"] += len(crops)
            stats["reused"] += len(matches)
            stats["fresh"] += len(fresh_indices)
            if stats["frames"] == 1 or stats["frames"] % 200 == 0:
                print("SDK 连续帧复用: frames={frames} targets={targets} reused={reused} "
                      "fresh={fresh}".format(**stats), flush=True)

        objects = []  # type: List[Dict[str, Any]]
        for index, detection in enumerate(detections):
            person_bbox = detection.person_bbox
            head_bbox = detection.head_bbox
            if head_bbox is None:
                if person_bbox is not None:
                    objects.append(_bbox_object(person_bbox, detection.target_id, "person"))
                continue
            if index not in crop_indices:
                continue
            prediction = predictions[crop_indices[index]]
            if prediction.toward == "back":
                # 榜单示例中 toward=back 的目标所有属性均为 -1；且 ACC 规则 2 规定
                # back 识别正确即判对，属性不参与判分，这里统一置为未知。
                prediction = AttributePrediction(toward=prediction.toward)

            if person_bbox is not None:
                objects.append(
                    _bbox_object(person_bbox, detection.target_id, "person")
                )
            head = _bbox_object(head_bbox, detection.target_id, "head")
            head.update(
                {
                    "toward": prediction.toward,
                    "glasses": prediction.glasses,
                    "gender": prediction.gender,
                    "age": prediction.age,
                    "race": prediction.race,
                    "emotion": prediction.emotion,
                    # Preserve the protocol already verified by the platform.
                    "expression": prediction.emotion,
                    "mask": prediction.mask,
                    "hat": prediction.hat,
                    "whiskers": prediction.whiskers,
                }
            )
            objects.append(head)
        return objects


def _class_value(value, classes):
    try:
        index = int(value)
    except (ValueError, TypeError):
        return UNKNOWN_ATTRIBUTE
    return str(index) if 0 <= index < classes else UNKNOWN_ATTRIBUTE


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
    sdk_root = Path(__file__).resolve().parents[1]
    candidates = [sdk_root / "src", Path("/project/ev_sdk/src"), Path("/project/src")]
    for candidate in candidates:
        if (candidate / "face_attr").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from face_attr.analyzer import AttributeAnalyzer

    if facexformer_path is None:
        facexformer_path = _optional_weight(
            "facexformer/model.pt", sdk_root / "model" / "facexformer" / "model.pt"
        )
    if swinface_path is None:
        swinface_path = _optional_weight(
            "swinface/checkpoint_step_79999_gpu_0.pt",
            sdk_root / "model" / "swinface" / "checkpoint_step_79999_gpu_0.pt",
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


def _race_value(value: Any) -> str:
    """榜单 race 只接受 0-3（黄/白/黑/印第安）与 -1。

    FaceXFormer 的 race 头是 5 类，比榜单多一类；越界的预测一律返回未知，
    避免输出榜单未定义的 ``4``。
    """
    if value is None:
        return UNKNOWN_ATTRIBUTE
    try:
        index = int(value)
    except (TypeError, ValueError):
        return UNKNOWN_ATTRIBUTE
    return str(index) if 0 <= index <= 3 else UNKNOWN_ATTRIBUTE


def _age_value(value: Any) -> str:
    if value is None:
        return UNKNOWN_ATTRIBUTE
    return str(max(0, min(100, round(float(value)))))


def _gender_value(value: Any) -> str:
    if value in {"0", "1"}:
        return value
    if value == "male":
        return "1"
    if value == "female":
        return "0"
    return UNKNOWN_ATTRIBUTE


def _toward_value(value: Any) -> str:
    if value in {"front", "back", "other"}:
        return value
    if value in {"left", "right", "up", "down"}:
        return "other"
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
