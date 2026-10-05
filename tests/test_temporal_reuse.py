"""连续帧属性复用：刷新、关联歧义和输出协议的定向检查。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "project" / "ev_sdk" / "src"))
from model_api import AttributeModelAdapter, AttributePrediction, FaceAttributeRuntime, FaceDetection
from test_sdk_interface import _load_ji


class Adapter(AttributeModelAdapter):
    def __init__(self, swin=False):
        self.swin = swin
        self.calls = []
        self.toward = "front"

    def predict(self, crop):
        return self.predict_batch([crop])[0]

    def predict_batch(self, crops):
        markers = [int(crop[0, 0, 0]) for crop in crops]
        self.calls.append(markers)
        if self.swin:
            return [AttributePrediction(age=str(20 + marker), mask="1", emotion="2")
                    for marker in markers]
        return [AttributePrediction(toward=self.toward, gender="1", race="0") for _ in crops]


def image_and_boxes(count=1):
    image = np.zeros((80, count * 70 + 20, 3), dtype=np.uint8)
    boxes = [(10 + i * 70, 10, 40, 40) for i in range(count)]
    rng = np.random.RandomState(7)
    for marker, (x, y, w, h) in enumerate(boxes):
        crop = rng.randint(10, 220, (h, w, 3)).astype(np.uint8)
        crop[:, :, 0] = marker
        image[y:y+h, x:x+w] = crop
    return image, boxes


def make_runtime(boxes, **kwargs):
    detections = [FaceDetection(box, target_id=str(i)) for i, box in enumerate(boxes)]
    detector_calls = []

    def detector(image):
        detector_calls.append(image.shape)
        return detections

    fx, sw = Adapter(), Adapter(True)
    runtime = FaceAttributeRuntime(detector, fx, sw, temporal_reuse=True, **kwargs)
    return runtime, fx, sw, detections, detector_calls


def test_reuse_keeps_every_frame_detection_current_boxes_and_periodic_refresh():
    image, boxes = image_and_boxes()
    runtime, fx, sw, detections, detector_calls = make_runtime(boxes)
    first = runtime.process(image)
    # Mutating returned JSON objects cannot corrupt the cached prediction.
    first[0]["age"] = "99"
    shifted = np.zeros_like(image)
    shifted[10:50, 11:51] = image[10:50, 10:50]
    detections[:] = [FaceDetection((11, 10, 40, 40), target_id="new_frame_id")]
    fx.toward = "back"
    second = runtime.process(shifted)
    assert second[0]["x"] == 11 and second[0]["id"] == "new_frame_id"
    assert second[0]["age"] == "20" and second[0]["toward"] == "front"
    assert runtime.process(shifted) == second
    fourth = runtime.process(shifted)
    assert fourth[0]["toward"] == "back"
    assert fourth[0]["age"] == fourth[0]["gender"] == "-1"
    assert len(fx.calls) == 2 and len(sw.calls) == 1
    assert len(detector_calls) == 4
    assert runtime.temporal_stats == {"frames": 4, "targets": 4, "reused": 2, "fresh": 2}


def test_mixed_cached_and_fresh_batch_does_not_mix_people_or_use_detector_ids():
    image, boxes = image_and_boxes(3)
    runtime, fx, sw, detections, _ = make_runtime(boxes)
    runtime.process(image)
    changed = image.copy()
    x, y, w, h = boxes[1]
    changed[y:y+h, x:x+w, 1:] = 255 - changed[y:y+h, x:x+w, 1:]
    detections[:] = [FaceDetection(boxes[i], target_id="same_id") for i in (2, 1, 0)]
    result = runtime.process(changed)
    assert fx.calls == [[0, 1, 2], [1]]
    assert sw.calls == [[0, 1, 2], [1]]
    assert [head["age"] for head in result] == ["22", "21", "20"]
    assert [head["x"] for head in result] == [boxes[i][0] for i in (2, 1, 0)]


@pytest.mark.parametrize("split", [False, True])
def test_ambiguous_overlap_in_either_direction_forces_fresh_inference(split):
    image, boxes = image_and_boxes()
    runtime, fx, _, detections, _ = make_runtime(boxes)
    if not split:
        detections.append(FaceDetection(boxes[0], target_id="2"))
    runtime.process(image)
    if split:
        detections.append(FaceDetection(boxes[0], target_id="2"))
    else:
        detections.pop()
    runtime.process(image)
    assert runtime.temporal_stats["reused"] == 0
    assert len(fx.calls) == 2


@pytest.mark.parametrize("change", ["scene", "shape", "sequence", "disappearance", "motion", "reset"])
def test_cache_is_invalidated_on_sequence_or_target_changes(change):
    image, boxes = image_and_boxes()
    runtime, fx, _, detections, _ = make_runtime(boxes)
    runtime.process(image, sequence_id="a")
    changed = image.copy()
    sequence = "a"
    if change == "scene":
        changed[:] = 255
        changed[10:50, 10:50] = image[10:50, 10:50]
    elif change == "shape":
        changed = np.pad(changed, ((0, 1), (0, 0), (0, 0)))
    elif change == "sequence":
        sequence = "b"
    elif change == "disappearance":
        detections.clear()
        assert runtime.process(image, sequence_id="a") == []
        detections.append(FaceDetection(boxes[0]))
    elif change == "motion":
        changed[:] = 0
        changed[10:50, 40:80] = image[10:50, 10:50]
        detections[:] = [FaceDetection((40, 10, 40, 40))]
    elif change == "reset":
        runtime.reset_temporal_cache()
    runtime.process(changed, sequence_id=sequence)
    assert len(fx.calls) == 2 and runtime.temporal_stats["reused"] == 0


def test_gradual_appearance_changes_are_compared_to_last_real_inference():
    image, boxes = image_and_boxes()
    runtime, fx, _, _, _ = make_runtime(boxes)
    runtime.process(image)
    runtime.process(image + np.uint8(2))
    runtime.process(image + np.uint8(4))
    assert runtime.temporal_stats["reused"] == 1 and len(fx.calls) == 2


def test_small_upper_region_change_is_not_hidden_by_unchanged_shoulders():
    image, boxes = image_and_boxes()
    runtime, fx, _, _, _ = make_runtime(boxes)
    runtime.process(image)
    changed = image.copy()
    changed[10:25, 10:50, 1:] = 255 - changed[10:25, 10:50, 1:]
    runtime.process(changed)
    assert len(fx.calls) == 2


def test_uniform_crops_and_unknown_orientation_are_never_reused():
    image, boxes = image_and_boxes()
    runtime, fx, _, _, _ = make_runtime(boxes)
    runtime.process(np.full_like(image, 100))
    runtime.process(np.full_like(image, 100))
    fx.toward = "-1"
    runtime.process(image)
    runtime.process(image)
    assert len(fx.calls) == 4 and runtime.temporal_stats["reused"] == 0


def test_disabling_reuse_restores_per_frame_inference_and_exact_protocol():
    image, boxes = image_and_boxes()
    runtime, fx, sw, _, _ = make_runtime(boxes)
    runtime.temporal_reuse = False
    first = runtime.process(image)
    assert runtime.process(image) == first
    assert len(fx.calls) == len(sw.calls) == 2
    assert runtime._temporal_entries == []


def test_sdk_sequence_metadata_resets_cache_without_changing_output_protocol(monkeypatch):
    ji = _load_ji(monkeypatch)
    image, boxes = image_and_boxes()
    runtime, fx, _, _, _ = make_runtime(boxes)
    first = json.loads(ji.process_image(runtime, image, args='{"sequence_id": "a"}'))
    assert json.loads(ji.process_image(runtime, image, args={"sequence_id": "a"})) == first
    assert len(fx.calls) == 1
    assert json.loads(ji.process_image(runtime, image, sequence_id="b")) == first
    assert len(fx.calls) == 2
    ji.process_image(runtime, None)
    ji.process_image(runtime, image, sequence_id="b")
    assert len(fx.calls) == 3


@pytest.mark.parametrize("switch,enabled", [(None, False), ("0", False), ("1", True)])
def test_sdk_default_and_rollback_do_not_change_model_loading(monkeypatch, switch, enabled):
    ji = _load_ji(monkeypatch)
    ji.torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "head_detector", SimpleNamespace(
        HeadShoulderDetector=lambda path, device: lambda image: [],
        build_legacy_mtcnn_detector=lambda device: lambda image: []))
    monkeypatch.setattr(ji, "_find_weight", lambda name: name)
    monkeypatch.setattr(ji, "FaceXFormerAdapter", lambda path, device: Adapter())
    monkeypatch.setattr(ji, "SwinFaceAdapter", lambda path, device: Adapter(True))
    monkeypatch.delenv("SDK_TEMPORAL_MAX_REUSE", raising=False)
    monkeypatch.delenv("SDK_TEMPORAL_REUSE", raising=False)
    if switch is not None:
        monkeypatch.setenv("SDK_TEMPORAL_REUSE", switch)
    runtime = ji.init()
    assert runtime.temporal_reuse is enabled and runtime.temporal_max_reuse == 2
    assert runtime.detector is not None and runtime.facexformer is not None and runtime.swinface is not None
