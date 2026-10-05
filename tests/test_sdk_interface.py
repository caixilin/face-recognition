"""极市模型榜 SDK 接口测试。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SDK_SRC = Path(__file__).parents[1] / "project" / "ev_sdk" / "src"
sys.path.insert(0, str(SDK_SRC))

from model_api import (  # noqa: E402
    AttributeModelAdapter,
    AttributePrediction,
    FaceAttributeRuntime,
    FaceDetection,
    _gender_value,
)


class StubAdapter(AttributeModelAdapter):
    def __init__(self, prediction: AttributePrediction) -> None:
        self.prediction = prediction
        self.call_count = 0

    def predict(self, face_crop: np.ndarray) -> AttributePrediction:
        assert face_crop.shape == (20, 20, 3)
        self.call_count += 1
        return self.prediction


def test_batched_runtime_preserves_target_order_crops_and_front_filter():
    class Adapter(AttributeModelAdapter):
        def __init__(self, front_only=False):
            self.front_only = front_only
            self.calls = []

        def predict(self, crop):
            return self.predict_batch([crop])[0]

        def predict_batch(self, crops):
            ids = [int(crop[0, 0, 0]) for crop in crops]
            self.calls.append(ids)
            if self.front_only:
                assert all(i % 3 == 0 for i in ids)
                return [AttributePrediction(age=str(20 + i), glasses=str(i % 3), mask="1") for i in ids]
            return [AttributePrediction(toward=["front", "back", "other"][i % 3],
                                        gender=str(i % 2), race=str(i % 4)) for i in ids]

    image = np.zeros((10, 80, 3), dtype=np.uint8)
    detections = [FaceDetection(None, (0, 0, 5, 10), target_id="person_only")]
    for i in range(7):
        image[:, i * 10:(i + 1) * 10] = i
        # First crop clips at the image edge; output must retain the original box.
        head = (-2, -2, 12, 12) if i == 0 else (i * 10, 0, 10, 10)
        detections.append(FaceDetection(head, (i * 10, 0, 10, 10), target_id=str(i)))
    detections.insert(3, FaceDetection((100, 0, 10, 10), (100, 0, 10, 10), target_id="invalid"))

    fx, sw = Adapter(), Adapter(front_only=True)
    batched = FaceAttributeRuntime(lambda _: detections, fx, sw, attribute_batch_size=4).process(image)
    serial = FaceAttributeRuntime(lambda _: detections, Adapter(), Adapter(True), attribute_batch_size=1).process(image)
    assert batched == serial
    assert fx.calls == [[0, 1, 2, 3], [4, 5, 6]]
    assert sw.calls == [[0, 3, 6]]
    assert [obj["id"] for obj in batched] == ["person_only"] + [str(i) for i in range(7) for _ in range(2)]
    heads = [obj for obj in batched if obj["name"] == "head"]
    assert heads[0]["x"] == -2
    assert heads[3]["age"] == "23"
    assert heads[6]["age"] == "26"
    assert heads[1]["gender"] == heads[1]["mask"] == "-1"
    assert heads[2]["race"] == "2" and heads[2]["mask"] == "-1"


def test_batched_runtime_rejects_missing_results_instead_of_mixing_targets():
    class BrokenAdapter:
        def predict_batch(self, crops):
            return [AttributePrediction(toward="front")]

    detector = lambda _: [FaceDetection((0, 0, 10, 10)) for _ in range(2)]
    runtime = FaceAttributeRuntime(detector, BrokenAdapter(), attribute_batch_size=4)
    with pytest.raises(ValueError, match="数量"):
        runtime.process(np.zeros((10, 10, 3), dtype=np.uint8))


def test_runtime_fuses_facexformer_and_swinface_outputs():
    detector = lambda image: [  # noqa: E731
        FaceDetection(
            head_bbox=(10, 5, 20, 20),
            person_bbox=(5, 0, 30, 60),
            target_id="7",
        )
    ]
    facexformer = StubAdapter(
        AttributePrediction(toward="front", gender="1", age="26", race="0")
    )
    swinface = StubAdapter(
        AttributePrediction(glasses="1", emotion="2", hat="0", whiskers="1")
    )
    runtime = FaceAttributeRuntime(detector, facexformer, swinface)

    objects = runtime.process(np.zeros((80, 80, 3), dtype=np.uint8))

    assert objects[0] == {
        "x": 5,
        "y": 0,
        "width": 30,
        "height": 60,
        "id": "7",
        "name": "person",
    }
    assert objects[1]["name"] == "head"
    assert objects[1]["toward"] == "front"
    assert objects[1]["gender"] == "1"
    assert objects[1]["emotion"] == "2"
    # 榜单 objects 字段同时列出 expression 与 emotion，两个键都要输出。
    assert objects[1]["expression"] == "2"
    assert objects[1]["age"] == "26"
    assert facexformer.call_count == 1
    assert swinface.call_count == 1


def test_runtime_skips_swinface_for_non_front_face():
    detector = lambda image: [  # noqa: E731
        FaceDetection(head_bbox=(10, 5, 20, 20))
    ]
    facexformer = StubAdapter(AttributePrediction(toward="back", gender="1"))
    swinface = StubAdapter(AttributePrediction(glasses="1", emotion="2"))
    runtime = FaceAttributeRuntime(detector, facexformer, swinface)

    objects = runtime.process(np.zeros((80, 80, 3), dtype=np.uint8))

    assert objects[0]["toward"] == "back"
    # 榜单示例中 toward=back 的目标所有属性均为 -1，且 ACC 规则 2 规定
    # back 识别正确即判对，属性不参与判分。
    assert objects[0]["gender"] == "-1"
    assert objects[0]["glasses"] == "-1"
    assert facexformer.call_count == 1
    assert swinface.call_count == 0


def test_gender_value_uses_rank_gender_encoding():
    assert _gender_value("female") == "0"
    assert _gender_value("male") == "1"
    assert _gender_value("0") == "0"
    assert _gender_value("1") == "1"


def _load_ji(monkeypatch):
    torch_stub = type(
        "TorchStub",
        (),
        {"no_grad": staticmethod(lambda: lambda function: function)},
    )()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    spec = importlib.util.spec_from_file_location("ji", SDK_SRC / "ji.py")
    assert spec is not None and spec.loader is not None
    ji = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ji)
    return ji


@pytest.mark.parametrize(
    "name,finetuned_name",
    [
        ("facexformer/model.pt", "facexformer_finetuned.pt"),
        ("swinface/checkpoint_step_79999_gpu_0.pt", "swinface_finetuned.pt"),
        ("headshoulder/model.pt", "headshoulder_finetuned.pt"),
    ],
)
def test_sdk_prefers_mounted_finetuned_weights(monkeypatch, name, finetuned_name):
    ji = _load_ji(monkeypatch)
    mounted_path = Path("/project/train/models") / finetuned_name
    sdk_path = Path("/project/ev_sdk/model") / name
    monkeypatch.setattr(Path, "is_file", lambda path: path in (mounted_path, sdk_path))
    assert ji._find_weight(name) == str(mounted_path)

    # 未选中微调产物时，仍可使用已有的 SDK 权重。
    monkeypatch.setattr(Path, "is_file", lambda path: path == sdk_path)
    assert ji._find_weight(name) == str(sdk_path)

    local_path = SDK_SRC / ".." / "model" / name
    local_path = local_path.resolve()
    monkeypatch.setattr(Path, "is_file", lambda path: path == local_path)
    assert ji._find_weight(name) == str(local_path)

    monkeypatch.setattr(Path, "is_file", lambda path: False)
    with pytest.raises(FileNotFoundError, match=finetuned_name):
        ji._find_weight(name)


def test_process_image_returns_extrememart_json(monkeypatch):
    ji = _load_ji(monkeypatch)

    runtime = FaceAttributeRuntime(
        detector=lambda image: [FaceDetection(head_bbox=(1, 2, 3, 4))]
    )
    result = json.loads(
        ji.process_image(
            handle=runtime,
            input_image=np.zeros((10, 10, 3), dtype=np.uint8),
        )
    )

    assert result["algorithm_data"]["is_alert"] is True
    assert result["algorithm_data"]["target_count"] == 1
    assert result["model_data"]["objects"][0]["emotion"] == "-1"


@pytest.mark.parametrize("toward", ["back", "other", "-1"])
def test_non_front_detections_preserve_existing_alert_behavior(monkeypatch, toward):
    ji = _load_ji(monkeypatch)
    runtime = FaceAttributeRuntime(
        detector=lambda image: [FaceDetection(head_bbox=(1, 2, 3, 4))],
        facexformer=type("Adapter", (), {"predict": lambda self, crop: AttributePrediction(toward=toward)})(),
    )
    result = json.loads(ji.process_image(runtime, np.zeros((10, 10, 3), dtype=np.uint8)))
    assert result["algorithm_data"]["is_alert"] is True
    assert result["algorithm_data"]["target_count"] == 1
    assert result["model_data"]["objects"][0]["toward"] == toward


def test_person_only_and_out_of_image_heads_do_not_call_attributes():
    adapter = StubAdapter(AttributePrediction(toward="front"))
    runtime = FaceAttributeRuntime(
        lambda image: [FaceDetection(None, (0, 0, 50, 60), target_id="1"),
                       FaceDetection((100, 100, 5, 5), target_id="2")], adapter, adapter)
    result = runtime.process(np.zeros((80, 80, 3), dtype=np.uint8))
    assert len(result) == 1 and result[0]["name"] == "person"
    assert adapter.call_count == 0
