"""极市模型榜 SDK 接口测试。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


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
    assert "expression" not in objects[1]
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
    assert objects[0]["gender"] == "1"
    assert objects[0]["glasses"] == "-1"
    assert facexformer.call_count == 1
    assert swinface.call_count == 0


def test_gender_value_uses_rank_gender_encoding():
    assert _gender_value("female") == "0"
    assert _gender_value("male") == "1"
    assert _gender_value("0") == "0"
    assert _gender_value("1") == "1"


def test_process_image_returns_extrememart_json(monkeypatch):
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