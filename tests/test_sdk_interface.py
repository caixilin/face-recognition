"""极市模型榜 SDK 接口测试。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SDK_SRC = Path(__file__).parents[1] / "ev_sdk" / "src"
sys.path.insert(0, str(SDK_SRC))

from model_api import (  # noqa: E402
    AttributeModelAdapter,
    AttributePrediction,
    FaceAttributeRuntime,
    FaceDetection,
)


class StubAdapter(AttributeModelAdapter):
    def __init__(self, prediction: AttributePrediction) -> None:
        self.prediction = prediction

    def predict(self, face_crop: np.ndarray) -> AttributePrediction:
        assert face_crop.shape == (20, 20, 3)
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