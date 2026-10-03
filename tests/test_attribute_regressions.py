"""训练评估、权重类别顺序与 SDK 朝向的回归测试，不依赖真实图片或大模型。"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SDK_SRC = ROOT / "project" / "ev_sdk" / "src"
TRAIN_SRC = ROOT / "project" / "train" / "src_repo"


def load_module(monkeypatch, name, path):
    # 本地未安装 OpenCV；这些测试只走张量、评估和 checkpoint 路径。
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_swinface_evaluation_counts_correct_wrong_and_missing_emotions(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAIN_SRC))
    model_module = types.ModuleType("model")
    model_module.build_model = lambda cfg: None
    monkeypatch.setitem(sys.modules, "model", model_module)
    training = load_module(monkeypatch, "swinface_regression", TRAIN_SRC / "finetune_swinface.py")

    class Predictions(torch.nn.Module):
        def forward(self, images):
            # 前三个对应 angry/happy/neutral；第四个标签缺失；第五个预测错误。
            expression = torch.zeros(5, 7)
            expression[torch.arange(5), torch.tensor([5, 3, 6, 2, 0])] = 10
            binary = torch.tensor([[10.0, 0.0]]).expand(5, -1)
            return {
                "Expression": expression, "Age": torch.full((5, 1), 30.0),
                "Eyeglasses": binary, "Wearing Hat": binary, "Mustache": binary,
            }

    labels = {
        "emotion": torch.tensor([0, 1, 2, -1, 0]),
        "age": torch.full((5,), 30.0),
        "glasses": torch.zeros(5, dtype=torch.long),
        "hat": torch.zeros(5, dtype=torch.long),
        "whiskers": torch.zeros(5, dtype=torch.long),
    }
    metrics = training.evaluate(
        Predictions(), [(torch.zeros(5, 3, 4, 4), labels)], torch.device("cpu")
    )
    assert metrics["emotion"] == pytest.approx(0.75)
    assert metrics["age_mae"] == 0
    assert labels["emotion"].tolist() == [0, 1, 2, -1, 0]


class TinyFaceXFormer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gender_logits = torch.nn.Parameter(torch.tensor([8.0, -8.0]))
        self.race_logits = torch.nn.Parameter(torch.zeros(5))
        self.pose = torch.nn.Parameter(torch.zeros(3))

    def forward(self, images, labels, tasks):
        batch = images.shape[0]
        empty = images.new_zeros(batch, 1)
        return (
            empty, self.pose.expand(batch, -1), empty, empty,
            images.new_zeros(batch, 8), self.gender_logits.expand(batch, -1),
            self.race_logits.expand(batch, -1), empty,
        )


def load_analyzer(monkeypatch, checkpoint):
    network = types.ModuleType("network")
    network.FaceXFormer = TinyFaceXFormer
    monkeypatch.setitem(sys.modules, "network", network)
    module = load_module(
        monkeypatch, "analyzer_regression", SDK_SRC / "face_attr" / "analyzer.py"
    )
    analyzer = module.AttributeAnalyzer(facexformer_weights=checkpoint)
    analyzer._preprocess_facexformer = lambda crop: torch.zeros(1, 3, 4, 4)
    analyzer._ensure_facexformer()
    return analyzer


@pytest.mark.parametrize("classes", [None, ["female", "male"]])
@pytest.mark.parametrize("index", [0, 1])
def test_sdk_gender_decoding_uses_checkpoint_metadata(monkeypatch, tmp_path, classes, index):
    monkeypatch.delenv("FACEXFORMER_GENDER_ORDER", raising=False)
    model = TinyFaceXFormer()
    with torch.no_grad():
        model.gender_logits.copy_(torch.tensor([8.0, -8.0]) if index == 0 else torch.tensor([-8.0, 8.0]))
    checkpoint = {"state_dict_backbone": model.state_dict()}
    if classes is not None:
        checkpoint["gender_classes"] = classes
    path = tmp_path / "model.pt"  # 重命名为 SDK 文件名后也必须保留解码顺序。
    torch.save(checkpoint, path)
    analyzer = load_analyzer(monkeypatch, path)
    prediction = analyzer._run_facexformer(np.zeros((4, 4, 3), dtype=np.uint8))
    assert prediction["gender"] == (classes or ["male", "female"])[index]


def test_old_finetuned_checkpoint_supports_explicit_gender_order(monkeypatch, tmp_path):
    monkeypatch.setenv("FACEXFORMER_GENDER_ORDER", "female,male")
    path = tmp_path / "model.pt"
    torch.save({"state_dict_backbone": TinyFaceXFormer().state_dict()}, path)
    analyzer = load_analyzer(monkeypatch, path)
    assert analyzer._run_facexformer(np.zeros((4, 4, 3), dtype=np.uint8))["gender"] == "female"


def test_invalid_gender_order_fails_instead_of_silently_swapping_labels(monkeypatch, tmp_path):
    monkeypatch.setenv("FACEXFORMER_GENDER_ORDER", "female,female")
    path = tmp_path / "model.pt"
    torch.save({"state_dict_backbone": TinyFaceXFormer().state_dict()}, path)
    with pytest.raises(ValueError, match="gender_classes"):
        load_analyzer(monkeypatch, path)


def test_training_saves_gender_order_for_sdk(monkeypatch, tmp_path):
    monkeypatch.delenv("FACEXFORMER_GENDER_ORDER", raising=False)
    monkeypatch.syspath_prepend(str(TRAIN_SRC))
    network = types.ModuleType("network")
    network.FaceXFormer = TinyFaceXFormer
    monkeypatch.setitem(sys.modules, "network", network)
    training = load_module(monkeypatch, "facexformer_regression", TRAIN_SRC / "finetune_facexformer.py")

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, index):
            return torch.zeros(3, 4, 4), {
                "gender": torch.tensor(0), "race": torch.tensor(0), "toward": torch.tensor(0),
            }

    monkeypatch.setattr(training, "FaceAttributeDataset", lambda *a, **kw: Dataset())
    initial = tmp_path / "original.pt"
    output = tmp_path / "model.pt"
    torch.save({"state_dict_backbone": TinyFaceXFormer().state_dict()}, initial)
    monkeypatch.setattr(sys, "argv", [
        "finetune_facexformer.py", "--epochs", "1", "--batch_size", "2",
        "--num_workers", "0", "--device", "cpu", "--checkpoint", str(initial),
        "--output", str(output),
    ])
    training.main()
    analyzer = load_analyzer(monkeypatch, output)
    assert analyzer._run_facexformer(np.zeros((4, 4, 3), dtype=np.uint8))["gender"] == "female"


def test_other_orientation_survives_sdk_fusion(monkeypatch):
    monkeypatch.syspath_prepend(str(SDK_SRC))
    from model_api import AttributePrediction, FaceAttributeRuntime, FaceDetection, _toward_value

    class FaceXFormer:
        def predict(self, crop):
            return AttributePrediction(toward=_toward_value("other"))

    class SwinFace:
        def predict(self, crop):
            pytest.fail("other 朝向不应调用 SwinFace")

    runtime = FaceAttributeRuntime(
        lambda image: [FaceDetection((0, 0, 4, 4))], FaceXFormer(), SwinFace()
    )
    objects = runtime.process(np.zeros((4, 4, 3), dtype=np.uint8))
    assert objects[0]["toward"] == "other"
    assert objects[0]["emotion"] == "-1"
