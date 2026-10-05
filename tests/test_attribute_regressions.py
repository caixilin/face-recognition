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


@pytest.mark.parametrize("fallback", ["", "<mask>0</mask><gender>0</gender>",
    "<attributes><attribute><name>mask</name><value>0</value></attribute>"
    "<attribute><name>gender</name><value>0</value></attribute></attributes>"])
def test_missing_object_labels_do_not_copy_another_person(monkeypatch, fallback):
    import xml.etree.ElementTree as ET
    dataset = load_module(monkeypatch, "dataset_label_isolation", TRAIN_SRC / "dataset.py")
    box = "<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>11</xmax><ymax>22</ymax></bndbox>"
    root = ET.fromstring("<annotation>" + fallback +
        "<object><name>head</name><mask>1</mask><gender>1</gender>" + box + "</object>" +
        "<object><name>head</name>" + box + "</object></annotation>")
    labels = object.__new__(dataset.FaceAttributeDataset)._parse_xml(root)
    assert len(labels) == 2
    assert labels[0]["mask"] == 1 and labels[0]["gender"] == 1
    expected = 0 if fallback else -1
    assert labels[1]["mask"] == expected and labels[1]["gender"] == expected


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

    def forward(self, images, labels, tasks, sdk_only=False):
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


@pytest.mark.parametrize("classes,override,expected", [
    (None, None, ["female", "male"]),
    (["male", "female"], None, ["male", "female"]),
    (None, "male,female", ["male", "female"]),
])
@pytest.mark.parametrize("index", [0, 1])
def test_old_named_finetuned_gender_order_and_explicit_overrides(monkeypatch, tmp_path, classes, override, expected, index):
    monkeypatch.delenv("FACEXFORMER_GENDER_ORDER", raising=False)
    if override is not None:
        monkeypatch.setenv("FACEXFORMER_GENDER_ORDER", override)
    model = TinyFaceXFormer()
    with torch.no_grad():
        model.gender_logits.copy_(torch.tensor([8.0, -8.0]) if index == 0 else torch.tensor([-8.0, 8.0]))
    checkpoint = {"state_dict_backbone": model.state_dict()}
    if classes is not None:
        checkpoint["gender_classes"] = classes
    path = tmp_path / "facexformer_finetuned.pt"
    torch.save(checkpoint, path)
    analyzer = load_analyzer(monkeypatch, path)
    assert analyzer._run_facexformer(np.zeros((4, 4, 3), dtype=np.uint8))["gender"] == expected[index]


def test_invalid_gender_order_fails_instead_of_silently_swapping_labels(monkeypatch, tmp_path):
    monkeypatch.setenv("FACEXFORMER_GENDER_ORDER", "female,female")
    path = tmp_path / "model.pt"
    torch.save({"state_dict_backbone": TinyFaceXFormer().state_dict()}, path)
    with pytest.raises(ValueError, match="gender_classes"):
        load_analyzer(monkeypatch, path)


@pytest.mark.parametrize("check_only", [False, True])
def test_training_saves_gender_order_for_sdk(monkeypatch, tmp_path, check_only):
    monkeypatch.delenv("FACEXFORMER_GENDER_ORDER", raising=False)
    monkeypatch.syspath_prepend(str(TRAIN_SRC))
    network = types.ModuleType("network")
    network.FaceXFormer = TinyFaceXFormer
    monkeypatch.setitem(sys.modules, "network", network)
    training = load_module(monkeypatch, "facexformer_regression", TRAIN_SRC / "finetune_facexformer.py")

    class Dataset(torch.utils.data.Dataset):
        samples = [(f"image-{index}.jpg", {"toward": index % 3}) for index in range(5)]
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
    argv = [
        "finetune_facexformer.py", "--epochs", "1", "--batch_size", "2",
        "--num_workers", "0", "--device", "cpu", "--checkpoint", str(initial),
        "--output", str(output),
    ]
    if check_only:
        argv.append("--check_only")
        monkeypatch.setattr(training, "run_epoch", lambda *a: pytest.fail("preflight must not train"))
        monkeypatch.setattr(training, "evaluate", lambda *a, **kw: pytest.fail("preflight must not evaluate"))
    monkeypatch.setattr(sys, "argv", argv)
    training.main()
    if check_only:
        assert not output.exists() and not Path(str(output) + ".validation.json").exists()
        return
    import json

    report = json.loads(Path(str(output) + ".validation.json").read_text(encoding="utf-8"))
    assert report["validation_evaluated"] == 1
    assert report["baseline_gender_classes"] == ["male", "female"]
    assert report["baseline"]["gender"] == 0
    assert report["epochs"][0]["metrics"]["gender"] == 1
    assert report["baseline"]["classification"]["toward"]["support"] == [1, 0, 0]
    assert len(report["epochs"]) == 1
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


def test_glasses_three_class_diagnostic_does_not_merge_sunglasses(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAIN_SRC))
    model_module = types.ModuleType("model")
    model_module.build_model = lambda cfg: None
    monkeypatch.setitem(sys.modules, "model", model_module)
    training = load_module(monkeypatch, "swinface_track_metrics", TRAIN_SRC / "finetune_swinface.py")

    class Predictions(torch.nn.Module):
        def forward(self, images):
            binary = torch.tensor([[10., 0.], [0., 10.], [0., 10.], [0., 10.]])
            return {"Age": torch.zeros(4), "Expression": torch.zeros(4, 7),
                    "Eyeglasses": binary, "Wearing Hat": binary, "Mustache": binary}

    labels = {"age": torch.full((4,), -1.), "emotion": torch.full((4,), -1),
              "glasses": torch.tensor([0, 1, 2, -1]),
              "hat": torch.tensor([0, 1, 0, -1]), "whiskers": torch.tensor([0, 1, 0, -1])}
    metrics = training.evaluate(Predictions(), [(torch.zeros(4, 3, 4, 4), labels)], torch.device("cpu"))
    assert metrics["glasses"] == pytest.approx(2 / 3)
    assert metrics["glasses_binary"] == 1
    glasses = metrics["classification"]["glasses"]
    assert glasses["support"] == [1, 1, 1]
    assert glasses["per_class_recall"] == [1, 1, 0]
    assert glasses["ignored_count"] == 1
    assert metrics["classification"]["hat"]["per_class_recall"] == [0.5, 1]
    assert metrics["age_labeled_count"] == 0
    assert metrics["age_ignored_count"] == 4

    class TiedPredictions(Predictions):
        def forward(self, images):
            outputs = super().forward(images)
            for key in ("Eyeglasses", "Wearing Hat", "Mustache"):
                outputs[key] = torch.zeros(4, 2)
            return outputs

    tied_labels = dict(labels, glasses=torch.ones(4, dtype=torch.long),
                       hat=torch.ones(4, dtype=torch.long), whiskers=torch.ones(4, dtype=torch.long))
    tied = training.evaluate(TiedPredictions(), [(torch.zeros(4, 3, 4, 4), tied_labels)], torch.device("cpu"))
    assert tied["glasses"] == tied["hat"] == tied["whiskers"] == 1


def test_facexformer_validation_uses_sdk_thresholds_and_original_gender_order(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAIN_SRC))
    network = types.ModuleType("network")
    network.FaceXFormer = TinyFaceXFormer
    monkeypatch.setitem(sys.modules, "network", network)
    training = load_module(monkeypatch, "facexformer_track_metrics", TRAIN_SRC / "finetune_facexformer.py")
    analyzer = load_module(monkeypatch, "analyzer_pose_metrics", SDK_SRC / "face_attr" / "analyzer.py")

    class Predictions(TinyFaceXFormer):
        def forward(self, images, labels, tasks):
            outputs = list(super().forward(images, labels, tasks))
            # yaw 1.6 is back under SDK thresholds but other under the old nearest-prototype rule.
            outputs[1] = torch.tensor([[0., 0., 0.], [0., 0.6, 0.], [0., 1.6, 0.]])
            return tuple(outputs)

    model = Predictions()
    poses = model(torch.zeros(3, 3, 4, 4), None, None)[1].numpy()
    labels = {"toward": torch.tensor([0, 2, 1]), "gender": torch.ones(3, dtype=torch.long),
              "race": torch.zeros(3, dtype=torch.long)}
    metrics = training.evaluate(model, [(torch.zeros(3, 3, 4, 4), labels)],
                                torch.device("cpu"), gender_classes=["male", "female"])
    assert [analyzer.AttributeAnalyzer._headpose_to_orientation(pose) for pose in poses] == ["front", "other", "back"]
    assert metrics["toward"] == 1
    assert metrics["gender"] == 1
    assert metrics["classification"]["toward"]["support"] == [1, 1, 1]


def test_validation_report_keeps_missing_classes_and_unknown_predictions_explicit(monkeypatch, tmp_path):
    import json

    monkeypatch.syspath_prepend(str(TRAIN_SRC))
    from validation_metrics import ClassificationCounts, write_validation_report

    counts = ClassificationCounts(["absent", "present"])
    counts.update([0, 0, -1], [0, -1, 1])
    detail = counts.report()
    assert detail["accuracy"] == 0.5
    assert detail["support"] == [2, 0]
    assert detail["per_class_recall"] == [0.5, None]
    assert detail["confusion_matrix"] == [[1, 0, 1], [0, 0, 0]]
    path = write_validation_report(tmp_path / "model.pt", {"detail": detail, "age_mae": float("nan")})
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["age_mae"] is None
    assert report["detail"]["per_class_recall"][1] is None


@pytest.mark.parametrize("cap, expected_count", [(0, 2), (1, 1)])
@pytest.mark.parametrize("archive,check_only", [(False, False), (True, False), (False, True)])
def test_swinface_baseline_and_epoch_use_same_validation_targets(monkeypatch, tmp_path, cap, expected_count, archive, check_only):
    import json

    monkeypatch.syspath_prepend(str(TRAIN_SRC))
    model_module = types.ModuleType("model")
    monkeypatch.setitem(sys.modules, "model", model_module)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("backbone", "fam", "tss", "om"):
                setattr(self, name, torch.nn.Linear(1, 1))
            self.om.enable_platform_heads = lambda: None

        def forward(self, images):
            batch = len(images)
            binary = images.new_tensor([[10., 0.]]).expand(batch, -1)
            return {"Age": images.new_zeros(batch), "Expression": images.new_zeros(batch, 7),
                    "Eyeglasses": binary, "Wearing Hat": binary, "Mustache": binary}

    model_module.build_model = lambda cfg: Model()
    training = load_module(monkeypatch, "swinface_baseline_entry", TRAIN_SRC / "finetune_swinface.py")

    class Dataset(torch.utils.data.Dataset):
        samples = [(f"image-{index}.jpg", {"glasses": index % 3, "mask": index % 2}) for index in range(10)]
        def __len__(self):
            return 10

        def __getitem__(self, index):
            return torch.zeros(3, 4, 4), {
                "age": torch.tensor(float(index)), "emotion": torch.tensor(2),
                "glasses": torch.tensor(index % 3), "hat": torch.tensor(index % 2),
                "whiskers": torch.tensor(index % 2),
            }

    monkeypatch.setattr(training, "FaceAttributeDataset", lambda *a, **kw: Dataset())
    trained_batches = []
    def train_and_record(model, loader, optimizer, device, loss_stats=None):
        trained_batches.append([len(images) for images, _ in loader])
        with torch.no_grad():
            model.backbone.weight.add_(1)
        if loss_stats is not None:
            loss_stats["mask"] = 0.25
        return 0.25
    monkeypatch.setattr(training, "train_epoch", train_and_record)
    evaluation = training.evaluate
    seen = []

    def evaluate_and_record(model, loader, device):
        batches = list(loader)
        seen.append([int(age) for _, labels in batches for age in labels["age"]])
        return evaluation(model, batches, device)

    monkeypatch.setattr(training, "evaluate", evaluate_and_record)
    initial = tmp_path / "original.pt"
    output = tmp_path / "finetuned.pt"
    model = Model()
    torch.save({"state_dict_" + name: getattr(model, name).state_dict()
                for name in ("backbone", "fam", "tss", "om")}, initial)
    argv = [
        "finetune_swinface.py", "--epochs", "2", "--num_workers", "0", "--device", "cpu",
        "--checkpoint", str(initial), "--output", str(output), "--max_val_samples", str(cap),
        "--batch_size", "7",
    ]
    if archive:
        argv.append("--keep_epoch_checkpoints")
    if check_only:
        argv.append("--check_only")
    monkeypatch.setattr(sys, "argv", argv)
    training.main()
    if check_only:
        assert trained_batches == [] and seen == []
        assert not output.exists() and not Path(str(output) + ".validation.json").exists()
        return
    report = json.loads(Path(str(output) + ".validation.json").read_text(encoding="utf-8"))
    assert seen[0] == seen[1] == seen[2]
    assert len(seen[0]) == expected_count
    assert report["validation_total"] == 2
    assert report["validation_evaluated"] == expected_count
    assert report["baseline"] == report["epochs"][0]["metrics"]
    assert trained_batches == [[7], [7]]  # 8 train targets: singleton BN batch skipped
    assert report["training_drop_singleton_batch"] is True
    assert report["epochs"][0]["loss_components"] == {"mask": 0.25}
    first_path = output.with_name("finetuned.epoch_01.pt")
    second_path = output.with_name("finetuned.epoch_02.pt")
    assert first_path.exists() is archive
    assert second_path.exists() is archive
    if archive:
        first = torch.load(first_path, map_location="cpu", weights_only=False)
        second = torch.load(second_path, map_location="cpu", weights_only=False)
        assert first["epoch"] == 1 and second["epoch"] == 2
        torch.testing.assert_close(second["state_dict_backbone"]["weight"],
                                   first["state_dict_backbone"]["weight"] + 1)
        assert torch.load(output, map_location="cpu", weights_only=False)["epoch"] == 2


@pytest.mark.parametrize("check_only", [False, True])
def test_full_head_training_keeps_independent_epoch_checkpoints(monkeypatch, tmp_path, check_only):
    monkeypatch.syspath_prepend(str(TRAIN_SRC))
    head = types.ModuleType("head_detector")
    head.ARCHITECTURE = "test_head"
    head.DETECTOR_CLASSES = ["background", "person", "head"]
    head.build_head_detector = lambda *a, **kw: pytest.fail("must reuse the existing head model")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.transform = types.SimpleNamespace(min_size=(640,), max_size=1280)
        def forward(self, images, targets):
            return {"test_loss": self.weight.square()}
    head.load_head_detector = lambda path, device: Model().to(device)
    monkeypatch.setitem(sys.modules, "head_detector", head)
    check = types.ModuleType("check_detection_boxes")
    check.box_iou = lambda *a: 0
    monkeypatch.setitem(sys.modules, "check_detection_boxes", check)
    data = types.ModuleType("detection_dataset")
    limits = []
    class Dataset(torch.utils.data.Dataset):
        samples = [(f"image-{i}.jpg", {}) for i in range(4)]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, index):
            return torch.zeros(3, 4, 4), {"boxes": torch.tensor([[0., 0., 2., 2.]]),
                                        "labels": torch.tensor([2])}
    def dataset(path, limit):
        limits.append(limit)
        return Dataset()
    data.HeadShoulderDataset = dataset
    data.collate_detection_batch = lambda batch: tuple(zip(*batch))
    monkeypatch.setitem(sys.modules, "detection_dataset", data)
    training = load_module(monkeypatch, "head_training_entry", TRAIN_SRC / "train_head_detector.py")
    monkeypatch.setattr(training, "evaluate", lambda *args: {"head": {"precision": 1., "recall": 1.}})
    initial, output = tmp_path / "existing.pt", tmp_path / "headshoulder_finetuned.pt"
    initial.touch()
    argv = ["train_head_detector.py", "--checkpoint", str(initial),
        "--output", str(output), "--max_images", "0", "--epochs", "2", "--num_workers", "0",
        "--device", "cpu", "--keep_epoch_checkpoints"]
    if check_only:
        argv.append("--check_only")
        monkeypatch.setattr(training, "evaluate", lambda *a: pytest.fail("preflight must not evaluate"))
    monkeypatch.setattr(sys, "argv", argv)
    training.main()
    assert limits == [0]
    if check_only:
        assert not output.exists() and not Path(str(output) + ".validation.json").exists()
        return
    first = torch.load(tmp_path / "headshoulder_finetuned.epoch_01.pt", weights_only=False)
    last = torch.load(output, weights_only=False)
    assert first["epoch"] == 1 and last["epoch"] == 2
    assert first["state_dict"]["weight"] > last["state_dict"]["weight"]
    assert (tmp_path / "headshoulder_finetuned.epoch_02.pt").read_bytes() == output.read_bytes()


def test_sunglasses_and_mask_receive_supervised_gradients(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAIN_SRC))
    model_module = types.ModuleType("model")
    model_module.build_model = lambda cfg: None
    monkeypatch.setitem(sys.modules, "model", model_module)
    training = load_module(monkeypatch, "platform_attribute_loss", TRAIN_SRC / "finetune_swinface.py")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.eye = torch.nn.Parameter(torch.zeros(3))
            self.mask = torch.nn.Parameter(torch.zeros(2))
            self.other = torch.nn.Parameter(torch.zeros(7))

        def forward(self, images):
            n = len(images)
            return {"Eyeglasses": self.eye.expand(n, -1), "Mask": self.mask.expand(n, -1),
                    "Age": self.other[:1].expand(n), "Expression": self.other.expand(n, -1),
                    "Wearing Hat": self.mask.expand(n, -1), "Mustache": self.mask.expand(n, -1)}

    model = Model()
    labels = {key: torch.full((2,), -1, dtype=torch.long)
              for key in ("emotion", "hat", "whiskers")}
    labels.update(glasses=torch.tensor([2, -1]), mask=torch.tensor([1, -1]), age=torch.full((2,), -1.))
    loss_stats = {}
    loss = training.train_epoch(model, [(torch.zeros(2, 3, 4, 4), labels)],
                                torch.optim.SGD(model.parameters(), lr=0), torch.device("cpu"), loss_stats=loss_stats)
    assert np.isfinite(loss)
    assert loss_stats["age_mse"] == 0
    assert loss_stats["mask"] > 0 and loss_stats["glasses"] > 0
    assert sum(loss_stats.values()) == pytest.approx(loss)
    assert model.eye.grad[2] < 0  # sunglasses class receives its own target, not class 1
    assert model.mask.grad[1] < 0
    metrics = training.evaluate(model, [(torch.zeros(2, 3, 4, 4), labels)], torch.device("cpu"))
    assert metrics["classification"]["glasses"]["support"] == [0, 0, 1]
    assert metrics["classification"]["mask"]["support"] == [0, 1]


def test_new_orientation_checkpoint_decodes_classifier_instead_of_angles(monkeypatch, tmp_path):
    monkeypatch.delenv("FACEXFORMER_GENDER_ORDER", raising=False)
    model = TinyFaceXFormer()
    model.toward_classifier = torch.nn.Linear(3, 3)
    with torch.no_grad():
        model.toward_classifier.weight.zero_()
        model.toward_classifier.bias.copy_(torch.tensor([0., 0., 10.]))
    path = tmp_path / "new_fx.pt"
    torch.save({"state_dict_backbone": model.state_dict(), "gender_classes": ["female", "male"],
                "toward_classes": ["front", "back", "other"]}, path)
    analyzer = load_analyzer(monkeypatch, path)
    assert analyzer._run_facexformer(np.zeros((4, 4, 3), dtype=np.uint8))["orientation"] == "other"


def test_real_swinface_output_heads_expand_and_reload(monkeypatch):
    analysis_path = SDK_SRC / "face_attr/models/swinface/analysis"
    package = types.ModuleType("test_swin_analysis")
    package.__path__ = [str(analysis_path)]
    monkeypatch.setitem(sys.modules, "test_swin_analysis", package)
    layers = types.ModuleType("timm.models.layers")
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    monkeypatch.setitem(sys.modules, "timm.models.layers", layers)
    module = load_module(monkeypatch, "test_swin_analysis.subnets", analysis_path / "subnets.py")
    old = module.OutputModule(feature_dim=8)
    features = torch.randn(11, 2, 8)
    embedding = torch.randn(2, 8)
    original = old(features, embedding)["Eyeglasses"].detach().clone()
    old.enable_platform_heads()
    results = old(features, embedding)
    assert results["Eyeglasses"].shape == (2, 3)
    assert results["Mask"].shape == (2, 2)
    torch.testing.assert_close(results["Eyeglasses"][:, :2], original)
    copy = module.OutputModule(feature_dim=8)
    copy.enable_platform_heads()
    copy.load_state_dict(old.state_dict())
    torch.testing.assert_close(copy(features, embedding)["Mask"], results["Mask"])


@pytest.mark.parametrize("new_heads", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shared_conv", [False, True])
def test_swinface_sdk_skips_independent_branches_and_preserves_outputs(
        monkeypatch, new_heads, batch_size, shared_conv):
    analysis_path = SDK_SRC / "face_attr/models/swinface/analysis"
    package = types.ModuleType("sdk_swin_analysis")
    package.__path__ = [str(analysis_path)]
    monkeypatch.setitem(sys.modules, "sdk_swin_analysis", package)
    layers = types.ModuleType("timm.models.layers")
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    monkeypatch.setitem(sys.modules, "timm.models.layers", layers)
    module = load_module(monkeypatch, "sdk_swin_analysis.subnets", analysis_path / "subnets.py")

    class Backbone(torch.nn.Module):
        def forward(self, images):
            return images, images, images.mean(dim=(2, 3))

    fam = module.FeatureAttentionModule(in_chans=16, conv_mode="normal", conv_shared=shared_conv)
    tss = module.TaskSpecificSubnets()
    om = module.OutputModule()
    if new_heads:
        om.enable_platform_heads()
    model = module.ModelBox(Backbone(), fam, tss, om, feature="local").eval()
    images = torch.randn(batch_size, 16, 3, 3)
    keys_before = set(model.state_dict())
    with torch.no_grad():
        full = model(images)
    assert len(full) == (44 if new_heads else 43)
    expected_branches = [1, 2, 4, 5, 6, 8] if new_heads else [1, 2, 4, 5, 8]
    assert om.sdk_branches() == expected_branches

    def unused_called(*args):
        pytest.fail("SDK executed an unused SwinFace branch or output head")

    hooks = []
    for i in set(range(11)) - set(expected_branches):
        hooks.append(fam.nets[i].register_forward_pre_hook(unused_called))
        hooks.append(tss.nets[i].register_forward_pre_hook(unused_called))
    used_heads = {sum(len(group) for group in om.output_sizes[:branch]) + offset
                  for branch, offset in om.SDK_HEADS.values()}
    for i, head in enumerate(om.output_fcs):
        if i not in used_heads:
            hooks.append(head.register_forward_pre_hook(unused_called))
    with torch.no_grad():
        reduced = model(images, sdk_only=True)
    expected_names = set(om.SDK_HEADS) | ({"Mask"} if new_heads else set())
    assert set(reduced) == expected_names
    for name in reduced:
        torch.testing.assert_close(reduced[name], full[name], rtol=0, atol=0)
    assert set(model.state_dict()) == keys_before
    for hook in hooks:
        hook.remove()

    # Default training/evaluation path retains every branch and strict loading.
    model.load_state_dict(model.state_dict(), strict=True)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model(images, sdk_only=True)
    training_output = model(torch.randn(2, 16, 3, 3))
    assert set(training_output) == set(full)
    training_output["Gender"].sum().backward()
    assert om.output_fcs[0].weight.grad is not None
    assert tss.nets[0].feature[0].weight.grad is not None


def test_sdk_adapter_returns_sunglasses_and_mask(monkeypatch):
    monkeypatch.syspath_prepend(str(SDK_SRC))
    import model_api

    analyzer = types.SimpleNamespace(_ensure_swinface=lambda: None,
        _run_swinface=lambda crop: {"glasses_class": 2, "mask": 0.9, "age": 23.7,
                                   "expression": "neutral", "mustache": 0.8, "wearing_hat": 0.1})
    monkeypatch.setattr(model_api, "_build_analyzer", lambda *args: analyzer)
    prediction = model_api.SwinFaceAdapter("new.pt", "cpu").predict(np.zeros((4, 4, 3), dtype=np.uint8))
    assert prediction.glasses == "2"
    assert prediction.mask == "1"
    assert prediction.age == "24"


@pytest.mark.parametrize("batch_size", [1, 4])
def test_sdk_facexformer_decoder_skips_unused_heads_without_changing_predictions(monkeypatch, batch_size):
    network_path = SDK_SRC / "face_attr/models/facexformer/network"
    for name, path in (("perf_fx", network_path), ("perf_fx.models", network_path / "models")):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    backbone = types.ModuleType("perf_fx.swin_b_compat")
    backbone.SwinBCompat = torch.nn.Identity
    monkeypatch.setitem(sys.modules, "perf_fx.swin_b_compat", backbone)
    module = load_module(monkeypatch, "perf_fx.models.facexformer", network_path / "models/facexformer.py")
    decoder = module.FaceDecoder(transformer_dim=32, transformer=module.TwoWayTransformer(
        depth=1, embedding_dim=32, num_heads=4, mlp_dim=64)).eval()
    embeddings, positions = torch.randn(batch_size, 32, 4, 4), torch.randn(1, 32, 4, 4)
    state_keys = set(decoder.state_dict())
    with torch.no_grad():
        full = decoder(embeddings, positions)
    assert all(isinstance(value, torch.Tensor) for value in full)

    def unused_head_called(*args):
        pytest.fail("SDK executed an unused output branch")

    for name in ("landmarks_prediction_head", "attribute_prediction_head",
                 "visibility_prediction_head", "output_upscaling", "output_hypernetwork_mlps"):
        getattr(decoder, name).register_forward_pre_hook(unused_head_called)
    with torch.no_grad():
        reduced = decoder(embeddings, positions, sdk_only=True)
    for index in (1, 4, 5, 6):
        torch.testing.assert_close(reduced[index], full[index], rtol=0, atol=0)
    with torch.no_grad():
        singles = [decoder(embedding.unsqueeze(0), positions, sdk_only=True) for embedding in embeddings]
    for index in (1, 4, 5, 6):
        torch.testing.assert_close(reduced[index], torch.cat([row[index] for row in singles]), rtol=1e-5, atol=1e-6)
    assert all(reduced[index] is None for index in (0, 2, 3, 7))
    assert set(decoder.state_dict()) == state_keys


@pytest.mark.parametrize("new_heads", [False, True])
def test_swinface_packed_results_preserve_legacy_values_and_copy_once(monkeypatch, new_heads):
    module = load_module(monkeypatch, "analyzer_packed_swin", SDK_SRC / "face_attr/analyzer.py")
    binary = torch.tensor([[0.25, -0.75]])
    output = {"Age": torch.tensor([[23.7]]), "Expression": torch.tensor([[0., 0., 0., 2., 0., 0., 0.]]),
              "Smiling": binary, "Wearing Hat": -binary, "Mustache": binary, "No Beard": -binary,
              "Eyeglasses": torch.tensor([[0., 1., 2.]]) if new_heads else binary}
    if new_heads:
        output["Mask"] = torch.tensor([[0.3]])  # single-logit compatibility
    expected = {"age": output["Age"].item(), "expression": "happy",
                "smiling": torch.softmax(binary, -1)[0, 1].item(),
                "wearing_hat": torch.softmax(-binary, -1)[0, 1].item(),
                "mustache": torch.softmax(binary, -1)[0, 1].item(),
                "no_beard": torch.softmax(-binary, -1)[0, 1].item()}
    if new_heads:
        expected.update(glasses_class=2, mask=torch.sigmoid(output["Mask"][0, 0]).item())
    else:
        expected["eyeglasses"] = torch.softmax(binary, -1)[0, 1].item()
    analyzer = module.AttributeAnalyzer()
    analyzer._swinface = lambda images, sdk_only=False: output
    analyzer._preprocess_swinface = lambda crop: torch.zeros(1, 3, 4, 4)
    copies = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    actual = analyzer._run_swinface(np.zeros((4, 4, 3), dtype=np.uint8))
    assert actual == expected
    assert len(copies) == 1
    if new_heads:
        assert isinstance(actual["glasses_class"], int)


@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("scalar_age", [False, True])
def test_facexformer_batch_decodes_each_target_and_copies_once(monkeypatch, classifier, scalar_age):
    module = load_module(monkeypatch, "analyzer_batch_fx", SDK_SRC / "face_attr/analyzer.py")
    calls = []

    class Model:
        def __call__(self, images, labels, tasks, sdk_only=False):
            assert sdk_only
            calls.append(len(images))
            ids = images[:, 0, 0, 0].long()
            pose = torch.tensor([[0., 0., 0.], [0., 3.14, 0.], [0., .8, 0.]])[ids]
            age = (ids.float() + 20).unsqueeze(1) if scalar_age else torch.eye(8)[ids + 2]
            return None, pose, None, None, age, torch.eye(2)[ids % 2], torch.eye(5)[ids + 1], None

    analyzer = module.AttributeAnalyzer()
    analyzer._facexformer = Model()
    if classifier:
        analyzer._facexformer.toward_classifier = lambda pose: torch.eye(3)[
            torch.where(pose[:, 1] > 2, 1, torch.where(pose[:, 1] > .5, 2, 0))]
    analyzer._preprocess_facexformer = lambda crop: torch.full((1, 3, 4, 4), float(crop[0, 0, 0]))
    crops = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(3)]
    copies = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copies.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    actual = analyzer._run_facexformer_batch(crops)
    assert actual == [dict(orientation=toward, age=float(i + (20 if scalar_age else 2)),
                           gender=["male", "female"][i % 2], race=i + 1)
                      for i, toward in enumerate(["front", "back", "other"])]
    assert copies == [(3, 7 if classifier else 6)]
    assert calls == [3]
    assert analyzer._run_facexformer_batch([]) == [] and calls == [3]
    assert actual == [analyzer._run_facexformer(crop) for crop in crops]


@pytest.mark.parametrize("new_heads", [False, True])
def test_swinface_batch_decodes_each_target_and_copies_once(monkeypatch, new_heads):
    module = load_module(monkeypatch, "analyzer_batch_sw", SDK_SRC / "face_attr/analyzer.py")
    calls = []

    def model(images, sdk_only=False):
        assert sdk_only
        calls.append(len(images))
        ids = images[:, 0, 0, 0].long()
        binary = torch.tensor([[1., -1.], [-1., 1.], [2., 0.]])[ids]
        output = dict(Age=(ids.float() + 23.7).unsqueeze(1), Expression=torch.eye(7)[ids + 3],
                      Smiling=binary, Eyeglasses=torch.eye(3)[ids] if new_heads else binary,
                      Mustache=binary, **{"Wearing Hat": -binary, "No Beard": -binary})
        if new_heads:
            output["Mask"] = torch.tensor([[-1.], [1.], [2.]])[ids]
        return output

    analyzer = module.AttributeAnalyzer()
    analyzer._swinface = model
    analyzer._preprocess_swinface = lambda crop: torch.full((1, 3, 4, 4), float(crop[0, 0, 0]))
    crops = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(3)]
    copies = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copies.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    actual = analyzer._run_swinface_batch(crops)
    assert calls == [3] and len(copies) == 1 and copies[0][0] == 3
    assert [row["expression"] for row in actual] == ["happy", "sad", "angry"]
    if new_heads:
        assert [row["glasses_class"] for row in actual] == [0, 1, 2]
        assert actual[0]["mask"] < .5 < actual[1]["mask"]
    else:
        assert actual[0]["eyeglasses"] < .5 < actual[1]["eyeglasses"]
    assert actual[0]["mustache"] < .5 < actual[1]["mustache"]
    assert analyzer._run_swinface_batch([]) == [] and calls == [3]
    assert actual == [analyzer._run_swinface(crop) for crop in crops]


@pytest.mark.parametrize("classifier", [False, True])
def test_facexformer_packed_results_preserve_values_and_copy_once(monkeypatch, tmp_path, classifier):
    monkeypatch.delenv("FACEXFORMER_GENDER_ORDER", raising=False)
    model = TinyFaceXFormer()
    with torch.no_grad():
        model.pose.copy_(torch.tensor([0.0, 0.8, 0.0]))
        model.race_logits[3] = 5
    checkpoint = {"state_dict_backbone": model.state_dict()}
    if classifier:
        model.toward_classifier = torch.nn.Linear(3, 3)
        with torch.no_grad():
            model.toward_classifier.weight.zero_()
            model.toward_classifier.bias.copy_(torch.tensor([10., 0., 0.]))
        checkpoint.update(state_dict_backbone=model.state_dict(), toward_classes=["front", "back", "other"])
    path = tmp_path / "fx.pt"
    torch.save(checkpoint, path)
    analyzer = load_analyzer(monkeypatch, path)
    copies = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    assert analyzer._run_facexformer(np.zeros((4, 4, 3), dtype=np.uint8)) == {
        "orientation": "front" if classifier else "other", "age": 0.0, "gender": "male", "race": 3}
    assert len(copies) == 1
