"""Check diagnostic IoU and real overlay/report output without MTCNN weights."""

import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).parents[1] / "project/ev_sdk/src/check_detection_boxes.py"
spec = importlib.util.spec_from_file_location("detection_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_nested_face_box_fails_head_iou_threshold():
    assert check.box_iou((0, 0, 40, 40), (10, 10, 30, 30)) == 0.25
    assert check.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert check.box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0


def test_orientation_matching_does_not_reuse_a_prediction():
    heads = [{"bbox": (0, 0, 20, 20)}, {"bbox": (0, 0, 20, 20)}]
    matches = check.match_heads(heads, [(0, 0, 20, 20)], [0.9])
    assert len(matches) == 1
    assert len(set(matches.values())) == 1
    assert check.match_heads(heads, [(40, 40, 50, 50)], [0.9]) == {}


def test_detection_reasons_account_for_all_unmatched_boxes():
    heads = [{"bbox": box} for box in [(0, 0, 10, 10), (4, 0, 14, 10),
                                       (20, 0, 30, 10), (40, 0, 50, 10)]]
    boxes = [(0, 0, 14, 10), (4, 0, 14, 10), (20, 0, 25, 5), (60, 0, 70, 10)]
    matches, details, stats = check.diagnose_matching(heads, boxes, [.9, .8, .7, .6])
    assert len(matches) == 1
    assert (stats["tp"], stats["fp"], stats["fn"]) == (1, 3, 3)
    assert stats["unmatched_gt_reasons"] == {
        "no_overlapping_prediction": 1, "iou_below_0.5": 1, "matching_conflict": 1}
    assert stats["unmatched_prediction_reasons"] == {
        "no_gt_overlap": 1, "iou_below_0.5": 1, "duplicate_or_conflict": 1}
    assert sorted(detail["match_reason"] for detail in details) == sorted([
        "matched", "matching_conflict", "iou_below_0.5", "no_overlapping_prediction"])


def test_threshold_sweep_exposes_recall_and_false_positive_tradeoff():
    rows = [{"heads": [{"bbox": (0, 0, 10, 10)}], "predictions": [], "scores": [],
             "candidate_predictions": [(0, 0, 10, 10), (30, 30, 40, 40)],
             "candidate_scores": [.4, .9]}]
    result = check.summarize_detection(rows, [.3, .5, .7])
    assert result["0.3"]["tp"] == 1 and result["0.3"]["fp"] == 1
    assert result["0.3"]["precision"] == .5 and result["0.3"]["recall"] == 1
    assert result["0.5"]["tp"] == 0 and result["0.5"]["fn"] == 1
    assert result["0.7"] == result["0.5"]
    empty = check.summarize_detection([{"heads": [], "predictions": [], "scores": []}], [.5])["0.5"]
    assert empty["precision"] is None and empty["recall"] is None


def test_validation_paths_reuse_real_training_split_and_reject_different_counts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.syspath_prepend(str(SCRIPT.parents[2] / "train" / "src_repo"))
    from detection_dataset import HeadShoulderDataset
    from validation_metrics import split_image_indices
    for number in range(10):
        name = f"sample{number}"
        Image.new("RGB", (60, 60)).save(tmp_path / (name + ".png"))
        (tmp_path / (name + ".xml")).write_text(
            f"<annotation><filename>{name}.png</filename><object><name>head</name><toward>back</toward>"
            "<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>"
            "<object><name>person</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>40</xmax>"
            "<ymax>60</ymax></bndbox></object></annotation>", encoding="utf-8")
    dataset = HeadShoulderDataset(tmp_path, max_images=0, seed=42)
    train_ids, val_ids = split_image_indices(dataset.samples)
    expected = {str(Path(dataset.samples[index][0]).resolve()) for index in val_ids}
    paths, info = check.validation_image_paths(tmp_path, expected_images=10)
    assert paths == expected and len(paths) == 2
    assert paths.isdisjoint(str(Path(dataset.samples[index][0]).resolve()) for index in train_ids)
    assert info["train_images"] == 8 and info["validation_images"] == 2
    with pytest.raises(ValueError, match="不符"):
        check.validation_image_paths(tmp_path, expected_images=11)


def test_orientation_summary_separates_detection_and_classifier_failures():
    def head(matched, sdk, train):
        return {"toward": "back", "matched_prediction": matched,
                "gt_crop": {"sdk": "back", "train": "back"},
                "detected_crop": {} if matched is None else {"sdk": sdk, "train": train}}
    row = check.summarize_orientations([{"heads": [head(None, None, None),
        head(0, "other", "back"), head(1, "back", "back")]}])["back"]
    assert row["gt"] == 3 and row["unmatched"] == 1 and row["matched"] == 2
    assert row["matched_wrong_orientation_sdk"] == 1
    assert row["matched_wrong_orientation_train"] == 0
    assert row["gt_crop_accuracy_sdk"] == 1
    assert row["matched_crop_accuracy_sdk"] == .5
    assert row["end_to_end_accuracy_sdk"] == pytest.approx(1/3)
    assert row["end_to_end_accuracy_train"] == pytest.approx(2/3)


def test_orientation_preprocess_is_restored_after_failed_comparison(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    class Analyzer:
        _preprocess_facexformer = staticmethod(lambda image: image)
        calls = 0
        def _run_facexformer(self, image):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("comparison failed")
            return {"orientation": "back"}
    analyzer = Analyzer()
    original = analyzer._preprocess_facexformer
    predict = check.orientation_predictor(analyzer)
    with pytest.raises(RuntimeError, match="comparison failed"):
        predict(None)
    assert analyzer._preprocess_facexformer is original


def test_crop_pairs_use_the_same_matched_targets_and_show_both_directions():
    heads = []
    for index, (gt, detected) in enumerate([("back", "back"), ("back", "other"),
                                           ("other", "back"), ("other", "other")]):
        heads.append({"toward": "back", "bbox": (0, 0, 10, 10), "matched_prediction": index,
                      "gt_crop": {"sdk": gt, "train": "back"},
                      "detected_crop": {"sdk": detected, "train": "back"}})
    heads.append({"toward": "back", "matched_prediction": None,
                  "gt_crop": {"sdk": "back"}, "detected_crop": {}})
    rows = [{"image": "scene.jpg", "heads": heads, "predictions": [(0, 0, 10, 10)] * 4}]
    paired = check.summarize_crop_pairs(rows)["back"]
    assert paired["sdk"]["matched"] == 4
    assert [paired["sdk"][key] for key in ("both_correct", "gt_correct_detected_wrong",
                                          "gt_wrong_detected_correct", "both_wrong")] == [1, 1, 1, 1]
    assert paired["train"]["both_correct"] == 4
    cases = check.back_error_examples(rows)
    assert [row["head_index"] for row in cases] == [1, 3]
    assert cases[0]["gt_crop_prediction"] == "back" and cases[1]["gt_crop_prediction"] == "other"
    assert len(check.back_error_examples(rows, limit=1)) == 1


def test_orientation_scores_are_stable_for_large_logits_and_preserve_margin():
    scores = check.orientation_scores([[1000, 1003, 1001]])
    assert sum(scores["softmax"].values()) == pytest.approx(1)
    assert scores["softmax"]["back"] > scores["softmax"]["other"] > scores["softmax"]["front"]
    assert scores["back_minus_other_logit"] == 2


@pytest.mark.parametrize("logits", [[0, 1], [0, float("nan"), 1]])
def test_invalid_orientation_scores_fail_explicitly(logits):
    with pytest.raises(ValueError, match="logits"):
        check.orientation_scores(logits)


@pytest.mark.parametrize("fail_on_train", [False, True])
def test_score_hook_observes_existing_forward_and_is_always_removed(monkeypatch, fail_on_train):
    import sys
    import types
    import torch
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    class Analyzer:
        def __init__(self):
            self._facexformer = torch.nn.Module()
            self._facexformer.toward_classifier = torch.nn.Identity()
            self.original = self._preprocess_facexformer = lambda image: image
            self.calls = 0
        def _run_facexformer(self, crop):
            self.calls += 1
            sdk = self._preprocess_facexformer is self.original
            logits = self._facexformer.toward_classifier(torch.tensor([[0., 3., 1.]] if sdk else [[0., 1., 4.]]))
            if not sdk and fail_on_train:
                raise RuntimeError("inference failed")
            return {"orientation": ("front", "back", "other")[int(logits.argmax(1).item())]}
    analyzer = Analyzer()
    predictor = check.orientation_predictor(analyzer, include_scores=True)
    if fail_on_train:
        with pytest.raises(RuntimeError, match="inference failed"):
            predictor(None)
    else:
        result = predictor(None)
        assert result["sdk"] == "back" and result["train"] == "other"
        assert result["scores"]["sdk"]["back_minus_other_logit"] == 2
        assert result["scores"]["train"]["back_minus_other_logit"] == -3
    assert analyzer.calls == 2
    assert analyzer._preprocess_facexformer is analyzer.original
    assert not analyzer._facexformer.toward_classifier._forward_hooks


def test_orientation_mode_writes_one_report_and_skips_unlabelled_crops(tmp_path, monkeypatch):
    import sys
    import types
    cv2 = types.ModuleType("cv2")
    cv2.imread = lambda path: np.zeros((60, 60, 3), dtype=np.uint8)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    for name, toward in (("scene", "<toward>back</toward>"), ("crop", "")):
        Image.new("RGB", (60, 60)).save(tmp_path / (name + ".png"))
        (tmp_path / (name + ".xml")).write_text(
            f"<annotation><filename>{name}.png</filename><object><name>head</name>{toward}"
            "<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>30</xmax><ymax>30</ymax>"
            "</bndbox></object></annotation>", encoding="utf-8")
    class Detector:
        def detect(self, image):
            return np.array([[0, 0, 30, 30]]), np.array([.9])
    output = tmp_path / "diagnosis"
    report = check.run_orientation_check(tmp_path, output, Detector(),
        lambda crop: {"sdk": "other", "train": "back"}, max_images=10)
    assert report["images"] == 1
    assert report["counts"]["back"]["matched_wrong_orientation_sdk"] == 1
    assert report["counts"]["back"]["end_to_end_accuracy_train"] == 1
    assert report["detection_thresholds"]["0.5"]["fp"] == 0
    assert report["paired_crops"]["back"]["sdk"]["both_wrong"] == 1
    assert report["paired_crops"]["back"]["train"]["both_correct"] == 1
    assert sorted(path.name for path in output.iterdir()) == ["orientation_check.json"]


def test_validation_mode_filters_train_images_and_counts_heads_without_toward(tmp_path, monkeypatch):
    import sys
    import types
    cv2 = types.ModuleType("cv2")
    cv2.imread = lambda path: np.zeros((60, 60, 3), dtype=np.uint8)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    for name in ("train", "validation"):
        Image.new("RGB", (60, 60)).save(tmp_path / (name + ".png"))
        (tmp_path / (name + ".xml")).write_text(
            f"<annotation><filename>{name}.png</filename><object><name>head</name>"
            "<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>20</xmax><ymax>20</ymax>"
            "</bndbox></object></annotation>", encoding="utf-8")
    class Detector:
        def detect(self, image):
            return np.array([[0, 0, 20, 20], [40, 40, 50, 50]]), np.array([.4, .9])
    report = check.run_orientation_check(tmp_path, tmp_path / "output", Detector(),
        lambda crop: pytest.fail("unknown toward should not invoke the classifier"),
        validation_paths={str((tmp_path / "validation.png").resolve())},
        split_info={"mode": "reconstructed_head_validation"}, thresholds=(.3, .5, .7))
    assert report["images"] == 1 and report["per_image"][0]["image"].endswith("validation.png")
    assert report["detection_thresholds"]["0.3"]["tp"] == 1
    assert report["detection_thresholds"]["0.5"]["fn"] == 1
    assert report["counts"]["back"]["gt"] == 0
    assert "possible_training_images" not in report["limitations"]
    assert "not_guaranteed_attribute_model_holdout" in report["limitations"]


@pytest.mark.parametrize("detected,expected_coverage", [(True, 1), (False, 0)])
def test_writes_gt_prediction_overlay_and_report(tmp_path, detected, expected_coverage):
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (60, 60), "black").save(data / "sample.png")
    (data / "sample.xml").write_text("""<annotation><filename>sample.png</filename>
    <object><name>head</name><bndbox><xmin>0</xmin><ymin>0</ymin>
    <xmax>40</xmax><ymax>40</ymax></bndbox><attributes><attribute>
    <name>toward</name><value>other</value></attribute></attributes></object>
    <object><name>head</name><bndbox><xmin>10</xmin><ymin>10</ymin>
    <xmax>30</xmax><ymax>30</ymax></bndbox><toward>front</toward></object>
    <object><name>person</name></object></annotation>""", encoding="utf-8")

    class Detector:
        def detect(self, image):
            assert image.shape == (60, 60, 3)
            return (np.array([[10.9, 10.9, 30.9, 30.9]]), None) if detected else (None, None)

    output = tmp_path / "output"
    report = check.run_check(data, output, Detector())
    assert report["images"] == 1
    assert report["gt_heads"] == 2
    assert report["gt_with_iou_ge_0_5_candidate"] == expected_coverage
    row = report["per_image"][0]
    assert [head["toward"] for head in row["ground_truth"]] == ["other", "front"]
    assert [head["best_iou"] for head in row["ground_truth"]] == ([0.25, 1.0] if detected else [0, 0])
    with Image.open(row["overlay"]) as overlay:
        assert overlay.size == (60, 60)
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["gt_with_iou_ge_0_5_candidate"] == expected_coverage


def test_missing_annotations_has_actionable_error(tmp_path):
    with pytest.raises(RuntimeError, match="没有找到 XML"):
        check.run_check(tmp_path, tmp_path / "output", None)


@pytest.mark.parametrize("detected", [True, False])
def test_single_image_control_uses_rgb_and_needs_no_xml(tmp_path, capsys, detected):
    image_path = tmp_path / "front.png"
    Image.new("RGB", (80, 60), (50, 100, 150)).save(image_path)

    class Detector:
        def detect(self, image):
            assert image[0, 0].tolist() == [50, 100, 150]
            return (np.array([[10.5, 10.5, 30.5, 30.5]]), np.array([0.95])) if detected else (None, None)

    output = tmp_path / "control"
    report = check.run_single_image(image_path, output, Detector(), "cpu")
    assert report["predicted_faces"] == int(detected)
    assert report["image_size"] == [80, 60]
    assert report["predictions"] == ([(10, 10, 30, 30)] if detected else [])
    assert f"PRED={int(detected)}" in capsys.readouterr().out
    saved = json.loads((output / "control_report.json").read_text(encoding="utf-8"))
    assert saved["device"] == "cpu"
    assert saved["probabilities"] == ([0.95] if detected else [])
    assert Path(saved["overlay"]).is_file()
