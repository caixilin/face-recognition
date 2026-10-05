"""Real detector loss/checkpoint tests plus VOC geometry and sampling checks."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project" / "ev_sdk" / "src"))
sys.path.insert(0, str(ROOT / "project" / "train" / "src_repo"))
from head_detector import ARCHITECTURE, DETECTOR_CLASSES, HeadShoulderDetector, build_head_detector, decode_detections, load_head_detector
from detection_dataset import HeadShoulderDataset
from validation_metrics import split_image_indices, require_training_classes


def test_head_boxes_are_learned_coordinates_and_person_only_boxes_survive():
    output = {"boxes": torch.tensor([[5., 5., 45., 75.], [10., 5., 35., 30.],
                                     [60., 10., 85., 40.], [95., 5., 105., 90.],
                                     [float("nan"), 0., 30., 30.]]),
              "labels": torch.tensor([1, 2, 2, 1, 2]),
              "scores": torch.tensor([.9, .85, .8, .7, .95])}
    rows = decode_detections(output, 100, 80)
    assert rows[0].head_bbox == (10, 5, 25, 25)
    assert rows[0].person_bbox == (5, 5, 40, 70)
    assert rows[0].target_id == "1"
    assert rows[1].head_bbox == (60, 10, 25, 30)
    assert rows[1].person_bbox is None  # no inferred/expanded person box
    assert rows[2].head_bbox is None
    assert rows[2].person_bbox == (95, 5, 5, 75)


def test_voc_detection_dataset_excludes_attribute_crops_and_clips_boxes(tmp_path):
    for index in range(3):
        Image.new("RGB", (100, 80), (20, 40, 60)).save(tmp_path / f"image{index}.png")
        if index < 2:
            body = """<object><name>person</name><bndbox><xmin>-4</xmin><ymin>-2</ymin>
            <xmax>40</xmax><ymax>100</ymax></bndbox></object>
            <object><name>head</name><toward>back</toward><bndbox><xmin>0</xmin><ymin>0</ymin>
            <xmax>30</xmax><ymax>30</ymax></bndbox></object>"""
        else:
            body = """<object><name>head</name><glasses>2</glasses><bndbox><xmin>0</xmin><ymin>0</ymin>
            <xmax>100</xmax><ymax>80</ymax></bndbox></object>"""
        (tmp_path / f"image{index}.xml").write_text(
            f"<annotation><filename>image{index}.png</filename>{body}</annotation>", encoding="utf-8")
    dataset = HeadShoulderDataset(tmp_path)
    assert len(dataset) == 2
    image, target = dataset[0]
    assert image.shape == (3, 80, 100)
    assert image[:, 0, 0].tolist() == pytest.approx([20/255, 40/255, 60/255])
    assert target["labels"].tolist() == [1, 2]
    assert target["boxes"].tolist() == [[0, 0, 40, 80], [0, 0, 30, 30]]


def test_image_split_keeps_all_targets_of_same_image_together():
    samples = [("same.jpg", {}), ("other.jpg", {}), ("same.jpg", {}), ("third.jpg", {})]
    train, val = split_image_indices(samples)
    assert ({samples[index][0] for index in train} & {samples[index][0] for index in val}) == set()
    assert (0 in train) == (2 in train)
    assert sorted(train + val) == [0, 1, 2, 3]


def test_real_detector_loss_and_offline_sdk_checkpoint_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: pytest.fail("SDK must not download weights"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = build_head_detector(min_size=64, max_size=96)
        model.rpn._pre_nms_top_n = {"training": 30, "testing": 20}
        model.rpn._post_nms_top_n = {"training": 20, "testing": 20}
        model.train()
        target = {"boxes": torch.tensor([[1., 1., 50., 62.], [5., 5., 35., 30.]]),
                  "labels": torch.tensor([1, 2])}
        image = torch.rand(3, 64, 64)
        losses = model([image], [target])
        loss = sum(losses.values())
        assert torch.isfinite(loss)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        assert model.roi_heads.box_predictor.cls_score.weight.grad is not None
        optimizer.step()
        path = tmp_path / "head.pt"
        torch.save({"architecture": ARCHITECTURE, "classes": DETECTOR_CLASSES,
                    "min_size": 64, "max_size": 96, "state_dict": model.state_dict()}, path)
        loaded = load_head_detector(path, "cpu")
        with torch.no_grad():
            output = loaded([image])[0]
        assert set(output) == {"boxes", "labels", "scores"}
        assert output["labels"].ge(1).all() and output["labels"].le(2).all()
        assert output["boxes"].ge(0).all() and output["boxes"].le(64).all()
    finally:
        torch.set_num_threads(old_threads)


def test_generic_coco_checkpoint_is_not_accepted_as_head_model(tmp_path):
    path = tmp_path / "coco.pt"
    torch.save({"classes": ["person"]}, path)
    with pytest.raises(ValueError, match="不能用 MTCNN 或 COCO"):
        load_head_detector(path, "cpu")


@pytest.mark.parametrize("empty", [False, True])
def test_detector_packed_results_preserve_boxes_and_copy_once(monkeypatch, empty):
    output = {"boxes": torch.tensor([[1.2, 2.4, 50.3, 70.7], [4.1, 3.2, 25.9, 30.6]]),
              "labels": torch.tensor([1, 2]), "scores": torch.tensor([0.9, 0.8])}
    if empty:
        output = {key: value[:0] for key, value in output.items()}
    expected = decode_detections(output, 100, 80)
    detector = HeadShoulderDetector.__new__(HeadShoulderDetector)
    detector.device, detector.score_threshold = torch.device("cpu"), 0.5
    detector.model = lambda images: [output]
    copies = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    assert detector(np.zeros((80, 100, 3), dtype=np.uint8)) == expected
    assert len(copies) == 1


def test_detector_transfers_bytes_before_normalizing_and_preserves_rgb_input(monkeypatch):
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    original_image = image.copy()
    expected = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1]).transpose(2, 0, 1)).float().div(255)
    detector = HeadShoulderDetector.__new__(HeadShoulderDetector)
    detector.device, detector.score_threshold = torch.device("cpu"), .5
    transfers = []
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if args == (detector.device,):
            transfers.append((tensor.dtype, tensor.numel() * tensor.element_size()))
        return original_to(tensor, *args, **kwargs)

    def model(images):
        assert images[0].dtype == torch.float32
        torch.testing.assert_close(images[0], expected, rtol=0, atol=0)
        return [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long),
                 "scores": torch.empty(0)}]

    monkeypatch.setattr(torch.Tensor, "to", to)
    detector.model = model
    assert detector(image) == []
    assert transfers == [(torch.uint8, image.size)]
    np.testing.assert_array_equal(image, original_image)


def test_missing_sunglasses_or_mask_supervision_stops_training():
    samples = [("one.jpg", {"glasses": 0, "mask": 0}), ("two.jpg", {"glasses": 1, "mask": 1})]
    with pytest.raises(ValueError, match="缺少任务类别"):
        require_training_classes(samples, [0, 1], {"glasses": 3, "mask": 2})
    samples.append(("three.jpg", {"glasses": 2, "mask": 1}))
    assert require_training_classes(samples, [0, 1, 2], {"glasses": 3, "mask": 2}) == {
        "glasses": [1, 1, 1], "mask": [1, 2]}
