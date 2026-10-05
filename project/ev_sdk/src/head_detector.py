"""Person/head-shoulder detector shared by training, SDK and diagnostics."""

from pathlib import Path
import math
import numpy as np
import torch

DETECTOR_CLASSES = ["background", "person", "head"]
ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
COCO_URL = "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"


def _freeze_batch_norm(module):
    # Torchvision changes normalization with pretrained=True. Use the same
    # architecture for offline SDK loading and COCO-initialized training.
    from torchvision.ops import FrozenBatchNorm2d
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            replacement = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                for key in ("weight", "bias", "running_mean", "running_var"):
                    getattr(replacement, key).copy_(getattr(child, key))
            setattr(module, name, replacement)
        else:
            _freeze_batch_norm(child)


def build_head_detector(min_size=640, max_size=1280, pretrained=False, coco_path=None):
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        pretrained=False, pretrained_backbone=False, min_size=min_size, max_size=max_size,
        rpn_pre_nms_top_n_test=300, rpn_post_nms_top_n_test=300,
        box_detections_per_img=200,
    )
    _freeze_batch_norm(model.backbone)
    if pretrained:
        weights = (torch.load(coco_path, map_location="cpu") if coco_path else
                   torch.hub.load_state_dict_from_url(COCO_URL, map_location="cpu", progress=True))
        model.load_state_dict(weights)
    old = model.roi_heads.box_predictor
    predictor = FastRCNNPredictor(old.cls_score.in_features, len(DETECTOR_CLASSES))
    if pretrained:
        # Retain the COCO background and person rows; the new head class is learned.
        with torch.no_grad():
            predictor.cls_score.weight[:2].copy_(old.cls_score.weight[:2])
            predictor.cls_score.bias[:2].copy_(old.cls_score.bias[:2])
            predictor.bbox_pred.weight[:8].copy_(old.bbox_pred.weight[:8])
            predictor.bbox_pred.bias[:8].copy_(old.bbox_pred.bias[:8])
    model.roi_heads.box_predictor = predictor
    return model


def load_head_detector(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"头肩检测权重不存在: {path}；先运行 train_head_detector.py")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("classes") != DETECTOR_CLASSES:
        raise ValueError("不是本项目的 person/head 头肩检测权重，不能用 MTCNN 或 COCO 权重代替")
    model = build_head_detector(checkpoint["min_size"], checkpoint["max_size"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def _clip_box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        return None
    x1, y1 = max(0, math.floor(float(box[0]))), max(0, math.floor(float(box[1])))
    x2, y2 = min(width, math.ceil(float(box[2]))), min(height, math.ceil(float(box[3])))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def decode_detections(output, width, height, score_threshold=0.5):
    """Pair real person detections to real head detections; never synthesize boxes."""
    from model_api import FaceDetection
    people, heads = [], []
    for box, label, score in zip(output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist()):
        clipped = _clip_box(box, width, height)
        if clipped is None or not math.isfinite(score) or score < score_threshold:
            continue
        if label in (1, 2):
            (people if label == 1 else heads).append((clipped, score))
    people.sort(key=lambda item: -item[1])
    heads.sort(key=lambda item: -item[1])
    paired, used = [], set()
    for head, score in heads:
        area = (head[2] - head[0]) * (head[3] - head[1])
        candidates = []
        for index, (person, _) in enumerate(people):
            if index in used:
                continue
            overlap = max(0, min(head[2], person[2]) - max(head[0], person[0])) * max(
                0, min(head[3], person[3]) - max(head[1], person[1]))
            cx, cy = (head[0] + head[2]) / 2, (head[1] + head[3]) / 2
            if person[0] <= cx <= person[2] and person[1] <= cy <= person[3] and overlap / area >= 0.5:
                candidates.append((overlap / area, people[index][1], index))
        person = None
        if candidates:
            index = max(candidates)[2]
            used.add(index)
            person = people[index][0]
        paired.append((head, person, score))
    paired.extend((None, person, score) for index, (person, score) in enumerate(people) if index not in used)

    def xywh(box):
        return None if box is None else (box[0], box[1], box[2] - box[0], box[3] - box[1])
    return [FaceDetection(xywh(head), xywh(person), score, str(index))
            for index, (head, person, score) in enumerate(paired, 1)]


class HeadShoulderDetector:
    def __init__(self, checkpoint_path, device="cpu", score_threshold=0.5):
        if not 0 <= score_threshold <= 1:
            raise ValueError("检测置信度阈值必须在 0–1 内")
        self.device = torch.device(device)
        self.model = load_head_detector(checkpoint_path, self.device)
        self.score_threshold = score_threshold

    @torch.no_grad()
    def __call__(self, image_bgr):
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        # Transfer uint8 pixels first (one byte instead of four per channel),
        # then retain the existing float32 / 255 input on the target device.
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(self.device).float().div_(255)
        output = self.model([tensor])[0]
        # Copy boxes, scores and labels together rather than synchronizing for
        # each tensor's tolist() inside the decoder.
        packed = torch.cat((output["boxes"], output["scores"].unsqueeze(1),
                            output["labels"].unsqueeze(1).to(output["boxes"].dtype)), dim=1).cpu()
        cpu_output = {"boxes": packed[:, :4], "scores": packed[:, 4],
                      "labels": packed[:, 5].long()}
        return decode_detections(cpu_output, image_bgr.shape[1], image_bgr.shape[0], self.score_threshold)

    def detect(self, image_rgb):
        """Diagnostic interface: only head boxes, in original RGB image coordinates."""
        detections = self(np.ascontiguousarray(image_rgb[:, :, ::-1]))
        heads = [item for item in detections if item.head_bbox is not None]
        if not heads:
            return None, None
        boxes = [(x, y, x + w, y + h) for x, y, w, h in (item.head_bbox for item in heads)]
        return np.asarray(boxes), np.asarray([item.confidence for item in heads])


def build_legacy_mtcnn_detector(device):
    """Preserve the already-tested SDK while no new head-shoulder weights exist."""
    from facenet_pytorch import MTCNN
    from model_api import FaceDetection
    model = MTCNN(keep_all=True, device=device)

    def detect(image):
        boxes, probabilities = model.detect(image[:, :, ::-1])
        if boxes is None:
            return []
        if probabilities is None:
            probabilities = np.ones(len(boxes), dtype=np.float32)
        order = sorted(range(len(boxes)), key=lambda index: -float(probabilities[index]))
        result = []
        for number, index in enumerate(order, 1):
            x1, y1, x2, y2 = [int(value) for value in boxes[index]]
            width, height = x2 - x1, y2 - y1
            px, py = max(0, x1 - width // 2), max(0, y1 - height)
            result.append(FaceDetection((x1, y1, width, height),
                (px, py, min(image.shape[1], x2 + width // 2) - px,
                 min(image.shape[0], y2 + height * 2) - py), float(probabilities[index]), str(number)))
        return result
    return detect
