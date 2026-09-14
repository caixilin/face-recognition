"""人脸检测模块测试。"""

import numpy as np
import pytest

from face_attr.detector import FaceBox, FaceDetector


def test_facebox_dataclass():
    box = FaceBox(0, 0, 10, 10, 0.9)
    assert box.x_min == 0
    assert box.x_max == 10
    assert box.confidence == 0.9


def test_detect_blank_image_returns_empty():
    detector = FaceDetector()
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    faces = detector.detect(blank)
    assert faces == []
