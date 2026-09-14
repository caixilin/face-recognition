"""属性分析模块测试。"""

import numpy as np

from face_attr.analyzer import AttributeAnalyzer, FaceAttributes


def test_analyzer_returns_empty_attributes():
    analyzer = AttributeAnalyzer()
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    attrs = analyzer.analyze(crop)
    assert isinstance(attrs, FaceAttributes)
    assert attrs.age is None
    assert attrs.gender is None
