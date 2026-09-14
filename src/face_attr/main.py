"""人脸属性分析主入口。

流程：读取图像 -> 人脸检测 -> 属性分析 -> 输出结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from face_attr.analyzer import AttributeAnalyzer
from face_attr.detector import FaceDetector


def process_image(image_path: str, device: str = "cpu") -> dict:
    """处理单张图像，返回人脸属性结果。"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    detector = FaceDetector(device=device)
    analyzer = AttributeAnalyzer(device=device)

    faces = detector.detect(image)
    results = []
    for box in faces:
        crop = detector.crop(image, box)
        attrs = analyzer.analyze(crop)
        results.append(
            {
                "bbox": [box.x_min, box.y_min, box.x_max, box.y_max],
                "confidence": box.confidence,
                "attributes": {
                    "age": attrs.age,
                    "gender": attrs.gender,
                    "expression": attrs.expression,
                    "orientation": attrs.orientation,
                },
            }
        )

    return {"face_count": len(results), "faces": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="人脸属性分析")
    parser.add_argument("image", type=str, help="输入图像路径")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备 (cpu/cuda)")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    result = process_image(args.image, device=args.device)
    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"结果已保存到: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
