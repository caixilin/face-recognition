"""Compare VOC boxes; optionally diagnose orientation with explicit model weights."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area2 = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def parse_heads(root):
    heads = []
    for obj in root.findall("object"):
        if (obj.findtext("name") or "").strip().lower() != "head":
            continue
        try:
            bbox = tuple(float(obj.findtext("bndbox/" + key)) for key in
                         ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(bbox)) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        toward = (obj.findtext("toward") or "unknown").strip()
        for attribute in obj.findall("attributes/attribute"):
            if (attribute.findtext("name") or "").strip().lower() == "toward":
                toward = (attribute.findtext("value") or "unknown").strip()
        heads.append({"bbox": bbox, "toward": toward})
    return heads


def find_image(xml_path, root, data_root, index):
    filename = (root.findtext("filename") or "").strip()
    # VOC exports can contain Windows paths even when used on Linux.
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    candidates = []
    if name:
        for directory in (xml_path.parent, xml_path.parent.parent / "images",
                          data_root / "images", data_root):
            candidates.append(directory / name)
    candidates.extend(xml_path.with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = index.get(name.lower(), [])
    # Prefer an image from the same data source when basenames repeat.
    relative = xml_path.relative_to(data_root)
    source = data_root / relative.parts[0] if len(relative.parts) > 1 else data_root
    local_matches = [path for path in matches if source in path.parents]
    choices = local_matches or matches
    return choices[0] if len(choices) == 1 else None


def run_single_image(image_path, output_root, detector, device=None, detector_name="MTCNN"):
    """Positive control: detect a clear face without annotations or attribute weights."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    boxes, probabilities = detector.detect(np.array(image))
    predictions = [] if boxes is None else [tuple(int(value) for value in box)
                                            for box in boxes]
    scores = [] if probabilities is None else [float(value) for value in probabilities]
    output_root.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(image)
    for number, box in enumerate(predictions, 1):
        draw.rectangle(box, outline=(255, 0, 0), width=3)
        draw.text((max(0, box[0]), max(20, box[1])), f"PRED{number}", fill=(255, 0, 0))
    output_path = output_root / f"control_{image_path.stem}.jpg"
    image.save(output_path, quality=95)
    report = {"image": str(image_path), "image_size": list(image.size), "device": device,
              "detector": detector_name,
              "predicted_faces": len(predictions), "predictions": predictions,
              "probabilities": scores, "overlay": str(output_path),
              "note": "单图检测对照；没有标注，不计算 IoU 或平台分数。"}
    if detector_name != "MTCNN":
        report["predicted_heads"] = report.pop("predicted_faces")
    report_path = output_root / "control_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"IMAGE={image_path} SIZE={image.width}x{image.height} PRED={len(predictions)}", flush=True)
    print(f"对照图: {output_path}", flush=True)
    print(f"报告: {report_path}", flush=True)
    return report


def run_check(data_root, output_root, detector, max_images=12, seed=42, detector_name="MTCNN"):
    xml_paths = sorted(data_root.rglob("*.xml"))
    if not xml_paths:
        raise RuntimeError(f"没有找到 XML 标注: {data_root}。请指定可访问的图片和标注目录。")
    index = defaultdict(list)
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name.lower()].append(path)
    random.Random(seed).shuffle(xml_paths)
    output_root.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
            gt = parse_heads(root)
            image_path = find_image(xml_path, root, data_root, index)
        except (ET.ParseError, OSError):
            skipped += 1
            continue
        if not gt or image_path is None:
            skipped += 1
            continue
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError):
            skipped += 1
            continue
        boxes, _ = detector.detect(np.array(image))
        # Both diagnostics and SDK use original-image coordinates.
        predictions = [] if boxes is None else [tuple(int(value) for value in box)
                                                for box in boxes]
        for head in gt:
            head["best_iou"] = max((box_iou(head["bbox"], box) for box in predictions),
                                   default=0.0)
        draw = ImageDraw.Draw(image)
        for number, head in enumerate(gt, 1):
            bbox = head["bbox"]
            draw.rectangle(bbox, outline=(0, 255, 0), width=3)
            draw.text((max(0, bbox[0]), max(20, bbox[1] - 14)),
                      f"GT{number} {head['toward']} IoU={head['best_iou']:.3f}",
                      fill=(0, 255, 0))
        for number, box in enumerate(predictions, 1):
            draw.rectangle(box, outline=(255, 0, 0), width=2)
            draw.text((max(0, box[0]), max(20, box[1])), f"PRED{number}", fill=(255, 0, 0))
        draw.text((5, 5), f"GREEN=GT head | RED={detector_name} | IoU threshold=0.5", fill="white")
        output_path = output_root / f"{len(rows) + 1:03d}_{xml_path.stem}.jpg"
        image.save(output_path, quality=95)
        row = {"image": str(image_path), "xml": str(xml_path),
               "overlay": str(output_path), "ground_truth": gt, "predictions": predictions}
        rows.append(row)
        covered = sum(head["best_iou"] >= 0.5 for head in gt)
        print(f"{output_path.name}: GT={len(gt)} PRED={len(predictions)} "
              f"GT with IoU>=0.5 candidate={covered}", flush=True)
        if len(rows) >= max_images:
            break
    if not rows:
        raise RuntimeError("没有可用的 head 标注和对应图片；请确认 XML 使用 VOC bndbox 且图片已准备好。")
    heads = [head for row in rows for head in row["ground_truth"]]
    summary = {"images": len(rows), "skipped_xml": skipped, "gt_heads": len(heads),
               "detector": detector_name,
               "predicted_heads": sum(len(row["predictions"]) for row in rows),
               "gt_with_iou_ge_0_5_candidate": sum(head["best_iou"] >= 0.5 for head in heads),
               "note": "诊断候选框覆盖，不是平台分数；每个 GT 独立取最大框 IoU，不做一对一匹配或多边形评分。",
               "per_image": rows}
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"报告: {output_root / 'summary.json'}", flush=True)
    return summary


def match_heads(ground_truth, predictions, probabilities, threshold=0.5):
    """Confidence-first one-to-one rectangle matching for diagnostics only."""
    matches = {}
    order = sorted(range(len(predictions)), key=lambda i: -probabilities[i])
    for index in order:
        candidates = [(box_iou(head["bbox"], predictions[index]), number)
                      for number, head in enumerate(ground_truth) if number not in matches]
        iou, number = max(candidates, default=(0.0, -1))
        if iou >= threshold:
            matches[number] = index
    return matches


def diagnose_matching(heads, predictions, scores):
    """Account for every GT and prediction at rectangle IoU .5."""
    matches = match_heads(heads, predictions, scores)
    gt_details = []
    for number, head in enumerate(heads):
        best = max((box_iou(head["bbox"], box) for box in predictions), default=0.0)
        reason = ("matched" if number in matches else "no_overlapping_prediction" if best == 0
                  else "iou_below_0.5" if best < 0.5 else "matching_conflict")
        gt_details.append({"best_iou": best, "match_reason": reason})
    unused = set(range(len(predictions))) - set(matches.values())
    fp_reasons = {"no_gt_overlap": 0, "iou_below_0.5": 0, "duplicate_or_conflict": 0}
    for number in unused:
        best = max((box_iou(head["bbox"], predictions[number]) for head in heads), default=0.0)
        reason = "no_gt_overlap" if best == 0 else "iou_below_0.5" if best < 0.5 else "duplicate_or_conflict"
        fp_reasons[reason] += 1
    fn_reasons = {"no_overlapping_prediction": 0, "iou_below_0.5": 0, "matching_conflict": 0}
    for detail in gt_details:
        if detail["match_reason"] != "matched":
            fn_reasons[detail["match_reason"]] += 1
    return matches, gt_details, {
        "tp": len(matches), "fp": len(unused), "fn": len(heads) - len(matches),
        "unmatched_gt_reasons": fn_reasons, "unmatched_prediction_reasons": fp_reasons,
    }


def summarize_detection(rows, thresholds):
    results = {}
    for threshold in thresholds:
        totals = {"tp": 0, "fp": 0, "fn": 0,
                  "unmatched_gt_reasons": {"no_overlapping_prediction": 0, "iou_below_0.5": 0, "matching_conflict": 0},
                  "unmatched_prediction_reasons": {"no_gt_overlap": 0, "iou_below_0.5": 0, "duplicate_or_conflict": 0}}
        for row in rows:
            boxes = row.get("candidate_predictions", row["predictions"])
            scores = row.get("candidate_scores", row["scores"])
            selected = [i for i, score in enumerate(scores) if score >= threshold]
            _, _, stats = diagnose_matching(row["heads"], [boxes[i] for i in selected],
                                             [scores[i] for i in selected])
            for key in ("tp", "fp", "fn"):
                totals[key] += stats[key]
            for key in ("unmatched_gt_reasons", "unmatched_prediction_reasons"):
                for reason, count in stats[key].items():
                    totals[key][reason] += count
        tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
        totals["precision"] = tp / (tp + fp) if tp + fp else None
        totals["recall"] = tp / (tp + fn) if tp + fn else None
        totals["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        results[str(threshold)] = totals
    return results


def validation_image_paths(data_root, expected_images=None):
    """Reconstruct the full-data head training split; never start training."""
    import sys
    training_source = Path(__file__).resolve().parents[2] / "train" / "src_repo"
    if not (training_source / "detection_dataset.py").is_file() or not (training_source / "validation_metrics.py").is_file():
        raise FileNotFoundError("留出集检查需要 /project/train/src_repo/detection_dataset.py 和 validation_metrics.py")
    sys.path.insert(0, str(training_source))
    try:
        from detection_dataset import HeadShoulderDataset
        from validation_metrics import split_image_indices
        dataset = HeadShoulderDataset(data_root, max_images=0, seed=42)
        if expected_images is not None and len(dataset) != expected_images:
            raise ValueError(f"检测数据共 {len(dataset)} 张，与指定训练数量 {expected_images} 不符；请核对数据集挂载，不能称为同一留出集")
        train_ids, val_ids = split_image_indices(dataset.samples, val_fraction=0.2, seed=42)
        paths = {str(Path(dataset.samples[index][0]).resolve()) for index in val_ids}
        info = {"mode": "reconstructed_head_validation", "split_seed": 42, "val_fraction": 0.2,
                "detection_images": len(dataset), "train_images": len(train_ids), "validation_images": len(val_ids),
                "expected_detection_images": expected_images,
                "requires_same_data_as_training": True, "attribute_model_holdout_not_guaranteed": True}
        print("头肩留出集重建: " + json.dumps(info, ensure_ascii=False), flush=True)
        return paths, info
    finally:
        sys.path.pop(0)


def _orientation_counts():
    return {name: {"gt": 0, "matched": 0, "unmatched": 0,
                   "unmatched_reasons": {"no_overlapping_prediction": 0, "iou_below_0.5": 0, "matching_conflict": 0},
                   "gt_crop_sdk": {}, "gt_crop_train": {},
                   "detected_crop_sdk": {}, "detected_crop_train": {}}
            for name in ("front", "back", "other")}


def summarize_orientations(rows):
    counts = _orientation_counts()
    for row in rows:
        for head in row["heads"]:
            label = head["toward"]
            if label not in counts:
                continue
            record = counts[label]
            record["gt"] += 1
            matched = head["matched_prediction"] is not None
            record["matched" if matched else "unmatched"] += 1
            if not matched and head.get("match_reason") in record["unmatched_reasons"]:
                record["unmatched_reasons"][head["match_reason"]] += 1
            for key in ("gt_crop", "detected_crop"):
                for variant in ("sdk", "train"):
                    predicted = head[key].get(variant)
                    if predicted is not None:
                        matrix = record[key + "_" + variant]
                        matrix[predicted] = matrix.get(predicted, 0) + 1
    for label, record in counts.items():
        gt, matched = record["gt"], record["matched"]
        record["detection_match_rate"] = matched / gt if gt else None
        for variant in ("sdk", "train"):
            correct = record["detected_crop_" + variant].get(label, 0)
            record["gt_crop_accuracy_" + variant] = record["gt_crop_" + variant].get(label, 0) / gt if gt else None
            record["matched_crop_accuracy_" + variant] = correct / matched if matched else None
            record["end_to_end_accuracy_" + variant] = correct / gt if gt else None
            record["matched_wrong_orientation_" + variant] = matched - correct
    return counts


def summarize_crop_pairs(rows):
    """Compare GT and detected crops for exactly the same matched targets."""
    result = {}
    for label in ("front", "back", "other"):
        result[label] = {}
        for variant in ("sdk", "train"):
            result[label][variant] = {"matched": 0, "both_correct": 0, "gt_correct_detected_wrong": 0,
                                      "gt_wrong_detected_correct": 0, "both_wrong": 0, "transitions": {}}
    for row in rows:
        for head in row["heads"]:
            label = head["toward"]
            if label not in result or head["matched_prediction"] is None:
                continue
            for variant in ("sdk", "train"):
                gt = head["gt_crop"].get(variant)
                detected = head["detected_crop"].get(variant)
                if gt is None or detected is None:
                    continue
                record = result[label][variant]
                record["matched"] += 1
                key = ("both_correct" if gt == label and detected == label else
                       "gt_correct_detected_wrong" if gt == label else
                       "gt_wrong_detected_correct" if detected == label else "both_wrong")
                record[key] += 1
                transition = record["transitions"].setdefault(gt, {})
                transition[detected] = transition.get(detected, 0) + 1
    return result


def orientation_scores(logits):
    """Raw three-class margins and softmax scores; not calibrated probabilities."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != 3 or not np.isfinite(values).all():
        raise ValueError("朝向分数需要三个有限 logits")
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    labels = ("front", "back", "other")
    return {"softmax": dict(zip(labels, probabilities.tolist())),
            "back_minus_other_logit": float(values[1] - values[2])}


def back_error_examples(rows, limit=50):
    cases = []
    for row in rows:
        for number, head in enumerate(row["heads"]):
            index = head["matched_prediction"]
            if head["toward"] != "back" or index is None or head["detected_crop"].get("sdk") == "back":
                continue
            if len(cases) >= limit:
                return cases
            cases.append({"image": row["image"], "head_index": number,
                          "gt_bbox": head["bbox"], "predicted_bbox": row["predictions"][index],
                          "matched_iou": box_iou(head["bbox"], row["predictions"][index]),
                          "gt_crop_prediction": head["gt_crop"].get("sdk"),
                          "detected_crop_prediction": head["detected_crop"].get("sdk"),
                          "gt_crop_scores": head["gt_crop"].get("scores", {}).get("sdk"),
                          "detected_crop_scores": head["detected_crop"].get("scores", {}).get("sdk")})
    return cases


def orientation_predictor(analyzer, include_scores=False):
    """Compare SDK bicubic vs the training OpenCV linear resize on one crop."""
    import cv2
    import torch
    sdk_preprocess = analyzer._preprocess_facexformer

    def training_preprocess(crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div(255)
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return ((tensor - mean) / std).unsqueeze(0)

    def classify(crop):
        if not include_scores:
            return analyzer._run_facexformer(crop)["orientation"], None
        captured = []
        hook = analyzer._facexformer.toward_classifier.register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach()))
        try:
            label = analyzer._run_facexformer(crop)["orientation"]
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("朝向分数采集次数不符，不能关联到本次裁剪预测")
        logits = captured[0].cpu().numpy()
        scores = orientation_scores(logits)
        if label != ("front", "back", "other")[int(np.argmax(logits))]:
            raise RuntimeError("采集到的朝向分数与 SDK 解码不一致")
        return label, scores

    def predict(crop):
        try:
            analyzer._preprocess_facexformer = sdk_preprocess
            sdk, sdk_scores = classify(crop)
            analyzer._preprocess_facexformer = training_preprocess
            train, train_scores = classify(crop)
            result = {"sdk": sdk, "train": train}
            if include_scores:
                result["scores"] = {"sdk": sdk_scores, "train": train_scores}
            return result
        finally:
            analyzer._preprocess_facexformer = sdk_preprocess
    return predict


def run_orientation_check(data_root, output_root, detector, predict, max_images=100, seed=42,
                          validation_paths=None, split_info=None, thresholds=(0.5,)):
    """Separate detection matching from crop classification; never update weights."""
    import cv2
    xml_paths = sorted(data_root.rglob("*.xml"))
    if not xml_paths:
        raise RuntimeError(f"没有找到 XML 标注: {data_root}")
    index = defaultdict(list)
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name.lower()].append(path)
    random.Random(seed).shuffle(xml_paths)
    rows, skipped, seen = [], 0, set()
    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
            heads = parse_heads(root)
            for head in heads:
                head["toward"] = head["toward"].strip().lower()
            # Attribute-only crops without toward cannot diagnose orientation.
            if not heads or (validation_paths is None and not any(
                    head["toward"] in {"front", "back", "other"} for head in heads)):
                skipped += 1
                continue
            image_path = find_image(xml_path, root, data_root, index)
        except (ET.ParseError, OSError, ValueError):
            skipped += 1
            continue
        if image_path is None or str(image_path) in seen:
            skipped += 1
            continue
        if validation_paths is not None and str(image_path.resolve()) not in validation_paths:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]

        def clipped(box):
            x1, y1, x2, y2 = (int(value) for value in box)
            return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

        for head in heads:
            head["bbox"] = clipped(head["bbox"])
        heads = [head for head in heads if head["bbox"][2] > head["bbox"][0]
                 and head["bbox"][3] > head["bbox"][1]]
        if not heads or (validation_paths is None and not any(
                head["toward"] in {"front", "back", "other"} for head in heads)):
            skipped += 1
            continue
        boxes, scores = detector.detect(np.ascontiguousarray(image[:, :, ::-1]))
        candidate_predictions = [] if boxes is None else [clipped(box) for box in boxes]
        candidate_scores = [] if scores is None else [float(value) for value in scores]
        if len(candidate_scores) != len(candidate_predictions):
            raise ValueError("检测框和置信度数量不一致")
        selected = [i for i, score in enumerate(candidate_scores) if score >= 0.5]
        predictions = [candidate_predictions[i] for i in selected]
        probabilities = [candidate_scores[i] for i in selected]
        matches, details, _ = diagnose_matching(heads, predictions, probabilities)
        for number, head in enumerate(heads):
            head.update(details[number])
            head["matched_prediction"] = matches.get(number)
            head["gt_crop"], head["detected_crop"] = {}, {}
            if head["toward"] not in {"front", "back", "other"}:
                continue
            x1, y1, x2, y2 = head["bbox"]
            head["gt_crop"] = predict(image[y1:y2, x1:x2].copy())
            if number in matches:
                x1, y1, x2, y2 = predictions[matches[number]]
                head["detected_crop"] = predict(image[y1:y2, x1:x2].copy())
        rows.append({"image": str(image_path), "xml": str(xml_path), "heads": heads,
                     "predictions": predictions, "scores": probabilities,
                     "candidate_predictions": candidate_predictions, "candidate_scores": candidate_scores})
        seen.add(str(image_path))
        print(f"朝向检查 {len(rows)}/{max_images}: GT={len(heads)} MATCH={len(matches)}", flush=True)
        if len(rows) >= max_images:
            break
    if not rows:
        raise RuntimeError("没有可用的朝向标注和对应原图，请挂载数据 A；纯属性裁剪图不适用。")
    report = {"scope": "diagnostic_not_platform_score", "seed": seed,
              "images": len(rows), "skipped_xml": skipped,
              "matching": "confidence_first_one_to_one_rectangle_iou_0.5",
              "variants": {"sdk": "SDK PIL bicubic", "train": "training OpenCV linear"},
              "split": split_info or {"mode": "mixed_data_sample"},
              "counts": summarize_orientations(rows), "per_image": rows,
              "paired_crops": summarize_crop_pairs(rows),
              "back_error_examples": back_error_examples(rows), "back_error_example_limit": 50,
              "detection_thresholds": summarize_detection(rows, thresholds),
              "limitations": ["not_hidden_test_labels", "not_official_matching", "depends_on_annotation_completeness"]}
    if validation_paths is None:
        report["limitations"].append("possible_training_images")
    else:
        report["limitations"].extend(["requires_same_data_as_head_training", "not_guaranteed_attribute_model_holdout"])
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "orientation_check.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2), flush=True)
    print("同目标裁剪对照: " + json.dumps(report["paired_crops"], ensure_ascii=False, indent=2), flush=True)
    print("背面分类错误明细（最多50个，分数不是校准概率）:", flush=True)
    for example in report["back_error_examples"]:
        print(json.dumps(example, ensure_ascii=False), flush=True)
    print("检测框阈值对照: " + json.dumps(report["detection_thresholds"], ensure_ascii=False, indent=2), flush=True)
    print(f"报告: {path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--data-dir", type=Path, default=Path("/home/data"))
    sources.add_argument("--image", type=Path, help="单张原图检测对照，不需要 XML 标注")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/project/train/result-graphs/detection_check"))
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--detector", choices=("auto", "headshoulder", "mtcnn"), default="auto")
    parser.add_argument("--checkpoint", type=Path, help="头肩检测权重；默认与 SDK 挂载路径一致")
    parser.add_argument("--check-orientation", action="store_true",
                        help="原图上检查框匹配/朝向分类，并比较 SDK 和训练缩放；只输出一个 JSON")
    parser.add_argument("--facexformer-checkpoint", type=Path, help="朝向检查所用 FaceXFormer 权重")
    parser.add_argument("--orientation-scores", action="store_true", help="只在诊断中记录朝向 logits 差值和 softmax 分数")
    parser.add_argument("--validation-only", action="store_true", help="复用全量头肩训练的数据划分，只检查其留出原图")
    parser.add_argument("--expected-detection-images", type=int, help="与原训练可读原图数量核对，不符则停止")
    parser.add_argument("--detection-thresholds", type=float, nargs="+", default=[0.5],
                        help="检测阈值对照（须含 0.5）；正式 SDK 阈值不变")
    args = parser.parse_args()
    if args.image is not None and not args.image.is_file():
        parser.error(f"图片不存在: {args.image}。请先上传图片并检查路径。")
    if args.image is None and not args.data_dir.is_dir():
        parser.error(f"数据目录不存在: {args.data_dir}。请用 --data-dir 指定可访问目录。")
    if args.max_images < 1:
        parser.error("--max-images 必须大于 0")
    if (args.orientation_scores or args.validation_only or args.expected_detection_images is not None or args.detection_thresholds != [0.5]) and not args.check_orientation:
        parser.error("留出集和阈值对照参数需要 --check-orientation")
    if args.expected_detection_images is not None and (not args.validation_only or args.expected_detection_images < 2):
        parser.error("--expected-detection-images 需配合 --validation-only 且至少为 2")
    if 0.5 not in args.detection_thresholds or not all(0 < value <= 1 for value in args.detection_thresholds):
        parser.error("检测阈值须在 (0,1] 内且包含 0.5")
    thresholds = sorted(set(args.detection_thresholds))
    if args.check_orientation:
        if args.image is not None or args.detector == "mtcnn":
            parser.error("朝向检查需要数据 A 原图/XML 和头肩检测器")
        if args.checkpoint is None or args.facexformer_checkpoint is None:
            parser.error("朝向检查必须显式指定 --checkpoint 和 --facexformer-checkpoint，避免误用旧权重")
        for path in (args.checkpoint, args.facexformer_checkpoint):
            if not path.is_file():
                parser.error(f"检查权重不存在: {path}；需要任务 182201 的实际可访问文件，不能用旧权重替代")
    # Delay heavyweight imports so --help and diagnostic helpers work offline.
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE={device} TORCH={torch.__version__}", flush=True)
    checkpoint = args.checkpoint
    validation_paths, split_info = None, None
    if args.validation_only:
        validation_paths, split_info = validation_image_paths(args.data_dir.resolve(), args.expected_detection_images)
    if args.detector != "mtcnn" and checkpoint is None:
        from ji import _find_weight
        try:
            checkpoint = _find_weight("headshoulder/model.pt")
        except FileNotFoundError:
            if args.detector == "headshoulder":
                raise
    if args.detector == "mtcnn" or checkpoint is None:
        from facenet_pytorch import MTCNN
        detector = MTCNN(keep_all=True, device=device)
        detector_name = "MTCNN"
    else:
        from head_detector import HeadShoulderDetector
        detector = HeadShoulderDetector(checkpoint, device, score_threshold=min(thresholds))
        detector_name = "HeadShoulder"
    with torch.no_grad():
        if args.check_orientation:
            from face_attr.analyzer import AttributeAnalyzer
            analyzer = AttributeAnalyzer(device=device, facexformer_weights=args.facexformer_checkpoint)
            analyzer._ensure_facexformer()
            if not hasattr(analyzer._facexformer, "toward_classifier"):
                raise ValueError("这次检查需要任务 182201 的 toward 三分类权重，当前文件仍是旧角度模型")
            report = run_orientation_check(args.data_dir.resolve(), args.output_dir.resolve(), detector,
                                           orientation_predictor(analyzer, args.orientation_scores), args.max_images, args.seed,
                                           validation_paths, split_info, thresholds)
            report["weights"] = {"head": str(args.checkpoint.resolve()),
                                 "facexformer": str(args.facexformer_checkpoint.resolve())}
            (args.output_dir / "orientation_check.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.image is not None:
            run_single_image(args.image.resolve(), args.output_dir.resolve(), detector, device, detector_name)
        else:
            run_check(args.data_dir.resolve(), args.output_dir.resolve(), detector,
                      args.max_images, args.seed, detector_name)


if __name__ == "__main__":
    main()
