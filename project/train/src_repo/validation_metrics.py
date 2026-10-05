"""Local crop diagnostics; these metrics do not reproduce the platform scorer."""

import json
import math
import random
from pathlib import Path


def split_image_indices(samples, val_fraction=0.2, seed=42):
    """Keep every target from the same original image in the same split."""
    groups = {}
    for index, (path, _) in enumerate(samples):
        groups.setdefault(str(Path(path).resolve()), []).append(index)
    paths = sorted(groups)
    if len(paths) < 2:
        raise ValueError("训练/验证至少需要两张不同的原图")
    random.Random(seed).shuffle(paths)
    size = max(1, min(len(paths) - 1, int(len(paths) * val_fraction)))
    val = [index for path in paths[:size] for index in groups[path]]
    train = [index for path in paths[size:] for index in groups[path]]
    return train, val


def require_training_classes(samples, indices, tasks):
    support = {key: [0] * count for key, count in tasks.items()}
    for index in indices:
        label = samples[index][1]
        for key, counts in support.items():
            value = int(label.get(key, -1))
            if 0 <= value < len(counts):
                counts[value] += 1
    missing = {key: [i for i, count in enumerate(counts) if not count]
               for key, counts in support.items() if not all(counts)}
    if missing:
        raise ValueError(f"训练部分缺少任务类别 {missing}，样本分布={support}；请挂载完整数据 A/B 或扩大抽样，不可用未监督的新分类头代替成品模型")
    return support


class ClassificationCounts:
    def __init__(self, classes):
        self.classes = list(classes)
        # The last column includes unknown or out-of-range predictions.
        self.matrix = [[0] * (len(self.classes) + 1) for _ in self.classes]
        self.ignored = 0

    def update(self, targets, predictions):
        for target, predicted in zip(targets, predictions):
            target, predicted = int(target), int(predicted)
            if not 0 <= target < len(self.classes):
                self.ignored += 1
                continue
            column = predicted if 0 <= predicted < len(self.classes) else len(self.classes)
            self.matrix[target][column] += 1

    def report(self):
        support = [sum(row) for row in self.matrix]
        correct = [self.matrix[i][i] for i in range(len(self.classes))]
        recalls = [hit / count if count else None for hit, count in zip(correct, support)]
        present = [value for value in recalls if value is not None]
        total = sum(support)
        return {
            "accuracy": sum(correct) / total if total else None,
            "labeled_count": total,
            "ignored_count": self.ignored,
            "classes": self.classes,
            "support": support,
            "correct": correct,
            "per_class_recall": recalls,
            "balanced_accuracy_present_classes": sum(present) / len(present) if present else None,
            "confusion_matrix": self.matrix,
            "prediction_columns": self.classes + ["unknown"],
        }


def write_validation_report(output_path, report):
    """Write strict JSON: a task without labels is null, never NaN."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path = Path(str(output_path) + ".validation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path
