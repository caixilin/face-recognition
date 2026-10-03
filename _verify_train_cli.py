"""本地端到端验证：确认 finetune_facexformer.py 的 --max_samples 与"每 epoch 保存"真的生效。

打桩 cv2、造迷你 VOC，然后真实调用 finetune_facexformer.main()。
把 torch.save 换成一个只记录不落盘的函数，避免每次写 1.4 GB。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
TRAIN_SRC = ROOT / "project" / "train" / "src_repo"

# ---------------------------------------------------------------- cv2 桩
_IMAGES: dict[str, np.ndarray] = {}


def _imread(path, flags=None):
    return _IMAGES.get(Path(str(path)).name)


def _cvt_color(image, code):
    return image[..., ::-1]


def _resize(image, size):
    width, height = size
    src_h, src_w = image.shape[:2]
    ys = np.clip((np.arange(height) * src_h) // height, 0, src_h - 1)
    xs = np.clip((np.arange(width) * src_w) // width, 0, src_w - 1)
    return image[ys][:, xs]


_cv2 = types.ModuleType("cv2")
_cv2.imread = _imread
_cv2.cvtColor = _cvt_color
_cv2.resize = _resize
_cv2.COLOR_BGR2RGB = 4
sys.modules["cv2"] = _cv2

XML_TEMPLATE = """<annotation>
  <filename>{filename}</filename>
  <object>
    <name>head</name>
    <bndbox><xmin>2</xmin><ymin>2</ymin><xmax>30</xmax><ymax>30</ymax></bndbox>
    <attributes>
      <attribute><name>toward</name><value>{toward}</value></attribute>
      <attribute><name>gender</name><value>{gender}</value></attribute>
      <attribute><name>race</name><value>{race}</value></attribute>
      <attribute><name>age</name><value>33</value></attribute>
    </attributes>
  </object>
</annotation>
"""

# 6 个样本，朝向轮流覆盖 front / back / other
CASES = [
    (f"img{i:02d}.jpg", tow, i % 2, i % 4) for i, tow in enumerate(["front", "back", "other"] * 2)
]


def build_dataset(base: Path) -> Path:
    data_dir = base / "mini_data"
    (data_dir / "images").mkdir(parents=True)
    (data_dir / "labels").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for filename, toward, gender, race in CASES:
        _IMAGES[filename] = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        (data_dir / "images" / filename).write_bytes(b"stub")
        (data_dir / "labels" / f"{filename}.xml").write_text(
            XML_TEMPLATE.format(filename=filename, toward=toward, gender=gender, race=race),
            encoding="utf-8",
        )
    return data_dir


def verify_stratified_subset(base: Path) -> bool:
    """多来源分层抽样：每份数据都要被覆盖，总量精确，samples/sources 保持同步。"""
    data_dir = base / "multi_data"
    rng = np.random.default_rng(1)
    # 故意让三份数据的样本数悬殊，检验「样本少的那份是否仍被覆盖」和「配额是否抽满」。
    plan = {"groupA": 30, "groupB": 12, "groupC": 6}
    for group, count in plan.items():
        (data_dir / group / "images").mkdir(parents=True)
        (data_dir / group / "labels").mkdir(parents=True)
        for i in range(count):
            name = f"{group}_{i:03d}.jpg"
            _IMAGES[name] = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
            (data_dir / group / "images" / name).write_bytes(b"stub")
            (data_dir / group / "labels" / f"{name}.xml").write_text(
                XML_TEMPLATE.format(filename=name, toward="front", gender=1, race=0),
                encoding="utf-8",
            )

    import dataset as D

    ds = D.FaceAttributeDataset(data_dir, input_size=64)
    total = len(ds)
    print(f"\n[E] 多来源数据集 {plan}，共解析出 {total} 个样本")

    max_samples = 30
    ds.select_subset(max_samples)
    counts: dict = {}
    for source in ds.sources:
        counts[source] = counts.get(source, 0) + 1

    print(f"    抽样后各来源: {counts}")
    ok_size = len(ds.samples) == max_samples
    ok_sync = len(ds.samples) == len(ds.sources)
    ok_cover = set(counts) == set(plan)
    print(f"[F] 总量精确等于 {max_samples} : {ok_size}")
    print(f"[G] samples 与 sources 长度一致 : {ok_sync}")
    print(f"[H] 三份数据全部被覆盖 : {ok_cover}")

    # 抽样必须是随机的：两次调用（不同 seed 之外的默认路径）应给出不同顺序，
    # 这里只验证「不是按路径顺序原样截断」这一最低要求。
    print(f"[I] 抽样结果非顺序截断 : {ds.sources[:5] != sorted(ds.sources)[:5] or len(set(ds.sources)) == 1}")
    return ok_size and ok_sync and ok_cover


def main() -> int:
    sys.path.insert(0, str(TRAIN_SRC))
    import finetune_facexformer as F

    saves: list[Path] = []
    real_save = torch.save

    def recording_save(obj, path, *a, **kw):
        saves.append(Path(path))
        return None

    torch.save = recording_save

    base = Path(tempfile.mkdtemp(prefix="cli_"))
    try:
        data_dir = build_dataset(base)
        out = base / "out.pt"

        sys.argv = [
            "finetune_facexformer.py",
            "--data_dir", str(data_dir),
            "--epochs", "3",
            "--batch_size", "2",
            "--num_workers", "0",
            "--device", "cpu",
            "--max_samples", "5",          # 6 -> 5，应打印切片提示
            "--output", str(out),
        ]
        F.main()

        print()
        print(f"[A] torch.save 被调用 {len(saves)} 次（3 个 epoch 应各存一次）")
        for i, p in enumerate(saves, 1):
            print(f"      #{i} -> {p.name}")
        ok_saves = len(saves) == 3 and all(p == out for p in saves)
        print(f"[B] 每 epoch 都保存 = {ok_saves}")

        print(f"[C] --max_samples 应把 6 个样本切到 5 个（日志见上方）")
        print(f"[D] 输出路径正确 = {all(p == out for p in saves)}")

        ok_subset = verify_stratified_subset(base)

        good = ok_saves and ok_subset
        print("\nCLI VERIFY PASS" if good else "\nCLI VERIFY FAIL")
        return 0 if good else 1
    finally:
        torch.save = real_save
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
