"""本地冒烟测试：验证 finetune_facexformer.py 的 headpose 目标构造在真实代码路径下可用。

本机没有 cv2 与平台数据集，所以：
  - 用 sys.modules 打桩替换 cv2（只实现 dataset.py 用到的 imread/cvtColor/resize）
  - 造一个覆盖 front / back / other 三种朝向的迷你 VOC 数据集
然后真实调用 finetune_facexformer.run_epoch，确认不再出现 dtype 错误。
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
CHECKPOINT = ROOT / "project" / "ev_sdk" / "model" / "facexformer" / "model_original.pt"

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

# ------------------------------------------------- 造迷你 VOC 数据集
XML_TEMPLATE = """<annotation>
  <filename>{filename}</filename>
  <object>
    <name>head</name>
    <bndbox><xmin>2</xmin><ymin>2</ymin><xmax>30</xmax><ymax>30</ymax></bndbox>
    <attributes>
      <attribute><name>toward</name><value>{toward}</value></attribute>
      <attribute><name>gender</name><value>{gender}</value></attribute>
      <attribute><name>race</name><value>{race}</value></attribute>
      <attribute><name>glasses</name><value>0</value></attribute>
      <attribute><name>emotion</name><value>2</value></attribute>
      <attribute><name>mask</name><value>0</value></attribute>
      <attribute><name>hat</name><value>0</value></attribute>
      <attribute><name>whiskers</name><value>0</value></attribute>
      <attribute><name>age</name><value>33</value></attribute>
    </attributes>
  </object>
</annotation>
"""

# 注意：gender 只有 0/1、race 只有 0-3，越界会被 dataset 直接跳过（这里踩过一次）。
# 样本按 XML 文件名排序：back -> front -> front -> other
CASES = [
    ("img_back_1.jpg", "back", 1, 1),
    ("img_front_1.jpg", "front", 0, 0),
    ("img_front_2.jpg", "front", 0, 3),
    ("img_other_1.jpg", "other", 0, 2),
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


def main() -> int:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    sys.path.insert(0, str(TRAIN_SRC))

    import finetune_facexformer as F
    from dataset import FaceAttributeDataset
    from network import FaceXFormer
    from torch.utils.data import DataLoader

    base = Path(tempfile.mkdtemp(prefix="smoke_"))
    try:
        data_dir = build_dataset(base)
        dataset = FaceAttributeDataset(str(data_dir), input_size=224)
        print(f"[1] 数据集样本数 = {len(dataset)}")
        assert len(dataset) == len(CASES), "有样本被跳过，检查测试数据的属性取值范围"

        raw_image, raw_label = dataset[0]  # back 样本
        print(f"[2] 单样本 image dtype={raw_image.dtype} toward={int(raw_label['toward'])} (back=1)")
        assert raw_image.dtype == torch.float32
        assert int(raw_label["toward"]) == 1

        device = torch.device("cpu")
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        model = FaceXFormer().to(device)

        if CHECKPOINT.exists():
            state = torch.load(str(CHECKPOINT), map_location=device)
            model.load_state_dict(state["state_dict_backbone"])
            print(f"[3] 已加载真实权重 {CHECKPOINT.name} 的 state_dict_backbone")
        else:
            print(f"[3] 警告: 未找到 {CHECKPOINT}，使用随机初始化权重")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        print("[4] 开始 run_epoch（CPU，2 个 batch）...")
        loss = F.run_epoch(model, loader, optimizer, device)
        print(f"[5] run_epoch 完成，loss={loss:.5f}")

        # 直接复检改动的那一行：三种朝向的目标值
        toward = torch.tensor([0, 1, 2], dtype=torch.long)
        headpose = torch.zeros(3, 3, device=device, dtype=torch.float32)
        headpose[:, 1] = headpose[:, 1].masked_fill(toward == 1, 3.14)
        headpose[:, 1] = headpose[:, 1].masked_fill(toward == 2, 0.8)
        headpose[:, 0] = headpose[:, 0].masked_fill(toward == 2, 0.8)
        print(f"[6] headpose dtype={headpose.dtype}  front/back/other ->")
        for name, row in zip(("front", "back", "other"), headpose.tolist()):
            print(f"      {name:5s} pitch={row[0]:.4f} yaw={row[1]:.4f}")
        assert headpose.dtype == torch.float32
        assert abs(headpose[1, 1].item() - 3.14) < 1e-5
        assert abs(headpose[2, 0].item() - 0.8) < 1e-5 and abs(headpose[2, 1].item() - 0.8) < 1e-5

        print("\nSMOKE PASS")
        return 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
