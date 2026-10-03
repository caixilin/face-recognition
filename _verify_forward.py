"""本地离线校验：证明 FaceXFormer.forward(tasks=None) 与按 task 切片的结果完全等价。

用于验证 finetune_facexformer.py / analyzer.py 从"按 task 调两次"改成
"一次 tasks=None"后，输出张量的形状与数值都不变。

不需要 cv2、不联网，CPU 即可（约 1 分钟，主要是加载 1.1 GB 权重）。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
FACEXFORMER_SRC = ROOT / "project" / "ev_sdk" / "src" / "face_attr" / "models" / "facexformer"
sys.path.insert(0, str(FACEXFORMER_SRC))

from network import FaceXFormer  # noqa: E402

CHECKPOINT = ROOT / "project" / "ev_sdk" / "model" / "facexformer" / "model_original.pt"


def main() -> None:
    model = FaceXFormer().eval()
    state = torch.load(str(CHECKPOINT), map_location="cpu")
    model.load_state_dict(state["state_dict_backbone"])
    print("[1] 已加载真实权重")

    torch.manual_seed(0)
    images = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        raw = model(images, None, None)
        sliced_pose = model(images, None, torch.tensor([2]))
        sliced_agr = model(images, None, torch.tensor([4]))

    pairs = [
        ("headpose (task=2)", raw[1], sliced_pose[1]),
        ("age      (task=4)", raw[4], sliced_agr[4]),
        ("gender   (task=4)", raw[5], sliced_agr[5]),
        ("race     (task=4)", raw[6], sliced_agr[6]),
    ]

    ok = True
    print("[2] tasks=None  vs  按 task 切片")
    for name, a, b in pairs:
        same = bool(torch.equal(a, b))
        ok &= same
        print(f"      {name}   raw={tuple(a.shape)}  sliced={tuple(b.shape)}  完全相等={same}")

    unused_raw = raw[0].shape
    unused_sliced = sliced_pose[0].shape
    print(f"      landmark (未使用)  raw={tuple(unused_raw)}  sliced={tuple(unused_sliced)}  → 形状不同但两边都不读")

    print("\nVERIFY PASS" if ok else "\nVERIFY FAIL")


if __name__ == "__main__":
    main()
