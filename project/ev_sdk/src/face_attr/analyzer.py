"""人脸属性分析模块。

提供年龄、性别、表情、朝向等属性的分析接口。
接入两个预训练模型：
- FaceXFormer: 朝向、年龄、性别、种族、可见性
- SwinFace: 年龄、表情、眼镜、帽子、胡须等 CelebA 属性
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

# 项目根目录（src/face_attr/ 的上级的上级，即 ev_sdk/）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认权重路径。SDK 权重目录名是单数 model/，与 train/models/ 不同。
DEFAULT_FACEXFORMER_WEIGHTS = _PROJECT_ROOT / "model" / "facexformer" / "model.pt"
DEFAULT_SWINFACE_WEIGHTS = (
    _PROJECT_ROOT / "model" / "swinface" / "checkpoint_step_79999_gpu_0.pt"
)

# FaceXFormer 任务 token
_TASK_HEADPOSE = torch.tensor([2])
_TASK_ATTRIBUTES = torch.tensor([3])
_TASK_AGE_GENDER_RACE = torch.tensor([4])

# SwinFace 表情类别（与 RAF-DB 一致）
SWINFACE_EXPRESSION_CLASSES = [
    "surprise",
    "fear",
    "disgust",
    "happy",
    "sad",
    "angry",
    "neutral",
]


def _positive_class_probability(logits: torch.Tensor) -> float:
    """把 SwinFace 的属性头输出转成「正类」概率。

    SwinFace 的 CelebA 属性头（Smiling / Eyeglasses / Wearing Hat / Mustache /
    No Beard）输出维度是 2，属于 2 分类而不是单 logit。这里取 softmax 的正类
    概率；若某个头真的是单 logit，则退回到 sigmoid。
    """
    flat = logits.reshape(-1)
    if logits.shape[-1] == 1:
        return float(torch.sigmoid(flat[0]).item())
    return float(torch.softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)[0, 1].item())


@dataclass
class FaceAttributes:
    """单张人脸的属性分析结果。"""

    age: float | None = None
    gender: str | None = None
    expression: str | None = None
    orientation: str | None = None
    extra: dict[str, float] = field(default_factory=dict)


class AttributeAnalyzer:
    """人脸属性分析器。

    加载 FaceXFormer 与 SwinFace 两个模型，对单张人脸裁剪图进行属性分析。
    """

    def __init__(
        self,
        device: str = "cpu",
        facexformer_weights: str | os.PathLike | None = None,
        swinface_weights: str | os.PathLike | None = None,
    ) -> None:
        self.device = torch.device(device)
        self._facexformer = None
        self._swinface = None
        self._facexformer_gender_classes = ("male", "female")
        self._facexformer_weights = (
            Path(facexformer_weights)
            if facexformer_weights
            else DEFAULT_FACEXFORMER_WEIGHTS
        )
        self._swinface_weights = (
            Path(swinface_weights) if swinface_weights else DEFAULT_SWINFACE_WEIGHTS
        )

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_facexformer(self) -> torch.nn.Module:
        """加载 FaceXFormer 模型。"""
        # 将 FaceXFormer 源码目录加入 sys.path，使 `from network import ...` 可用
        fx_root = Path(__file__).resolve().parent / "models" / "facexformer"
        if str(fx_root) not in sys.path:
            sys.path.insert(0, str(fx_root))

        from network import FaceXFormer  # 延迟导入，避免启动开销

        model = FaceXFormer().to(self.device)
        checkpoint = torch.load(self._facexformer_weights, map_location=self.device)
        # 新微调权重会记录类别顺序；旧原始权重没有元数据，继续沿用 0=男/1=女。
        # 旧版微调产物没有元数据，可显式设置 FACEXFORMER_GENDER_ORDER=female,male。
        gender_classes = checkpoint.get("gender_classes", ("male", "female"))
        override = os.environ.get("FACEXFORMER_GENDER_ORDER")
        if override is not None:
            gender_classes = [name.strip() for name in override.split(",")]
        if not isinstance(gender_classes, (list, tuple)) or tuple(gender_classes) not in (
            ("male", "female"), ("female", "male")
        ):
            raise ValueError("FaceXFormer gender_classes 必须为 male,female 或 female,male")
        self._facexformer_gender_classes = tuple(gender_classes)
        model.load_state_dict(checkpoint["state_dict_backbone"])
        model.eval()
        return model

    def _load_swinface(self) -> torch.nn.Module:
        """加载 SwinFace 模型。"""
        # 将 SwinFace 源码目录加入 sys.path，使 `from model import ...` 可用
        sw_root = Path(__file__).resolve().parent / "models" / "swinface"
        if str(sw_root) not in sys.path:
            sys.path.insert(0, str(sw_root))

        from model import build_model  # 延迟导入，避免启动开销

        cfg = SwinFaceCfg()
        model = build_model(cfg).to(self.device)
        checkpoint = torch.load(self._swinface_weights, map_location=self.device)
        model.backbone.load_state_dict(checkpoint["state_dict_backbone"])
        model.fam.load_state_dict(checkpoint["state_dict_fam"])
        model.tss.load_state_dict(checkpoint["state_dict_tss"])
        model.om.load_state_dict(checkpoint["state_dict_om"])
        model.eval()
        return model

    def _ensure_models(self) -> None:
        """按需加载两个模型（懒加载）。"""
        self._ensure_facexformer()
        self._ensure_swinface()

    def _ensure_facexformer(self) -> None:
        if self._facexformer is None:
            self._facexformer = self._load_facexformer()

    def _ensure_swinface(self) -> None:
        if self._swinface is None:
            self._swinface = self._load_swinface()

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_facexformer(face_crop: np.ndarray) -> torch.Tensor:
        """FaceXFormer 预处理：BGR -> RGB，resize 到 224x224，ImageNet 归一化。"""
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    size=(224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        return transform(face_crop).unsqueeze(0)

    @staticmethod
    def _preprocess_swinface(face_crop: np.ndarray) -> torch.Tensor:
        """SwinFace 预处理：BGR -> RGB，resize 到 112x112，(x/255-0.5)/0.5。"""
        img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        return img

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_facexformer(self, face_crop: np.ndarray) -> dict:
        """运行 FaceXFormer，返回朝向、年龄、性别、种族。"""
        image = self._preprocess_facexformer(face_crop).to(self.device)

        result: dict = {}

        # FaceXFormer.forward 内部会无条件算出全部 8 个头，task 只对返回值做切片，
        # 所以同一张脸跑一次前向就能同时拿到朝向和年龄/性别/种族。
        # 原来按 task 调两次，等于把整个解码器算两遍（推理耗时也接近 2 倍，
        # 而耗时直接计入榜单的 20% 性能分）。labels 参数在 forward 里从未被使用。
        _, headpose_output, _, _, age_output, gender_output, race_output, _ = self._facexformer(
            image, None, None
        )

        # 朝向
        result["orientation"] = self._headpose_to_orientation(
            headpose_output[0].cpu().numpy()
        )

        # 年龄 / 性别 / 种族
        if age_output.numel() == 1:
            result["age"] = float(age_output.item())
        else:
            result["age"] = float(age_output.argmax(1).item())
        gender_index = int(gender_output.argmax(1).item())
        result["gender"] = self._facexformer_gender_classes[gender_index]
        result["race"] = int(race_output.argmax(1).item())

        return result

    @torch.no_grad()
    def _run_swinface(self, face_crop: np.ndarray) -> dict:
        """运行 SwinFace，返回年龄、表情及 CelebA 属性。"""
        img = self._preprocess_swinface(face_crop).to(self.device)
        output = self._swinface(img)

        result: dict = {}
        age_output = output["Age"]
        result["age"] = float(age_output.reshape(-1)[0].item())
        result["expression"] = SWINFACE_EXPRESSION_CLASSES[
            int(output["Expression"].argmax(1).item())
        ]
        # CelebA 属性头在 SwinFace 中是 2 分类输出（shape (B, 2)），
        # 必须取正类概率，不能直接对 (B, 2) 张量做 sigmoid().item()。
        result["smiling"] = _positive_class_probability(output["Smiling"])
        result["eyeglasses"] = _positive_class_probability(output["Eyeglasses"])
        result["wearing_hat"] = _positive_class_probability(output["Wearing Hat"])
        result["mustache"] = _positive_class_probability(output["Mustache"])
        result["no_beard"] = _positive_class_probability(output["No Beard"])
        return result

    @staticmethod
    def _headpose_to_orientation(euler: np.ndarray) -> str:
        """根据欧拉角（弧度）判断人脸朝向。

        榜单只认三类：``front`` 正面（两眼可见）、``back`` 背面（两眼都不可见）、
        ``other`` 其它（侧面等）。这里的阈值与微调脚本写入的 headpose 目标一致：
        正面约 0 弧度、其它约 0.8 弧度、背面约 3.14 弧度。
        """
        pitch, yaw, roll = euler
        yaw_deg = yaw * 180 / np.pi
        pitch_deg = pitch * 180 / np.pi
        del roll  # 平面内旋转不影响是否能看到双眼，不参与朝向判断

        # 偏航角接近 180 度时人脸背对镜头，双眼均不可见。
        if abs(yaw_deg) >= 90:
            return "back"
        if abs(yaw_deg) < 30 and abs(pitch_deg) < 30:
            return "front"
        return "other"

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def analyze(self, face_crop: np.ndarray) -> FaceAttributes:
        """分析单张人脸裁剪图。

        Args:
            face_crop: 人脸区域图像（BGR）。

        Returns:
            属性分析结果。
        """
        self._ensure_models()

        fx = self._run_facexformer(face_crop)
        sw = self._run_swinface(face_crop)

        # 年龄优先取 SwinFace（更精确），否则取 FaceXFormer
        age = sw.get("age", fx.get("age"))
        gender = fx.get("gender")
        expression = sw.get("expression")
        orientation = fx.get("orientation")

        extra: dict[str, float] = {}
        for key in (
            "smiling",
            "eyeglasses",
            "wearing_hat",
            "mustache",
            "no_beard",
        ):
            if key in sw:
                extra[key] = sw[key]
        if "race" in fx:
            extra["race"] = float(fx["race"])

        return FaceAttributes(
            age=age,
            gender=gender,
            expression=expression,
            orientation=orientation,
            extra=extra,
        )


class SwinFaceCfg:
    """SwinFace 模型配置（与官方 inference.py 一致）。"""

    network = "swin_t"
    fam_kernel_size = 3
    fam_in_chans = 2112
    fam_conv_shared = False
    fam_conv_mode = "split"
    fam_channel_attention = "CBAM"
    fam_spatial_attention = None
    fam_pooling = "max"
    fam_la_num_list = [2 for j in range(11)]
    fam_feature = "all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512
