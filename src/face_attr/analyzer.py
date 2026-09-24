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

# 项目根目录（src/face_attr/ 的上级的上级）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认权重路径
DEFAULT_FACEXFORMER_WEIGHTS = _PROJECT_ROOT / "models" / "facexformer" / "model.pt"
DEFAULT_SWINFACE_WEIGHTS = (
    _PROJECT_ROOT / "models" / "swinface" / "checkpoint_step_79999_gpu_0.pt"
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
        labels = {
            "segmentation": torch.zeros([224, 224]),
            "lnm_seg": torch.zeros([5, 2]),
            "landmark": torch.zeros([68, 2]),
            "headpose": torch.zeros([3]),
            "attribute": torch.zeros([40]),
            "a_g_e": torch.zeros([3]),
            "visibility": torch.zeros([29]),
        }
        for k in labels:
            labels[k] = labels[k].unsqueeze(0).to(self.device)

        result: dict = {}

        # 朝向
        task = _TASK_HEADPOSE.to(self.device)
        data = {"image": image, "label": labels, "task": task}
        _, headpose_output, _, _, _, _, _, _ = self._facexformer(
            data["image"], data["label"], data["task"]
        )
        result["orientation"] = self._headpose_to_orientation(
            headpose_output[0].cpu().numpy()
        )

        # 年龄 / 性别 / 种族
        task = _TASK_AGE_GENDER_RACE.to(self.device)
        data = {"image": image, "label": labels, "task": task}
        _, _, _, _, age_output, gender_output, race_output, _ = self._facexformer(
            data["image"], data["label"], data["task"]
        )
        if age_output.numel() == 1:
            result["age"] = float(age_output.item())
        else:
            result["age"] = float(age_output.argmax(1).item())
        result["gender"] = "male" if gender_output.argmax(1).item() == 0 else "female"
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
        result["smiling"] = float(output["Smiling"].sigmoid().item())
        result["eyeglasses"] = float(output["Eyeglasses"].sigmoid().item())
        result["wearing_hat"] = float(output["Wearing Hat"].sigmoid().item())
        result["mustache"] = float(output["Mustache"].sigmoid().item())
        result["no_beard"] = float(output["No Beard"].sigmoid().item())
        return result

    @staticmethod
    def _headpose_to_orientation(euler: np.ndarray) -> str:
        """根据欧拉角（弧度）判断人脸朝向。"""
        pitch, yaw, roll = euler
        # 转换为角度
        yaw_deg = yaw * 180 / np.pi
        pitch_deg = pitch * 180 / np.pi
        roll_deg = roll * 180 / np.pi

        if abs(yaw_deg) < 30 and abs(pitch_deg) < 30 and abs(roll_deg) < 30:
            return "front"
        if yaw_deg > 30:
            return "left"
        if yaw_deg < -30:
            return "right"
        if pitch_deg > 30:
            return "up"
        if pitch_deg < -30:
            return "down"
        return "front"

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
