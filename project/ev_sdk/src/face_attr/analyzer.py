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


def _positive_class_tensor(logits: torch.Tensor) -> torch.Tensor:
    """把 SwinFace 的属性头输出转成「正类」概率。

    SwinFace 的 CelebA 属性头（Smiling / Eyeglasses / Wearing Hat / Mustache /
    No Beard）输出维度是 2，属于 2 分类而不是单 logit。这里取 softmax 的正类
    概率；若某个头真的是单 logit，则退回到 sigmoid。
    """
    flat = logits.reshape(-1)
    if logits.shape[-1] == 1:
        return torch.sigmoid(flat[0])
    return torch.softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)[0, 1]


def _positive_class_probability(logits: torch.Tensor) -> float:
    return float(_positive_class_tensor(logits).item())


def _positive_class_batch(logits: torch.Tensor) -> torch.Tensor:
    rows = logits.reshape(-1, logits.shape[-1])
    if rows.shape[1] == 1:
        return torch.sigmoid(rows[:, 0])
    return torch.softmax(rows, dim=-1)[:, 1]


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
        if "toward_classes" in checkpoint:
            if checkpoint["toward_classes"] != ["front", "back", "other"]:
                raise ValueError("无效 toward_classes")
            model.toward_classifier = torch.nn.Linear(3, 3).to(self.device)
        # 本项目旧微调产物按 0=女/1=男训练，但未记录类别元数据。
        # 仅识别其约定文件名；原始权重保留 0=男/1=女，显式元数据优先。
        default_gender_classes = (
            ("female", "male") if self._facexformer_weights.name == "facexformer_finetuned.pt"
            else ("male", "female")
        )
        gender_classes = checkpoint.get("gender_classes", default_gender_classes)
        override = os.environ.get("FACEXFORMER_GENDER_ORDER")
        if override is not None:
            gender_classes = [name.strip() for name in override.split(",")]
        if not isinstance(gender_classes, (list, tuple)) or tuple(gender_classes) not in (
            ("male", "female"), ("female", "male")
        ):
            raise ValueError("FaceXFormer gender_classes 必须为 male,female 或 female,male")
        self._facexformer_gender_classes = tuple(gender_classes)
        print(f"SDK 性别类别顺序: {list(gender_classes)} 权重: {self._facexformer_weights}", flush=True)
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
        has_mask_head = any("mask_head" in key for key in checkpoint.get("state_dict_om", {}))
        print(
            "SDK 口罩权重检查: "
            f"platform_heads={checkpoint.get('platform_heads')} "
            f"mask_classes={checkpoint.get('mask_classes')} "
            f"has_mask_head={has_mask_head} 权重={self._swinface_weights}",
            flush=True,
        )
        if checkpoint.get("platform_heads") == 1:
            if checkpoint.get("glasses_classes") != ["no_glasses", "glasses", "sunglasses"] or checkpoint.get("mask_classes") != ["no_mask", "mask"]:
                raise ValueError("无效平台眼镜/口罩类别顺序")
            model.om.enable_platform_heads()
        model.backbone.load_state_dict(checkpoint["state_dict_backbone"])
        model.fam.load_state_dict(checkpoint["state_dict_fam"])
        model.tss.load_state_dict(checkpoint["state_dict_tss"])
        model.om.load_state_dict(checkpoint["state_dict_om"])
        model.eval()
        print(f"SDK SwinFace 精简推理: 保留分支={model.om.sdk_branches()} / 总分支=11", flush=True)
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
        return self._run_facexformer_batch([face_crop])[0]

    @torch.no_grad()
    def _run_facexformer_batch(self, face_crops: list) -> list:
        """保持逐图预处理，将同帧目标合并为一次前向和结果传输。"""
        if not face_crops:
            return []
        image = torch.cat([self._preprocess_facexformer(crop) for crop in face_crops], dim=0).to(self.device)
        batch_size = len(face_crops)

        # SDK 只执行朝向、年龄、性别、人种输出头。共享注意力和权重不变。
        _, headpose_output, _, _, age_output, gender_output, race_output, _ = self._facexformer(
            image, None, None, sdk_only=True
        )

        age_rows = age_output.reshape(batch_size, -1)
        age = age_rows[:, 0] if age_rows.shape[1] == 1 else age_rows.argmax(1)
        values = [headpose_output, age.unsqueeze(1), gender_output.argmax(1).unsqueeze(1),
                  race_output.argmax(1).unsqueeze(1)]
        has_classifier = hasattr(self._facexformer, "toward_classifier")
        if has_classifier:
            values.append(self._facexformer.toward_classifier(headpose_output).argmax(1).unsqueeze(1))
        # One device-to-host copy; keep the original pose dtype for angle thresholds.
        packed = torch.cat(values, dim=1).cpu().numpy()
        results = []
        for row in packed:
            orientation = (["front", "back", "other"][int(row[6])] if has_classifier else
                           self._headpose_to_orientation(row[:3]))
            results.append({
                "orientation": orientation,
                "age": float(row[3]),
                "gender": self._facexformer_gender_classes[int(row[4])],
                "race": int(row[5]),
            })
        return results

    @torch.no_grad()
    def _run_swinface(self, face_crop: np.ndarray) -> dict:
        """运行 SwinFace，返回年龄、表情及 CelebA 属性。"""
        return self._run_swinface_batch([face_crop])[0]

    @torch.no_grad()
    def _run_swinface_batch(self, face_crops: list) -> list:
        if not face_crops:
            return []
        img = torch.cat([self._preprocess_swinface(crop) for crop in face_crops], dim=0).to(self.device)
        output = self._swinface(img, sdk_only=True)

        values = {
            "age": output["Age"].reshape(-1),
            "expression": output["Expression"].argmax(1),
            "smiling": _positive_class_batch(output["Smiling"]),
        }
        # CelebA 属性头在 SwinFace 中是 2 分类输出（shape (B, 2)），
        # 必须取正类概率，不能直接对 (B, 2) 张量做 sigmoid().item()。
        if output["Eyeglasses"].shape[-1] == 3:
            values["glasses_class"] = output["Eyeglasses"].argmax(1)
        else:
            values["eyeglasses"] = _positive_class_batch(output["Eyeglasses"])
        if "Mask" in output:
            values["mask"] = _positive_class_batch(output["Mask"])
        values["wearing_hat"] = _positive_class_batch(output["Wearing Hat"])
        values["mustache"] = _positive_class_batch(output["Mustache"])
        values["no_beard"] = _positive_class_batch(output["No Beard"])
        # Preserve all decoded fields, but transfer their scalar values together.
        packed = torch.stack(list(values.values()), dim=1).cpu().tolist()
        results = []
        for row in packed:
            result = dict(zip(values, row))
            result["expression"] = SWINFACE_EXPRESSION_CLASSES[int(result["expression"])]
            if "glasses_class" in result:
                result["glasses_class"] = int(result["glasses_class"])
            results.append(result)
        return results

    @staticmethod
    def _headpose_to_orientation(euler: np.ndarray) -> str:
        """根据欧拉角（弧度）判断人脸朝向。

        榜单只认三类：``front`` 正面（两眼可见）、``back`` 背面（两眼都不可见）、
        ``other`` 其它（侧面等）。角度阈值是当前实现的近似，不是官方双眼可见规则；
        训练验证使用同一阈值，但真实朝向效果仍需标注数据验证。训练目标为：
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
