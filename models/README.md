# 模型权重目录

本目录用于存放本地推理所需的模型权重。**权重文件体积较大（数百 MB），不纳入 Git 版本控制**，需自行下载后按下列路径放置。

## 目录结构

```text
models/
├── facexformer/
│   └── model.pt                          # FaceXFormer 官方权重（约 696 MB）
└── swinface/
    └── checkpoint_step_79999_gpu_0.pt    # SwinFace 官方权重（约 842 MB）
```

## 下载地址

| 模型 | 用途 | 下载地址 |
|------|------|----------|
| FaceXFormer | 朝向、年龄、性别、种族、可见性 | https://github.com/KanchanKumar/FaceXFormer |
| SwinFace | 表情、眼镜、帽子、胡须等属性 | https://github.com/lxq1000/SwinFace |

下载后请严格按上表路径重命名并放置，`src/face_attr/analyzer.py` 中的默认路径依赖该约定：

```python
DEFAULT_FACEXFORMER_WEIGHTS = _PROJECT_ROOT / "models" / "facexformer" / "model.pt"
DEFAULT_SWINFACE_WEIGHTS = _PROJECT_ROOT / "models" / "swinface" / "checkpoint_step_79999_gpu_0.pt"
```

也可以通过 `AttributeAnalyzer(facexformer_weights=..., swinface_weights=...)` 显式指定其他路径。
