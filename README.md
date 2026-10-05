# 人脸识别（人脸属性分析）

基于 **Python + PyTorch** 的人脸属性分析项目，用于检测图像中的人脸并分析年龄、性别、表情、朝向等属性。

按照**极市开发者平台**规范组织项目结构，可直接在极市平台进行模型开发、训练、测试。

已接入头肩/人体检测训练、眼镜三分类、口罩及朝向分类分支，需要线上重训产生新权重。未挂载头肩权重时保留原检测流程和输出协议。操作见 [头肩检测升级与线上重训](docs/头肩检测升级与线上重训.md)，规则核对见 [赛道 11022 要求核对](docs/赛道11022要求核对.md)。

## 功能

- **检测**：优先使用按平台标注训练的头肩/人体检测器；缺少新权重时兼容原 MTCNN 人脸检测。
- **属性分析**：基于 PyTorch 轻量级网络，输出年龄、性别、表情、朝向等属性。
- **结果输出**：以 JSON 格式输出检测与属性结果。

## 环境要求

| 组件 | 版本 |
|------|------|
| 操作系统 | Ubuntu 18.04 |
| CUDA | 11.3 |
| cuDNN | 8.2 |
| Miniconda | 3 |
| JupyterLab | 3.3 |
| Python | 3.7 |
| PyTorch | 1.11 (cu113) |

## 目录结构（极市平台规范）

```text
人脸识别/
├── project/                  # 可整体上传的云平台工程容器
│   ├── train/                # 模型开发（对应 /project/train/）
│   └── ev_sdk/               # 算法开发（对应 /project/ev_sdk/）
├── env/
│   └── install_env.sh        # 环境安装脚本
├── tests/                    # 单元测试
├── data/                     # 本地数据（不入库）
├── docs/                     # 文档
├── pyproject.toml            # 项目配置
└── requirements.txt          # 依赖
```

GitHub 仓库保留源码、文档和测试，不包含模型权重及本地数据。需要保留的空目录使用 `.gitkeep` 占位；克隆后请按 `docs/线上平台上传清单.md` 准备 SDK 权重，或在平台测试时选择训练任务产物。`.pt`、`.pth`、`.ckpt`、`.onnx`、`.safetensors` 文件在所有目录中均被忽略，本地虚拟环境和缓存也不上传。

## 安装

极市平台依赖及安装、验证命令见 [线上平台依赖安装](docs/线上平台依赖安装.md)。

### 1. 安装环境（Ubuntu 18.04）

```bash
bash env/install_env.sh
```

脚本会自动完成：
1. 安装 Miniconda3
2. 创建 Python 3.7 环境 `face_attr`
3. 配置清华源
4. 安装 PyTorch 1.11 (cu113)
5. 安装 JupyterLab 3.3
6. 安装项目依赖

### 2. 手动安装（可选）

```bash
# 创建 conda 环境
conda create -n face_attr python=3.7 -y
conda activate face_attr

# 安装 PyTorch 1.11 + CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 安装 JupyterLab 3.3
pip install "jupyterlab==3.3.*"

# 安装项目依赖
pip install -r requirements.txt
```

## 使用

### 极市平台训练

将本地 `project/train/` 目录内容上传到极市平台 `/project/train/`，然后在平台发起训练：

```bash
bash /project/train/src_repo/run.sh
```

### 极市平台测试

将本地 `project/ev_sdk/` 目录内容上传到极市平台 `/project/ev_sdk/`，发起模型测试。

## 测试

```bash
pytest
```

## 后续规划

- 接入 FaceXFormer：朝向、年龄、性别、种族、可见性
- 接入 SwinFace：表情、眼镜、帽子、胡须等属性
- 支持视频流与多目标跟踪
