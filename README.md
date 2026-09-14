# 人脸识别（人脸属性分析）

基于 **Python + PyTorch** 的人脸属性分析项目，用于检测图像中的人脸并分析年龄、性别、表情、朝向等属性。

按照**极市开发者平台**规范组织项目结构，可直接在极市平台进行模型开发、训练、测试。

## 功能

- **人脸检测**：基于 MTCNN 检测图像中的人脸，返回边界框与置信度。
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
├── env/
│   └── install_env.sh        # 环境安装脚本
├── train/                    # 模型开发（对应 /project/train/）
│   ├── src_repo/             # 训练代码（对应 /project/train/src_repo/）
│   │   ├── train.py          # 训练主脚本
│   │   ├── model.py          # 模型定义
│   │   ├── dataset.py        # 数据集加载
│   │   └── run.sh            # 训练命令脚本
│   ├── models/               # 模型保存（对应 /project/train/models/）
│   ├── log/                  # 日志保存（对应 /project/train/log/）
│   └── result-graphs/        # 训练结果图（对应 /project/train/result-graphs/）
├── ev_sdk/                   # 算法开发（对应 /project/ev_sdk/）
│   └── src/
│       └── ji.py             # 自动测试脚本
├── src/face_attr/            # 本地核心源码
├── tests/                    # 单元测试
├── models/                   # 本地模型权重（不入库）
├── data/                     # 本地数据（不入库）
├── docs/                     # 文档
├── pyproject.toml            # 项目配置
└── requirements.txt          # 依赖
```

## 安装

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

### 本地运行

```bash
# 处理单张图像
python -m face_attr.main path/to/image.jpg --device cpu

# 保存结果到文件
python -m face_attr.main path/to/image.jpg --output result.json
```

### 极市平台训练

将 `train/` 目录内容上传到极市平台 `/project/train/`，然后在平台发起训练：

```bash
bash /project/train/src_repo/run.sh
```

### 极市平台测试

将 `ev_sdk/src/ji.py` 放到 `/project/ev_sdk/src/`，发起模型测试。

## 测试

```bash
pytest
```

## 后续规划

- 接入 FaceXFormer：朝向、年龄、性别、种族、可见性
- 接入 SwinFace：表情、眼镜、帽子、胡须等属性
- 支持视频流与多目标跟踪
