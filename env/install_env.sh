#!/usr/bin/env bash
# =============================================================================
# 极市平台 - 人脸属性分析项目 环境安装脚本
# -----------------------------------------------------------------------------
# 目标环境:
#   Ubuntu 18.04
#   CUDA 11.3
#   cuDNN 8.2
#   Miniconda3
#   JupyterLab 3.3
#   Python 3.7
#   PyTorch 1.11 (cu113)
#
# 用法:
#   bash install_env.sh
# =============================================================================
set -e

echo "============================================================"
echo " 开始安装人脸属性分析项目环境"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. 检查并安装 Miniconda3
# ---------------------------------------------------------------------------
MINICONDA_DIR="$HOME/miniconda3"
if [ ! -d "$MINICONDA_DIR" ]; then
    echo "[1/6] 安装 Miniconda3 ..."
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f /tmp/miniconda.sh
else
    echo "[1/6] Miniconda3 已存在，跳过安装"
fi

# 初始化 conda（若尚未初始化）
if ! grep -q "miniconda3" "$HOME/.bashrc" 2>/dev/null; then
    "$MINICONDA_DIR/bin/conda" init bash
fi
source "$HOME/.bashrc"

# ---------------------------------------------------------------------------
# 2. 创建 Python 3.7 环境
# ---------------------------------------------------------------------------
ENV_NAME="face_attr"
echo "[2/6] 创建 conda 环境: $ENV_NAME (python=3.7) ..."
if ! conda env list | grep -q "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.7 -y
fi
source activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 3. 配置清华源（极市平台推荐）
# ---------------------------------------------------------------------------
echo "[3/6] 配置 pip 清华源 ..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# ---------------------------------------------------------------------------
# 4. 安装 PyTorch 1.11 + CUDA 11.3
# ---------------------------------------------------------------------------
echo "[4/6] 安装 PyTorch 1.11 (cu113) ..."
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ---------------------------------------------------------------------------
# 5. 安装 JupyterLab 3.3
# ---------------------------------------------------------------------------
echo "[5/6] 安装 JupyterLab 3.3 ..."
pip install "jupyterlab==3.3.*"

# ---------------------------------------------------------------------------
# 6. 安装项目依赖
# ---------------------------------------------------------------------------
echo "[6/6] 安装项目依赖 ..."
pip install -r "$(dirname "$0")/../requirements.txt"

echo "============================================================"
echo " 环境安装完成！"
echo " 激活环境:  conda activate $ENV_NAME"
echo " 启动 JupyterLab: jupyter lab"
echo "============================================================"
