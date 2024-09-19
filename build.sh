#!/bin/bash

set -ex  # 开启详细模式，输出所有命令和错误

# 检查是否安装了 python3.8
if command -v python3.8 &>/dev/null; then
    echo "Using Python 3.8"
    python3.8 -m venv venv
else
    echo "Error: Python 3.8 not found. Please ensure Python 3.8 is installed."
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 升级 pip 到最新版本
pip install --upgrade pip

# 安装 setuptools 和 wheel
pip install setuptools==59.6.0 wheel

# 安装依赖的构建工具，如 Cython 和 numpy
pip install Cython numpy

# 安装项目依赖
pip install -r requirements.txt
