#!/bin/bash

set -ex  # 开启详细模式，输出所有命令和错误

# 强制使用 Python 3.8 创建虚拟环境，如果有 python3.8 可用
if command -v python3.8 &>/dev/null; then
    python3.8 -m venv venv
else
    echo "Python 3.8 not found, using default python3"
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 升级 pip 到最新版本
pip install --upgrade pip

# 安装 setuptools 和 wheel
pip install setuptools==59.6.0 wheel

# 安装 distutils，如果需要
pip install distlib

# 安装依赖的构建工具，如 Cython 和 numpy
pip install Cython numpy

# 安装项目依赖
pip install -r requirements.txt
