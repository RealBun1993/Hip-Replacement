#!/bin/bash

# 确保使用 Python 3.8 创建虚拟环境
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

# 安装 distutils (distlib 可作为 distutils 的替代)
pip install distlib

# 安装项目依赖
pip install -r requirements.txt
