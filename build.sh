#!/bin/bash

# 使用指定的 Python 版本创建虚拟环境
python3.8 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级 pip 到最新版本
pip install --upgrade pip

# 安装 setuptools 和 wheel，指定稳定版本
pip install setuptools==59.6.0 wheel

# 安装 distutils
pip install python3-distutils

# 安装项目依赖
pip install -r requirements.txt
