#!/bin/bash

# 确保 Python 和 pip 都安装了
python3 -m ensurepip --upgrade || true

# 升级 pip 到最新版本
python3 -m pip install --upgrade pip

# 卸载任何现有的 setuptools
python3 -m pip uninstall -y setuptools

# 安装 setuptools, wheel 和 six
python3 -m pip install "setuptools==65.0.0" wheel six

# 安装项目依赖
python3 -m pip install -r requirements.txt
