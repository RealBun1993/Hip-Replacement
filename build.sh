#!/bin/bash

# 确保 Python 和 pip 都安装了
python3 -m ensurepip --upgrade || true

# 升级 pip 到最新版本
python3 -m pip install --upgrade pip

# 安装 setuptools 和 wheel
python3 -m pip install setuptools wheel

# 安装项目依赖
python3 -m pip install -r requirements.txt
