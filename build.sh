#!/bin/bash

# 更新 pip 和 setuptools
python3 -m pip install --upgrade pip
pip install --upgrade setuptools==65.5.0

# 安装依赖
pip install -r requirements.txt
