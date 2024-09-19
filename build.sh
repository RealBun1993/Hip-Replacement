#!/bin/bash

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级 pip 到最新版本
pip install --upgrade pip

# 安装 setuptools 和 wheel 以及 six
pip install "setuptools==58.0.0" wheel six

# 安装项目依赖
pip install -r requirements.txt

# 继续执行其他步骤
