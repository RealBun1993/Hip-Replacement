#!/bin/bash

# 使用 python3 -m pip 来安装 setuptools 和 distutils
python3 -m pip install setuptools==65.5.0

# 执行默认的构建命令
python3 -m pip install --disable-pip-version-check --upgrade -r requirements.txt
