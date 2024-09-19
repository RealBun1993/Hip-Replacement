#!/bin/bash

# 安装 setuptools 和 distutils
pip install setuptools==65.5.0

# 执行默认的构建命令
pip install --disable-pip-version-check --upgrade -r requirements.txt
