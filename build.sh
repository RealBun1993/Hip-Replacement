#!/bin/bash

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级 pip 到最新版本
pip install --upgrade pip

<<<<<<< HEAD
# 安装 setuptools 和 wheel，指定一个较旧的稳定版本
pip install setuptools==59.6.0 wheel
=======
# 安装 setuptools 和 wheel
pip install setuptools wheel
>>>>>>> df62c378496a714879fea115461c2f8537b36557

# 安装项目依赖
pip install -r requirements.txt
