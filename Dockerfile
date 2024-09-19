# 使用 Python 3.8 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 文件到工作目录
COPY requirements.txt .

# 安装 Python 依赖和 distutils
RUN apt-get update && apt-get install -y python3-distutils \
    && pip install --no-cache-dir -r requirements.txt

# 复制项目所有内容到工作目录
COPY . .

# 暴露 Flask 的默认端口
EXPOSE 5000

# 运行 Flask 应用
CMD ["python", "./api/generate_data.py"]
