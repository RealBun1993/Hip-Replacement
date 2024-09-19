# 使用 Python 3.8 的官方镜像
FROM python:3.8-slim

# 设置环境变量，防止缓存生成并确保日志立即输出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 将 requirements.txt 复制到容器中
COPY requirements.txt /app/

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码到容器中
COPY . /app/

# 暴露 Flask 应用运行的端口
EXPOSE 5000

# 运行 Flask 应用
CMD ["python", "api/generate_data.py"]
