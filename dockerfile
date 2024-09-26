# 使用最新的 PyTorch 鏡像，包含 CUDA 支持
FROM pytorch/pytorch:latest

# 設置工作目錄
WORKDIR /app

# 安裝額外的依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製 Python 測試腳本到容器中
COPY simple.py .

# 設置環境變量
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# 運行 Python 腳本
CMD ["python", "simple.py"]