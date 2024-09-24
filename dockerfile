# FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
# FROM nvidia/cuda:11.8.0-base-ubuntu22.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04
# FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04
# FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

# 更新並安裝基本工具
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安裝 Python 和必要的工具
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 創建並激活虛擬環境
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 安裝 TensorFlow 和 Matplotlib
RUN pip install --no-cache-dir tensorflow matplotlib
    
# 安裝 CUDA 相關庫
# RUN apt-get update && \
#     apt-get install -y --allow-change-held-packages \
#         libcudnn8 \
#         libnvinfer8 \
#         libnvinfer-plugin8 && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*    

# 設置 CUDA 環境變量
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

COPY *.py .

CMD ["python", "gpu_test.py"]
# CMD ["nvidia-smi"]
# CMD tail -f /dev/null