# 構建階段
FROM nvidia/cuda:12.6.1-base-ubuntu24.04

WORKDIR /app

# 更新並安裝基本工具
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 設置 CUDA 環境變量
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

CMD ["nvidia-smi"]