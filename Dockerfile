FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3.11-dev \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの作成
WORKDIR /app

# whisper-finetuneリポジトリをコピー
COPY whisper-finetune /app/whisper-finetune

# Pythonの仮想環境を作成
RUN python3.11 -m venv /app/env_whisper-finetune

# 環境変数の設定
ENV PATH="/app/env_whisper-finetune/bin:$PATH"
ENV PYTHONPATH="/app"

# 仮想環境をアクティベートしてパッケージをインストール
RUN . /app/env_whisper-finetune/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/whisper-finetune/requirements.txt

# git-lfsの初期化
RUN git lfs install

# whisper-jaxのためのJAX環境変数
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

# エントリーポイントスクリプトの作成
RUN echo '#!/bin/bash\n\
source /app/env_whisper-finetune/bin/activate\n\
exec "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# エントリーポイントの設定
ENTRYPOINT ["/app/entrypoint.sh"]

# デフォルトコマンド
CMD ["bash"] 