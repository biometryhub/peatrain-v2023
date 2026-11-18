FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && pip install "tensorflow[and-cuda]" opencv-python-headless pandas numpy

ARG USERNAME=vscode
RUN useradd -m -s /bin/bash ${USERNAME}
USER ${USERNAME}
