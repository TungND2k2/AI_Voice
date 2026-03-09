FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV TZ=UTC DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV POETRY_VERSION=1.8.3

RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

ENV POETRY_CACHE_DIR=/tmp/poetry_cache
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_REQUESTS_TIMEOUT=15

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ libc-dev libffi-dev libgmp-dev libmpfr-dev libmpc-dev \
    ffmpeg \
    sox

RUN apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY ./src /app/src
COPY ./samples /app/samples
COPY ./web /app/web
COPY ./pyproject.toml /app/
COPY ./README.md /app/

RUN conda install -y setuptools && \
    pip install --upgrade pip "setuptools<81" wheel && \
    pip install --no-build-isolation "openai-whisper==20240930" && \
    pip install -e . && \
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true && \
    pip install --no-cache-dir --force-reinstall "onnx==1.14.1" "onnxruntime-gpu==1.15.1" && \
    pip install --no-cache-dir "numpy<2.0" && \
    pip install "ruamel.yaml<0.18" && \
    pip cache purge