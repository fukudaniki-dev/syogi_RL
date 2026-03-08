FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# dlshogi を submodule のソースからビルドしてインストール
COPY dlshogi/ /src/dlshogi/
RUN pip install --no-cache-dir cython numpy scipy && \
    pip install --no-cache-dir /src/dlshogi/ && \
    pip install --no-cache-dir fastapi "uvicorn[standard]"

WORKDIR /workspace
COPY docker/inference_test.py .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
