# syntax=docker/dockerfile:1
#
# Reference build for the AMD AI League Match 3 container.
#
#   docker build -t provenance-rag:local .
#   docker run -d --network none --cap-drop DAC_OVERRIDE --device /dev/kfd --device /dev/dri \
#       --name prag provenance-rag:local
#   docker cp ./my-corpus/. prag:/app/corpus
#   docker exec prag python3 /app/app.py --index /app/corpus
#   docker exec prag python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
#
# Never squash this image: the final stage must keep the mandated base layers intact.

# ---- helper stage: fetch the weights at BUILD time (the network is closed at evaluation) ----
FROM python:3.12-slim AS weights
ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
RUN pip install --no-cache-dir "huggingface_hub[hf_xet]>=0.34" \
 && python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', local_dir='/models/Qwen3-VL-8B-Instruct', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])" \
 && rm -rf /models/Qwen3-VL-8B-Instruct/.cache

# ---- final stage: the mandated ROCm base ----
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONPATH=/opt/site:/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_DIR=/models/Qwen3-VL-8B-Instruct \
    PRAG_BACKEND=transformers \
    PRAG_INDEX_DIR=/app/index \
    PRAG_OUTPUT_DIR=/app/output \
    PRAG_SOCKET=/tmp/provenance_rag.sock

# Weights first: the largest layer, and identical across rebuilds.
COPY --from=weights /models /models

# Extra packages go to /opt/site with --no-deps so ROCm torch is never replaced.
COPY requirements-image.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps --target /opt/site -r /app/requirements.txt \
 && python3 -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__); assert '+rocm' in torch.__version__, torch.__version__"

COPY app/ /app/
COPY provenance_rag/ /app/provenance_rag/
RUN mkdir -p /app/corpus /app/output /app/index \
 && python3 -c "import provenance_rag, pymupdf, openpyxl, transformers; print('imports ok, transformers', transformers.__version__)"

WORKDIR /app
# Supervisor: starts the resident model server (unix socket), restarts it on crash, never exits.
CMD ["python3", "/app/server.py"]
