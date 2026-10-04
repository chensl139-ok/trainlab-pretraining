# Selected CUDA 13.0 stack; validate driver and kernels on the target RTX PRO machine.
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
LABEL org.opencontainers.image.source="https://github.com/chensl139-ok/trainlab-pretraining"
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/state/cache TRAINLAB_STATE_DIR=/state HOME=/state
# Reuse CUDA-enabled PyTorch; install app packages in an isolated virtualenv.
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv --system-site-packages /opt/trainlab-venv
ENV PATH="/opt/trainlab-venv/bin:${PATH}"
COPY server/requirements-api.txt server/requirements-train.txt /app/server/
RUN pip install --no-cache-dir --upgrade pip==26.2.1 && pip install --no-cache-dir -r server/requirements-train.txt && pip check \
    && python -c "import torch; assert torch.__version__.split('+')[0] == '2.13.0'; assert torch.version.cuda.startswith('13.0')"
COPY server /app/server
COPY dist /app/dist
COPY scripts /app/scripts
COPY examples /app/examples
RUN mkdir -p /state && chown -R 10001:10001 /state /app
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "server.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "64", "--timeout-keep-alive", "5", "--no-access-log"]
