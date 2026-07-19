ARG VLLM_IMAGE=vllm/vllm-openai:v0.10.2
FROM ${VLLM_IMAGE}

USER root
WORKDIR /workspace

COPY requirements.lock /workspace/requirements.lock
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.lock

RUN mkdir -p /workspace/scripts
COPY scripts/download_models.py /workspace/scripts/download_models.py
RUN HF_HOME=/tmp/hf-cache MODEL_DOWNLOAD_DIR=/opt/models python3 /workspace/scripts/download_models.py \
    && rm -rf /opt/models/*/.cache /tmp/hf-cache

COPY run.py /workspace/run.py
COPY app /workspace/app
COPY configs /workspace/configs
COPY prompts /workspace/prompts

RUN mkdir -p /model/test /result /benchmark_lib
RUN test -f /opt/models/generator/config.json \
    || (echo "Missing /opt/models/generator/config.json from Docker build download" && exit 1)
RUN test -f /opt/models/qwen_guard/config.json \
    || (echo "Missing /opt/models/qwen_guard/config.json from Docker build download" && exit 1)
RUN test -f /opt/models/thai_guard/config.json \
    || (echo "Missing /opt/models/thai_guard/config.json from Docker build download" && exit 1)

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV MODEL_ROOT=/opt/models

ENTRYPOINT []
CMD ["python3", "/workspace/run.py"]
