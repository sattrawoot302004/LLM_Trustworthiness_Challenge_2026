ARG VLLM_IMAGE=vllm/vllm-openai:v0.10.2
FROM ${VLLM_IMAGE}

USER root
WORKDIR /workspace

COPY requirements.lock /workspace/requirements.lock
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.lock

COPY run.py /workspace/run.py
COPY app /workspace/app
COPY configs /workspace/configs
COPY prompts /workspace/prompts

COPY models/generator /opt/models/generator
COPY models/qwen_guard /opt/models/qwen_guard
COPY models/thai_guard /opt/models/thai_guard

RUN mkdir -p /model/test /result /benchmark_lib
RUN test -f /opt/models/generator/config.json \
    || (echo "Missing models/generator/config.json. Run: python scripts/download_models.py" && exit 1)
RUN test -f /opt/models/qwen_guard/config.json \
    || (echo "Missing models/qwen_guard/config.json. Run: python scripts/download_models.py" && exit 1)
RUN test -f /opt/models/thai_guard/config.json \
    || (echo "Missing models/thai_guard/config.json. Run: python scripts/merge_thai_guard.py after download_models.py" && exit 1)

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV MODEL_ROOT=/opt/models

ENTRYPOINT []
CMD ["python3", "/workspace/run.py"]
