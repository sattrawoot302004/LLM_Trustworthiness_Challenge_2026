FROM vllm/vllm-openai:v0.19.0

WORKDIR /workspace

COPY requirements.lock /workspace/requirements.lock
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.lock

# Fail during image build instead of evaluator startup if the official runtime
# ever lacks a compiler or CUDA JIT component required by vLLM/Triton.
RUN command -v gcc \
    && command -v g++ \
    && command -v nvcc \
    && python3 -c "import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__)"

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

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
RUN test -f /opt/models/thai_guard/config.json \
    || (echo "Missing /opt/models/thai_guard/config.json from Docker build download" && exit 1)

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV MODEL_ROOT=/opt/models

# vllm/vllm-openai normally starts `vllm serve`; this submission runs its
# offline batch pipeline directly instead.
ENTRYPOINT []
CMD ["python3", "/workspace/run.py"]
