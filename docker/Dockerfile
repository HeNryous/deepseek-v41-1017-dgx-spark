# Overlay on top of the sudoingX dgx-spark-deepseek-v41 v1.1 serving image (built locally with its install.sh,
# tag dsv41-serve:v4). Adds our patches 0031/0031c (fair scheduling), 0032 (wo_a quant-dequant as one Triton kernel)
# and 0033 (RoPE cache capped at max-model-len, smaller indexer prefill buffer). Patch 0030 (1017 reference parity)
# is already part of the base image build. The published image ghcr.io/henryous/deepseek-v41-1017-dgx-spark is
# exactly this.
ARG BASE=dsv41-serve:v4
FROM ${BASE}
LABEL org.opencontainers.image.source="https://github.com/HeNryous/deepseek-v41-1017-dgx-spark" \
      org.opencontainers.image.description="vLLM serving image for DeepSeek-V4.1-Flash-1017 EXL3 on 2x DGX Spark (GB10), TP2" \
      org.opencontainers.image.licenses="Apache-2.0"
ARG SP=/usr/local/lib/python3.12/dist-packages/vllm
COPY files/core.py ${SP}/v1/engine/core.py
COPY files/scheduler.py ${SP}/v1/core/sched/scheduler.py
COPY files/o_proj.py ${SP}/models/deepseek_v4/nvidia/ops/o_proj.py
COPY files/deepseek_scaling_rope.py ${SP}/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py
COPY files/indexer.py ${SP}/v1/attention/backends/mla/indexer.py
