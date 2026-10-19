# Overlay on top of the sudoingX dgx-spark-deepseek-v41 v1.1 serving image (built locally with its install.sh,
# tag dsv41-serve:v4; patch 0030 is part of that build). docker/files/ mirrors site-packages and adds our patches:
#   0031/0031c fair scheduling · 0032 wo_a quant-dequant as one Triton kernel · 0033 RoPE/indexer memory trims
#   0034 replay trims layer 20 (vLLM #59894) · 0035 dense prompt path without copies · 0036 indexer rows split over TP
# The published image ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v4i (= :latest) is exactly this.
ARG BASE=dsv41-serve:v4
FROM ${BASE}
LABEL org.opencontainers.image.source="https://github.com/HeNryous/deepseek-v41-1017-dgx-spark" \
      org.opencontainers.image.description="vLLM serving image for DeepSeek-V4.1-Flash-1017 EXL3 on 2x DGX Spark (GB10), TP2" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY files/ /usr/local/lib/python3.12/dist-packages/
