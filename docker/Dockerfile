# Overlay on top of the sudoingX dgx-spark-deepseek-v41 v1.1 serving image (built locally with its install.sh,
# tag dsv41-serve:v4; patch 0030 is part of that build). docker/files/ mirrors site-packages and adds our patches:
#   0031/0031c fair scheduling · 0032 wo_a quant-dequant as one Triton kernel · 0033 RoPE/indexer memory trims
#   0034 replay trims layer 20 (vLLM #59894) · 0035 dense prompt path without copies · 0036 indexer rows split over TP
#   0041 joint DSpark depth choice · 0042b exl3_mrow up to 32 rows · 0043 prompt-lookup drafts
#   0045 decode all-gathers over the RDMA ring · 0046 row limit for the LM head on exl3_mrow · 0047 adaptive verify window
#   0048 mHC boundary in one launch · 0049 router prefetch build flags · 0050 mHC overlap up to 24 rows
#   0022 Triton verify attention (from the recipe's patches/experimental/, unchanged)
#   0051 indexer weights GEMV deferred · 0053/0054 display reservation as memory · 0055 hybrid drafting
#   0056 CRC for the file-system KV offload tier · 0057 paired KV pages · 0058 offload next to expandable_segments
#   0059 NVFP4 KV records · 0060 three KV pages per block · 0061 FP8 window + NVFP4 compressed KV
# New switches are off by default (0040 is configuration only, see serve/serve_example.sh). exl3_mrow, exl3_rdma,
# exl3_mhc and exl3_router have new sources: build the plugin's JIT kernels before the first start (README, step 4).
# The published image ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v12 (= :latest) has exactly these files.
ARG BASE=dsv41-serve:v4
FROM ${BASE}
LABEL org.opencontainers.image.source="https://github.com/HeNryous/deepseek-v41-1017-dgx-spark" \
      org.opencontainers.image.description="vLLM serving image for DeepSeek-V4.1-Flash-1017 EXL3 on 2x DGX Spark (GB10), TP2" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY files/ /usr/local/lib/python3.12/dist-packages/
