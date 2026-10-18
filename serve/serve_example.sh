#!/bin/bash
# Example: serve DeepSeek-V4.1-Flash-1017 EXL3 with TP2 on two DGX Sparks, using the launcher of the sudoingX recipe
# (scripts/launch/dsv41_tp2.sh from github.com/sudoingX/dgx-spark-deepseek-v41). Run on the head node; the worker is
# reached over ssh. Fill in the placeholders. The model folder must exist on both nodes at the same path.
set -euo pipefail
RECIPE=/path/to/dgx-spark-deepseek-v41
export HEAD=<head-hostname> WORKER=<worker-hostname>
export FAB_IF=<fabric-netdev, e.g. enp1s0f0np0> FAB_HCA=<RoCE device on it, e.g. rocep1s0f0>
export IMAGE=ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v4h
export MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark   # incl. the two official Engram shards
export SERVED=DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2
export TEXT_ONLY=0 MAX_LEN=327680 MAX_SEQS=4 MAX_BATCH=4096 UTIL=0.86
export KV_BYTES=4026531840                       # 3.75 GiB per rank -> 989,460 tokens; UTIL then only gates the startup check
# keep the background long-prefill threshold <= MAX_BATCH / 4, otherwise background prefills can fill a whole step and a
# new foreground request waits for them (see patches/0031c)
export EXTRA="--limit-mm-per-prompt '{\"image\":4}' --mm-processor-cache-gb 0 --long-prefill-token-threshold $((MAX_BATCH / 4))"
# 0031: requests with priority >= 5 count as background and prefill only every 24th step while a foreground request runs.
# NCCL over both PCIe functions of the ConnectX port (+3 % prefill); the plugin's RDMA ring stays on one device.
export DOCKER_ENV="NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 VLLM_DSPARK_ROUND_MS=33.26,36.33,38.91,42.23,46.03,48.15 \
VLLM_PREFILL_EVERY_N_STEPS=24 VLLM_PREFILL_THROTTLE_MIN_PRIO=5 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
NCCL_IB_HCA=<rdma-dev-1>,<rdma-dev-2> EXL3_RDMA_DEVICES=<rdma-dev-1> NCCL_PROTO=^LL128 NCCL_CUMEM_ENABLE=0 \
NCCL_CUMEM_HOST_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
SKIP_SERVE_HOLD=1 bash "$RECIPE/scripts/launch/dsv41_tp2.sh" up
