#!/bin/bash
# Example: serve DeepSeek-V4.1-Flash-1017 EXL3 with TP2 on two DGX Sparks, using the launcher of the sudoingX recipe
# (scripts/launch/dsv41_tp2.sh from github.com/sudoingX/dgx-spark-deepseek-v41). Run on the head node; the worker is
# reached over ssh (passwordless). Usage:  HEAD=<head-host> WORKER=<worker-host> bash serve/serve_example.sh
# The model folder must exist on both nodes at the same path. Start it with the server stopped (see the JIT step below).
set -euo pipefail
# 0047 adaptive verify window: margin over the accepted-draft average, smallest window, threshold above which a request
# keeps the full window (two regimes), and up to how many requests a step adapts
AK_MARGIN=${AK_MARGIN:-1.0}
AK_MIN=${AK_MIN:-2}
AK_HI=${AK_HI:-1.6}
AK_MAX_REQS=${AK_MAX_REQS:-4}
RECIPE=${RECIPE:-$HOME/dgx-spark-deepseek-v41}      # git clone https://github.com/sudoingX/dgx-spark-deepseek-v41
export HEAD=${HEAD:?set HEAD=<head node hostname>} WORKER=${WORKER:?set WORKER=<worker node hostname>}
export FAB_IF=${FAB_IF:-enp1s0f0np0} FAB_HCA=${FAB_HCA:-rocep1s0f0}   # fabric netdev and its RoCE device (DGX Spark names)
export IMAGE=ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v22
export MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark   # incl. the two 1017 Engram shards (scripts/download_model.sh)
export SERVED=DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2
export TEXT_ONLY=0 MAX_LEN=327680 MAX_SEQS=4 MAX_BATCH=4096 UTIL=0.86
export KV_BYTES=3758096384                       # 3.5 GiB per rank -> 1,429,164 tokens with 0057 + 0060 (3.75 GiB without the SSD offload
                                                 # below: 1,531,258); UTIL then only gates the startup check
# 0040: DSpark k = 5 at 1-4 streams (at most 24 verify rows, kept on exl3_mrow by 0042b; until 21.10. k = 4 at 3-4)
export SPEC='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5]]}'
# keep the background long-prefill threshold <= MAX_BATCH / 4, otherwise background prefills can fill a whole step and a
# new foreground request waits for them (see patches/0031c). --scheduling-policy priority: vLLM's waiting queue is
# ordered by the request field `priority` (lower first); --enable-prompt-tokens-details: usage reports cached prompt tokens.
export EXTRA="--limit-mm-per-prompt '{\"image\":4}' --mm-processor-cache-gb 0 --long-prefill-token-threshold $((MAX_BATCH / 4)) \
--scheduling-policy priority --enable-prompt-tokens-details"
# SSD KV offload (since 22.10.; 0056 CRC per block, 0058 next to expandable_segments): vLLM's OffloadingConnector, a 256 MiB
# CPU tier (taken from the KV pool above) and a file-system tier under ~/.cache/dsv41-serve/kv_offload that survives
# restarts. Prune it yourself (we delete blocks older than a day and cap the folder at 40 GB, hourly). Remove the line
# (and set KV_BYTES=4026531840) to serve without it.
export EXTRA="$EXTRA --kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":268435456,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"/root/.cache/kv_offload\",\"n_read_threads\":8,\"n_write_threads\":4}]}}'"
# 0031: requests with priority >= 5 count as background; with VLLM_PREFILL_EVERY_N_STEPS=100000 (until 21.10.: 24) their
# prefill runs only while no foreground request runs. 0073: also not while a foreground request waits for admission;
# 0074: background prefill chunks keep the long-prefill threshold when they run alone.
# NCCL and the plugin's RDMA ring over both PCIe functions of the ConnectX port (NCCL: +3 % prefill; the ring, measured
# together with 0051: -1.2 % per decode round).
# 0036: split the indexer prefill rows over both ranks from 2048 rows on. EXL3_RDMA_CPU: pin the RDMA ring's polling thread
# to an efficiency core (A725: cpus 0-4, 10-14) instead of the default last X925 core: same speed, hottest sensor ~10 C cooler.
# 0045: the decode round's TP all-gathers over the RDMA ring (up to 3 MB: DSpark Markov pairs, Engram rows, logits).
# 0046 + 0042b: the LM head on exl3_mrow up to 8 rows (1 stream); the dense layers on exl3_mrow up to 32 rows, from 9 rows
# on through 0042b's ring kernel. Recipe switches: router GEMM before the shared-expert fork, shared expert inside the
# routed-expert kernel. 0048: mHC boundary in one launch (bit-identical); 0050: mHC overlap path up to 24 rows.
# 0047: adaptive verify window (needs the 0040 schedule in SPEC), values at the top.
# 0057 + 0060: the ratio-1 KV group in blocks of three 64-state kernel pages (no allocator padding);
# VLLM_DSV41_KV_SWA_FILL=1 is part of the measured configuration and has no effect with FP8 records.
# 0051: the indexer's weights GEMV off the main stream's critical path (bit-identical). 0064: MoE calls of 33-63 rows
# (mixed steps, short prompts) as exl3_smoe row chunks without host syncs; EXL3_CPU_PIN: every thread of the vLLM
# processes on the Cortex-X925 cores (cpus 5-9, 15-19). 0066 + 0067: the DSpark Markov loop in fused launches (same
# drafts) and the pre-round SWA metadata in one launch (same bits).
# 0071 + 0072: while a request with priority < 10 runs, requests with priority >= 10 verify no drafts (0071) and do not
# decode at all (0072); they continue afterwards. Both need clients that send `priority` and change nothing without it.
export DOCKER_ENV="NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 VLLM_DSPARK_ROUND_MS=33.26,36.33,38.91,42.23,46.03,48.15 \
VLLM_PREFILL_EVERY_N_STEPS=100000 VLLM_PREFILL_THROTTLE_MIN_PRIO=5 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 EXL3_RDMA_DEVICES=rocep1s0f0,roceP2p1s0f0 NCCL_PROTO=^LL128 NCCL_CUMEM_ENABLE=0 \
NCCL_CUMEM_HOST_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_INDEXER_TP_SHARD_MIN_ROWS=2048 EXL3_RDMA_CPU=14 \
EXL3_RDMA_AG=1 EXL3_RDMA_AG_MAX_KB=3072 VLLM_EXL3_HEAD_MROW=1 EXL3_HEAD_MROW_MAX_ROWS=8 \
VLLM_EXL3_MROW_MAX_ROWS=32 EXL3_MROW32_VARIANT=ring EXL3_MROW32_MIN_ROWS=9 \
VLLM_MOE_GATE_FIRST=1 VLLM_EXL3_MOE_SMOE_SHARED=1 \
VLLM_DSV41_MHC_FUSED_POST=2 VLLM_DSV41_MHC_OVERLAP_MAX_TOKENS=24 \
VLLM_DSV41_ADAPTIVE_K=1 VLLM_DSV41_AK_MARGIN=$AK_MARGIN VLLM_DSV41_AK_MIN=$AK_MIN VLLM_DSV41_AK_HI=$AK_HI \
VLLM_DSV41_AK_MAX_REQS=$AK_MAX_REQS \
VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_KV_SWA_FILL=1 \
VLLM_DSV41_INDEXER_DEFER=1 VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63 EXL3_CPU_PIN=5-9,15-19 \
VLLM_DSV41_MARKOV_FUSED=1 VLLM_DSV41_VOR_FUSED=1 VLLM_DSV41_BG_DEPTH=0 VLLM_DSV41_BG_PAUSE_PRIO=10 \
VLLM_DSV41_FG_WAIT_THROTTLE=1 VLLM_DSV41_BG_LPT_SOLO=1 VLLM_DSV41_KV_CRC=1 VLLM_DSV41_OFFLOAD_EXPANDABLE_OK=1"
# The plugin's JIT kernels (TensorFold experts, exl3_mrow, RDMA ring, mHC boundary, router, routed-expert decode, L2
# prefetch) on both nodes before the model loads; once built for this image, this only loads them. Building them next to
# the loaded weights can run a node out of memory. EXL3_SMOE_FLAGS is the launcher's (part of the module name).
JIT="mkdir -p \$HOME/.cache/dsv41-serve && docker run --rm --gpus all -v \$HOME/.cache/dsv41-serve:/root/.cache \
-e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions -e MAX_JOBS=4 -e EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1 \
--entrypoint python3 $IMAGE \
-c 'from vllm_exl3_moe import tf_experts as t, mrow, rdma, mhc, router, smoe, l2pf; t._ext(); mrow.ext(); rdma.ext(); mhc.ext(); router.ext(); smoe.ext(); l2pf.ext()'"
bash -c "$JIT" & JIT_HEAD=$!
ssh "$WORKER" "$JIT" & JIT_WORKER=$!
wait $JIT_HEAD && wait $JIT_WORKER || { echo "JIT prebuild failed" >&2; exit 1; }
SKIP_SERVE_HOLD=1 bash "$RECIPE/scripts/launch/dsv41_tp2.sh" up
