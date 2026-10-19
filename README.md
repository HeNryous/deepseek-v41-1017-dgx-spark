# DeepSeek-V4.1-Flash-1017 on 2× DGX Spark (GB10) — EXL3 2.75 bpw, 320K context

A ready-to-run setup for **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** on
**two DGX Spark / GB10** nodes with tensor parallelism 2: our EXL3 quantization
([HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark),
**KLD 0.0201** against the native 1017 on sudoingX's KLD panel), a complete vLLM serving image, 320K context, image
input, tool calling and DSpark speculative decoding. Every number below was measured on this exact setup.

## Hardware

- 2× GB10 (DGX Spark class), 128 GB unified memory each
- the two nodes connected through their ConnectX-7 port (RoCE); head = rank 0, worker = rank 1
- ~420 GB free disk on **each** node (the model reads its Engram tables from local NVMe)

## Quick start

```bash
# 1) The serving image (linux/arm64, CUDA 13, ~25 GB unpacked) — on both nodes:
docker pull ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v4i

# 2) Model + the two Engram shards into ONE folder (~402 GB). On the head node, then copy to the worker:
git clone https://github.com/HeNryous/deepseek-v41-1017-dgx-spark && cd deepseek-v41-1017-dgx-spark
bash scripts/download_model.sh ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark <worker-host>

# 3) The launcher comes from the sudoingX recipe (v1.1):
git clone https://github.com/sudoingX/dgx-spark-deepseek-v41 ~/dgx-spark-deepseek-v41
git -C ~/dgx-spark-deepseek-v41 checkout cc59c36

# 4) Once per image, on both nodes: build the plugin's JIT kernels BEFORE the first start (building them next to
#    the loaded weights can run the node out of memory and leave a stale lock that blocks every later start):
mkdir -p ~/.cache/dsv41-serve && docker run --rm --gpus all -v $HOME/.cache/dsv41-serve:/root/.cache \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions -e MAX_JOBS=4 --entrypoint python3 \
  ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v4i -c 'from vllm_exl3_moe import tf_experts as t; t._ext()'

# 5) Start on the head node (network devices default to the DGX Spark names; see the script):
HEAD=<head-host> WORKER=<worker-host> bash serve/serve_example.sh
```

**About the Engram shards:** DeepSeek-V4.1 reads two n-gram tables (~95 GiB each) as-is (FP8, not re-quantized) from the base checkpoint.
They are not re-uploaded here; the script takes them straight from `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Without the
script it is just two commands into the same folder:

```bash
hf download HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark --local-dir ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --local-dir ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark
```

> ⚠️ Take them from the **1017** repo. The older `deepseek-ai/DeepSeek-V4.1-Flash` has files with the same names and the
> same sizes but different content.

The server is ready after ~6 minutes (OpenAI-compatible API on port 8000):

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2", "temperature": 1.0, "top_p": 1.0,
  "messages": [{"role": "user", "content": "Write a binary search in Python."}]}'
```

Sampling as recommended by DeepSeek (temperature 1.0, top_p 1.0). Thinking is on by default. The deepseek_v41 reasoning
parser reads the mode only from `chat_template_kwargs`: `{"thinking": false}` or `{"reasoning_effort": "none" | "low" |
"high" | "xhigh" | "max"}`; a top-level `reasoning_effort` does not reach the parser.

## Expected results

Measured on 2× GB10, TP2, GPU clock locked at 2000 MHz on both nodes, no other clients on the server.

**Quality** (sudoingX KLD panel against the native 1017 run through DeepSeek's reference code)

| | KLD ↓ | top-1 ↑ |
|---|---|---|
| this quant (routed experts 2.75 bpw per expert, dense 5 bpw, head 6 bpw) | **0.0201** | **95.96 %** |
| same with dense/head K4 (0xSero's dense layout) | 0.0225 | — |
| 0xSero 1017 EXL3 2.7 bpw (as reported, same panel, 95 % CI 0.0268–0.0343) | 0.0304 | 94.95 % |

Same pipeline as 0xSero's quant; the gap is budget (more bits in the target model, smaller drafter) and calibration
(sudoingX's V4.1-written texts vs native 1017 replies), see the model card.

**Speed**

| | |
|---|---|
| decode, 1 stream (German prose prompt, thinking off, T=1.0, DSpark k5, mean of 3) | ~57 tok/s |
| decode, 3 streams aggregate (same prompt) | ~91 tok/s |
| time to first token, cold 18.5K-token prompt | 8.4 s (~2,200 tok/s) |
| time to first token, cold 314K-token prompt | 152 s (2,066 tok/s); 3 of 3 needles found (10/50/90 % depth, T=0) |
| foreground stream next to 1 / 2 long background prefills | ~29 / ~23 tok/s, first token ≤ 3.8 s |
| KV pool (3.75 GiB per rank) | 989,460 tokens = 3 full 320K requests at once |
| lowest free memory over these runs (incl. the 314K prefill) | ~3.1 GiB head, ~4.2 GiB worker (watchdog stops at 1.5) |

**With Elevatormusic's TensorFold test scripts** (`tests/v41_suite.py` speed group and `tests/v41_bench.py` from
[their repo](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark), unchanged, same
2000 MHz clock; their own TensorFold numbers for comparison, clock not recorded, ~2.4 GHz in their logs):

| | this image | TensorFold (thread 386125) |
|---|---|---|
| decode, prose, 1 stream (suite / bench) | 50.6 / 47.5 tok/s | 56.8 tok/s |
| decode, code, 1 stream (suite / bench) | 82.4 / 88.9 tok/s | 92.2 tok/s |
| prose, aggregate 2 / 3 / 4 streams | 62 / 70 / 79 tok/s | – / 82 / 94 tok/s |
| code, aggregate 2 / 3 / 4 streams | 108 / 126 / 138 tok/s | – / 124 / 140 tok/s |
| prefill, cold 8K / 32K | 2,250 / 2,270 tok/s ¹ | 2,166 / 2,179 tok/s |
| decode with thinking on, prose / code | 46.1 / 63.0 tok/s | |

¹ Measured cold with a fresh `cache_salt` per request. The suite's own 32K run starts with its 8K prompt and hits the
prefix cache (it showed 2,984 tok/s); corrected on 19.10. after a reader-style cross-check.

Not measured: contexts above 320K (TensorFold goes to 1M). Long prompts: 314K at 2,066 tok/s (table above).

## What is in the image

The serving image of the [sudoingX recipe](https://github.com/sudoingX/dgx-spark-deepseek-v41) v1.1 (vLLM pin
`6e517b15` + patches 0001–0021 + the `vllm_exl3_moe` plugin with TensorFold prompt kernels) plus:

| patch | what it does | effect |
|---|---|---|
| 0030 | 1017 reference parity: wo_a FP8 activation quant-dequant; tool rounds kept together in the encoder | matches the 1017 reference |
| 0031 + 0031c | priority-aware prefill throttling: requests with `priority` ≥ `VLLM_PREFILL_THROTTLE_MIN_PRIO` (request field `priority` or header `X-Vllm-Priority`, default 0) count as background; while a foreground request runs, their prefills run only every `VLLM_PREFILL_EVERY_N_STEPS`-th step, capped at `--long-prefill-token-threshold` | foreground stream next to one / two 60K background prefills ~29 / ~23 tok/s, first token ≤ 3.8 s (stock vLLM in an earlier config: ~5 / ~2.6 tok/s); background prefills take longer |
| 0032 | the 0030 quant-dequant as one Triton kernel instead of 9 eager kernels, bit-exact (`VLLM_DSV41_WOA_QDQ_TRITON=0` = old path) | faster; not measured on its own |
| 0033 | RoPE cache only up to max-model-len + 2048; indexer prefill buffer factor 8 instead of 40 (no change to the math) | ~0.5 GiB less per rank at model load |
| 0034 | backport of vLLM #59894: layer 20 (last KV source) writes its KV for all rows, then runs as the first replay layer on the replay rows only (`VLLM_DSV41_REPLAY_TRIM_CUT=0` = old path) | with 0035/0036: 18.5K TTFT 9.04 → 8.42 s, 314K prefill 171 → 152 s; cold-prefill A/B 5/5 identical texts, corruption test 0 hits |
| 0035 | dense prompt path without three copies (group-major fp8 quant output fed straight to the GEMM), bit-exact (40/40) | part of the above |
| 0036 | backport of vLLM #54951 + #61647: indexer prefill rows split over both TP ranks from `VLLM_INDEXER_TP_SHARD_MIN_ROWS` rows on (we use 2048) | part of the above, grows with context |

`docker/Dockerfile` rebuilds the image from a locally built recipe image; `patches/` holds the diffs.

**Rule for 0031:** keep `long-prefill-token-threshold × (max-num-seqs − 1) < max-num-batched-tokens` (we use
max-num-batched-tokens / 4 with 4 seqs). vLLM stops throttling once a request is left waiting after a release step
(`prefill_capacity_bound`). If background chunks can fill a whole step, a new foreground request starves until the
background prefill is done (we measured 150 s). Without priorities from your clients everything is foreground.

## Tips that helped on GB10

- **Remove the kdump crash-kernel reservation** (`crashkernel=` on the kernel command line): +2.1 GiB usable memory per node.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (together with 0033): ~1.3 GiB more free at the 314K peak.
- **Don't upload, hash or copy big files on a node that is serving.** GB10 CUDA allocations don't reclaim page cache; an
  HF upload in a MemoryHigh-capped cgroup still pulled MemAvailable down by ~3.6 GiB and our watchdog had to stop the
  server. O_DIRECT copies (`dd iflag=direct`) to another machine stayed harmless.
- **Pin the RDMA ring's polling thread to an efficiency core** (`EXL3_RDMA_CPU=14`; A725 = cpus 0-4 and 10-14). By default it
  spins on the last X925 core, which then sets the hottest ACPI sensor: with the pin the hottest sensor ran ~10 °C cooler
  (75 → 65 °C / 78 → 68 °C) at the same speed.
- **Measure A/B with a fixed clock** (`nvidia-smi -lgc 0,2000` on both nodes) and with no other clients on the server
  (`Running: 0` in the engine log).

## Tried, and not tried

- DSpark drafter experts at 3 instead of 2 bpw: +1.9 % tokens per step, −1 % steps/s → decode +0.9 %, 3 streams −2.8 %, 1.3 GiB less headroom. Dropped.
- Probabilistic drafting: not tried. Mean p(top-1) of the target at T=1 is 0.78 (prose) / 0.93 (code), so there is little to gain; christopher_owen measured −0.8 to −2.9 % on his multi-node setup.
- THP `never` / proactive compaction off: not A/B-tested; compact_stall rose by only ~120–180 per node over three benches plus a 314K prefill.
- A second NCCL rail: TTFT 3 % shorter, decode ~1 % slower per step (measured with `Simple`; `^LL128` + cuMem off changed nothing measurable). We keep it.

## Note on untrusted text

The stock tokenizer path turns literal DeepSeek special tokens in user text (for example `<｜end▁of▁sentence｜>` in a
quoted e-mail) into the real special token. If you feed untrusted text, insert a zero-width space after `<` for every
`<｜…｜>` and `<|place_holder_mm_span_N|>` before the request reaches vLLM.

## Licenses and credits

Our patches: Apache-2.0 (`LICENSE`). The image contains vLLM (Apache-2.0), the sudoingX recipe and plugin
(Apache-2.0, see `NOTICE`), exllamav3 kernels (MIT), TensorFold kernels (Apache-2.0) and the NVIDIA CUDA base image of
`vllm/vllm-openai` (NVIDIA license terms). The weights are a derivative of DeepSeek-V4.1-Flash-1017 (MIT).
Thanks to DeepSeek (model and reference code); sudoingX (recipe, quantization pipeline, KLD panel, calibration data,
decode kernels, launcher); turboderp (EXL3); ashhart, bertholomus and the TensorFold contributors (RDMA ring, L2 prefetch,
EXL3 prompt kernels); 0xSero; christopher_owen (DSpark experiments, NCCL `^LL128`); jayleaton (special tokens in user
text, `NCCL_CUMEM_ENABLE=0`); see_spot_ruminate (X925 clock cap); shayanx45/Elevatormusic (compaction analysis) and the
DGX Spark forum.
