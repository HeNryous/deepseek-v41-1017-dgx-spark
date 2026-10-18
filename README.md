# DeepSeek-V4.1-Flash-1017 on 2× DGX Spark (GB10) — EXL3 2.75 bpw, 320K context

A ready-to-run setup for **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** on
**two DGX Spark / GB10** nodes with tensor parallelism 2: our EXL3 quantization
([HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark),
**KLD 0.0201** against the unquantized model), a complete vLLM serving image, 320K context, image input, tool calling
and DSpark speculative decoding. Every number below was measured on this exact setup.

## Hardware

- 2× GB10 (DGX Spark class), 128 GB unified memory each
- the two nodes connected through their ConnectX-7 port (RoCE); head = rank 0, worker = rank 1
- ~420 GB free disk on **each** node (the model reads its Engram tables from local NVMe)

## Quick start

```bash
# 1) The serving image (linux/arm64, CUDA 13, ~25 GB unpacked) — on both nodes:
docker pull ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v4h

# 2) Model + the two Engram shards into ONE folder (~402 GB). On the head node, then copy to the worker:
git clone https://github.com/HeNryous/deepseek-v41-1017-dgx-spark && cd deepseek-v41-1017-dgx-spark
bash scripts/download_model.sh ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark <worker-host>

# 3) The launcher comes from the sudoingX recipe (v1.1):
git clone https://github.com/sudoingX/dgx-spark-deepseek-v41 ~/dgx-spark-deepseek-v41

# 4) Fill in the placeholders (hostnames, fabric interface, RDMA devices, paths) and start on the head node:
nano serve/serve_example.sh && bash serve/serve_example.sh
```

**About the Engram shards:** DeepSeek-V4.1 reads two n-gram tables (~95 GiB each) unquantized from the base checkpoint.
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

Sampling as recommended by DeepSeek (temperature 1.0, top_p 1.0). Thinking is on by default; switch it with
`"chat_template_kwargs": {"thinking": false}` or `"reasoning_effort": "none" | "low" | "high" | "max"`.

## Expected results

Measured on 2× GB10, TP2, GPU clock fixed at 2000 MHz for comparable runs.

**Quality** (KLD panel of the sudoingX recipe against the unquantized 1017)

| | KLD ↓ | top-1 ↑ |
|---|---|---|
| this quant (routed experts 2.75 bpw per expert, dense 5 bpw, head 6 bpw) | **0.0201** | **95.96 %** |
| 0xSero 1017 EXL3 2.7 bpw (as reported, same panel) | 0.0304 | 94.95 % |

**Speed**

| | |
|---|---|
| decode, 1 stream (T=1.0, DSpark k5) | ~57 tok/s |
| decode, 3 streams aggregate | ~91 tok/s |
| time to first token, 16K-token prompt | 9.0 s |
| prefill, 314K-token prompt | 170 s (1,845 tok/s), needle 3/3 |
| foreground stream next to 1 / 2 long background prefills | ~29 / ~23 tok/s, first token ≤ 3.8 s |
| KV pool (3.75 GiB per rank) | 989,460 tokens = 3 full 320K requests at once |
| free memory at the 314K prefill peak | ~4 GiB on the head node |

## What is in the image

The serving image of the [sudoingX recipe](https://github.com/sudoingX/dgx-spark-deepseek-v41) v1.1 (vLLM pin
`6e517b15` + patches 0001–0021 + the `vllm_exl3_moe` plugin with TensorFold prompt kernels) plus:

| patch | what it does | effect |
|---|---|---|
| 0030 | 1017 reference parity: wo_a FP8 activation quant-dequant; tool rounds kept together in the encoder | matches the 1017 reference |
| 0031 + 0031c | fair scheduling: requests with priority ≥ `VLLM_PREFILL_THROTTLE_MIN_PRIO` count as background; their prefills run only every `VLLM_PREFILL_EVERY_N_STEPS`-th step while a foreground request runs | foreground stream next to a 60K background prefill: 5 → ~29 tok/s |
| 0032 | the 0030 quant-dequant as one Triton kernel instead of 9 eager kernels, bit-exact | decode +7 %, 16K TTFT −15 % (with a clock change) |
| 0033 | RoPE cache only up to max-model-len; indexer prefill buffer factor 8 instead of 40 (same results) | +0.66 GiB per rank |

`docker/Dockerfile` rebuilds the image from a locally built recipe image; `patches/` holds the diffs.

**Rule for 0031:** keep `--long-prefill-token-threshold` ≤ `--max-num-batched-tokens` / 4. vLLM stops throttling once a
request is left waiting after a release step (`prefill_capacity_bound`). If background chunks can fill a whole step, a
new foreground request starves until the background prefill is done (we measured 150 s).

## Tips that helped on GB10

- **Remove the kdump crash-kernel reservation** (`crashkernel=` on the kernel command line): +2.1 GiB usable memory per node.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: about +1 GiB headroom at prefill peaks.
- **Page cache is not reclaimable by CUDA on GB10.** Big copies or uploads next to a running server can starve its
  allocations: run them in a memory-capped cgroup (`systemd-run -p MemoryHigh=1G …`).
- **Measure A/B with a fixed clock** (`nvidia-smi -lgc 0,2000` on both nodes), and with no other clients on the server.

## Things we tried and dropped (measured)

- DSpark drafter experts at 3 instead of 2 bpw: +1.9 % acceptance, −1 % steps/s → decode +0.9 %, 3 streams −2.8 %, 1.3 GiB less headroom.
- Probabilistic drafting at T=1: mean p(top-1) 0.78 (prose) / 0.93 (code); christopher_owen measured −0.8 to −2.9 %.
- Transparent huge pages `never` / no proactive compaction: almost no compaction happens while serving.
- A second NCCL rail costs ~1 % decode per step (`Simple` or `^LL128` alike) and buys ~3 % prefill; we keep it.

## Note on untrusted text

The stock tokenizer path turns literal DeepSeek special tokens in user text (for example `<｜end▁of▁sentence｜>` in a
quoted e-mail) into the real special token. If you feed untrusted text, insert a zero-width space after `<` for every
`<｜…｜>` and `<|place_holder_mm_span_N|>` before the request reaches vLLM.

## Licenses and credits

Our patches: Apache-2.0 (`LICENSE`). The image contains vLLM (Apache-2.0), the sudoingX recipe and plugin
(Apache-2.0, see `NOTICE`), exllamav3 kernels (MIT), TensorFold kernels (Apache-2.0) and the NVIDIA CUDA base image of
`vllm/vllm-openai` (NVIDIA license terms). The weights are a derivative of DeepSeek-V4.1-Flash-1017 (MIT).
Thanks to sudoingX, turboderp, the TensorFold authors, christopher_owen and the DGX Spark forum.
