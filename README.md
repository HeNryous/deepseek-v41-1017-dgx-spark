# DeepSeek-V4.1-Flash-1017 on 2× DGX Spark (GB10): serving image and patches

A complete vLLM serving image (arm64, CUDA 13) for
[HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark),
an EXL3 quantization of DeepSeek-V4.1-Flash-1017 (KLD 0.0201 against the unquantized 1017), running with tensor
parallelism 2 across two DGX Spark / GB10 machines over RoCE: 320K context, image input, tool calling, DSpark
speculative decoding.

```bash
docker pull ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v4h
```

It is the serving image of the [sudoingX recipe](https://github.com/sudoingX/dgx-spark-deepseek-v41) (v1.1: vLLM pin
`6e517b15` + patches 0001–0021 + the `vllm_exl3_moe` plugin with TensorFold prompt kernels) plus the patches in this
repo. `docker/Dockerfile` rebuilds it from a locally built recipe image.

## Patches

| patch | what it does | effect (2× GB10) |
|---|---|---|
| 0030 | 1017 reference parity: wo_a FP8 activation quant-dequant; tool rounds kept together in the encoder | matches the 1017 reference |
| 0031 + 0031c | fair scheduling: requests with priority ≥ `VLLM_PREFILL_THROTTLE_MIN_PRIO` are background; their prefills run only every `VLLM_PREFILL_EVERY_N_STEPS`-th step while a foreground request runs (decode or prefill) | foreground stream next to one 60K background prefill: 5 → ~29 tok/s; new foreground request next to two background prefills: first token after ~3 s |
| 0032 | the 0030 quant-dequant as one Triton kernel instead of 9 eager kernels, bit-exact to the reference act_quant | decode +7 %, 16K TTFT −15 % (with the clock change) |
| 0033 | memory: RoPE cos/sin cache only up to max-model-len (+2048) instead of 1M positions; indexer prefill buffer factor 8 instead of 40. Results unchanged | +0.66 GiB per rank |

**Rule for 0031:** keep `--long-prefill-token-threshold` ≤ `--max-num-batched-tokens` / 4. vLLM stops throttling as soon
as a request is left waiting after a release step (`prefill_capacity_bound`). If background chunks can fill the whole
step budget, a new foreground request then starves until the background prefill is done (we measured 150 s).

## Running

`serve/serve_example.sh` shows our settings on top of the recipe's launcher (`scripts/launch/dsv41_tp2.sh`). In short:
max-model-len 327680, 4 sequences, max-num-batched-tokens 4096, KV cache pinned at 3.75 GiB per rank (989,460 tokens),
`expandable_segments`, indexer logits capped at 128 MB, NCCL over both PCIe functions of the ConnectX port with
`NCCL_PROTO=^LL128`, DSpark k5 with a round-cost table measured on this quant. Sampling as recommended by DeepSeek
(temperature 1.0, top_p 1.0); thinking mode via `chat_template_kwargs`.

Measured (GPU clock fixed at 2000 MHz for comparable runs): decode ~57 tok/s single stream (T=1.0), ~91 tok/s with 3
streams, 16K-token TTFT 9.0 s, 314K-token prefill 170 s (1,845 tok/s), ~4 GiB free memory on the head node at that peak.

## Things we tried and dropped (measured)

- DSpark drafter experts at K3 instead of K2: +1.9 % acceptance but −1 % steps/s; decode +0.9 %, 3 streams −2.8 %,
  1.3 GiB less headroom.
- Probabilistic drafting at T=1: mean p(top-1) was 0.78 (prose) and 0.93 (code). christopher_owen measured −0.8 to −2.9 %.
- Transparent huge pages `never` / no proactive compaction: during serving there is almost no compaction to avoid.
- A second NCCL rail costs ~1 % decode per step (protocol `Simple` or `^LL128` alike) and buys ~3 % prefill; we keep it.

## Note on untrusted text

The stock tokenizer path turns literal DeepSeek special tokens in user text (for example `<｜end▁of▁sentence｜>` in a
quoted e-mail) into the real special token. Our front proxy inserts a zero-width space after `<` for every `<｜…｜>` and
`<|place_holder_mm_span_N|>` in message text before the request reaches vLLM.

## Licenses and credits

Our patches: Apache-2.0 (`LICENSE`). The image contains vLLM (Apache-2.0), the sudoingX recipe and plugin
(Apache-2.0, see `NOTICE`), exllamav3 kernels (MIT), TensorFold kernels (Apache-2.0) and the NVIDIA CUDA base image
of `vllm/vllm-openai` (NVIDIA license terms). The model weights are a derivative of DeepSeek-V4.1-Flash-1017 under the
MIT License. Thanks to sudoingX, turboderp, the TensorFold authors, christopher_owen and the DGX Spark forum.
