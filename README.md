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
docker pull ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v22

# 2) Model + the two Engram shards into ONE folder (~402 GB). On the head node, then copy to the worker:
git clone https://github.com/HeNryous/deepseek-v41-1017-dgx-spark && cd deepseek-v41-1017-dgx-spark
bash scripts/download_model.sh ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark <worker-host>

# 3) The launcher comes from the sudoingX recipe (v1.1):
git clone https://github.com/sudoingX/dgx-spark-deepseek-v41 ~/dgx-spark-deepseek-v41
git -C ~/dgx-spark-deepseek-v41 checkout cc59c36

# 4) Once per image, on both nodes: build the plugin's JIT kernels (TensorFold experts, exl3_mrow, RDMA ring, mHC boundary,
#    router, routed-expert decode, L2 prefetch) BEFORE the first start (building them next to the loaded weights can run
#    the node out of memory and leave a stale lock that blocks every later start). serve/serve_example.sh runs the same
#    before each start; once built, it only loads them. EXL3_SMOE_FLAGS must match the launcher's: like the source hash,
#    it is part of the module name.
mkdir -p ~/.cache/dsv41-serve && docker run --rm --gpus all -v $HOME/.cache/dsv41-serve:/root/.cache \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions -e MAX_JOBS=4 -e EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1 \
  --entrypoint python3 ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v22 \
  -c 'from vllm_exl3_moe import tf_experts as t, mrow, rdma, mhc, router, smoe, l2pf; t._ext(); mrow.ext(); rdma.ext(); mhc.ext(); router.ext(); smoe.ext(); l2pf.ext()'

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

Measured on 2× GB10, TP2, GPU clock locked on both nodes (2000 MHz unless noted), no other clients on the server.

**Quality** (sudoingX KLD panel against the native 1017 run through DeepSeek's reference code)

| | KLD ↓ | top-1 ↑ |
|---|---|---|
| this quant (routed experts 2.75 bpw per expert, dense 5 bpw, head 6 bpw) | **0.0201** | **95.96 %** |
| same with dense/head K4 (0xSero's dense layout) | 0.0225 | — |
| 0xSero 1017 EXL3 2.7 bpw (as reported, same panel, 95 % CI 0.0268–0.0343) | 0.0304 | 94.95 % |

Same pipeline as 0xSero's quant; the gap is budget (more bits in the target model, smaller drafter) and calibration
(sudoingX's V4.1-written texts vs native 1017 replies), see the model card.

**Decode and prefill** with Elevatormusic's TensorFold test scripts (`tests/v41_suite.py` speed group and
`tests/v41_bench.py` from [their repo](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark),
unchanged), image v5 with the switches `serve/serve_example.sh` had on 20.10. Each cell is the median of three suite
runs or two bench runs (20.10.). The TensorFold numbers are their own from thread 386125; their clock is not recorded
and their logs show ~2.4 GHz:

| tok/s, median | v5, 2000 MHz | v5, 2400 MHz | v12, 2400 MHz ⁵ | TensorFold, ~2.4 GHz | vs TensorFold |
|---|---|---|---|---|---|
| prose, 1 stream (suite) | 54.6 | 56.6 | 56.0 | 56.8 | −0.4 % (v5) / −1.4 % (v12) at 2400 MHz |
| code, 1 stream (suite) | 91.5 | 88.9 | 91.1 | 92.2 | −0.8 % at 2000 / −1.2 % (v12) at 2400 MHz ¹ |
| prose, aggregate 2 / 3 / 4 streams (bench) | 71.9 / 85.3 / 94.4 | – ² | 71.5 / 85.1 / 95.3 | – / 82 / 94 | +3.7 / +1.4 % (v12, 2400 MHz) |
| code, aggregate 2 / 3 / 4 streams (bench) | 112.6 / 136.1 / 155.9 | – ² | 117.0 / 135.4 / 154.0 | – / 124 / 140 | +9.2 / +10.0 % (v12, 2400 MHz) |
| prose / code, 1 stream (bench) | 49.4 / 84.8 | – ² | 51.5 / 86.1 | – | |
| prefill, cold 8K / 32K | 2,203 ³ / 2,270 ⁴ | | 2,470 ³ / – | 2,166 / 2,179 | +1.7 % at 2000 / +14 % at 2400 MHz / – |
| decode with thinking on, prose / code (suite) | 52.8 / 78.8 | 54.6 / 59.2 | 52.2 / 64.6 | | |

¹ Single-stream code at T = 1 moves by up to ±10 % between runs (the three 2400 MHz runs: 88.4 / 88.9 / 94.4).
² At 2400 MHz the repeated bench runs heated the GPUs to 76 °C, and our thermal guard (80 °C GPU / 90 °C board sensor)
capped first one node, then both, to 2000 MHz mid-run; those runs are not a 2400 MHz measurement. One earlier single
run at 2400 MHz, before the guard engaged, gave prose 83.8 / 98.0 and code 134.4 / 161.6 tok/s at 3 / 4 streams.
³ Only the first suite run is cold; the later runs read the 8K prompt from the prefix cache. An earlier cold run the
same morning gave 2,326.
⁴ The suite's 32K run starts with its 8K prompt and hits the prefix cache; 2,270 tok/s is our cold measurement of 19.10.
(image v4i, fresh `cache_salt` per request).
⁵ 21.10. at night, image v12 (= v5 + KV patches, same decode path): before each run both GPUs cooled to ≤ 56 °C, the
clock logged every 5 s; all five runs stayed at ≥ 2,366 MHz without the thermal guard, so this is a clean 2400 MHz
measurement (three suite / two bench runs). Single-stream code at T = 1 again moved by ±5 % between runs.

The scripts sample at T = 1, so single-stream numbers move by several percent between runs. Our production image of
19.10. (v4i + 0040–0046, not published) reached 87.7 (prose) and 143.7 (code) tok/s at 4 streams with the same scripts
at 2000 MHz (single runs). Not measured: contexts above 320K (TensorFold goes to 1M).

Image v12 (21.10.) = v5 + patches 0051–0061, of which only the KV page layout (0057 + 0060) is switched on. Each of
them was A/B-tested against the image before it (cost per decode round within ±0.5 %), so the decode rows above stand
for v12 as well. Since the night of 21.10. we run image v22 = v12 + patches 0064, 0066, 0067, 0071, 0072, 0073 and
0074, with the switches listed in "From v12 to v22" below. v6–v11 and v13–v21 were internal builds and are not
published.

**Long context, fairness and memory** (image v4i unless noted; the later changes are mainly in decode-time paths, the
other rows were not re-measured)

| | |
|---|---|
| time to first token, cold 18.5K-token prompt | 8.4 s (~2,200 tok/s) |
| time to first token, cold 314K-token prompt | 152 s (2,066 tok/s); 3 of 3 needles found (10/50/90 % depth, T=0) |
| foreground stream next to 1 / 2 long background prefills | ~29 / ~23 tok/s, first token ≤ 3.8 s; image v22: 58.5 / 59.1 tok/s, first token 1.3–2.4 s (see "Foreground next to background prefills") |
| KV pool (3.75 GiB per rank), image v12 with 0057 + 0060 | 1,531,258 tokens = 4.67 full 320K requests at once (v5: 989,460) |
| lowest free memory over these runs (incl. the 314K prefill) | ~3.1 GiB head, ~4.2 GiB worker (watchdog stops at 1.5) |

**KV cache capacity** (image v12, patches 0057–0061; 3.75 GiB KV pool per rank, A/B in one window per step, 2000 MHz)

| | v5 | + 0057 paired pages | **+ 0060 three pages (in use)** | 0057 + 0059 NVFP4 records | 0060 + 0061 NVFP4 compressed only |
|---|---|---|---|---|---|
| KV pool, max-model-len 320K (tokens) | 989,460 | 1,346,928 | **1,531,258** | 2,013,868 | 1,837,489 |
| bytes per token, charged for a max-length request | 4,069 | 2,989 | 2,630 | 1,999 | 2,191 |
| bytes per token, live (250K-token prompt, `kv_cache_usage_perc`) | 3,267 | 2,187 | 1,827 | 1,574 | 1,522 |
| cost per verify round, geometric mean | – | −0.3 % | −0.5 % | −0.9 % | −0.7 % (tok/s −2.8 %) |
| KV-KLD against FP8 records, KL / top-1 ¹ | – | format unchanged | format unchanged | 0.028 / 95.0 % | 0.0146 / 96.6 % |
| adopted | | yes (v8) | **yes (v12)** | no (KLD) | no (KLD, tok/s) |

Charged per token (v5): 1,800 B data (1,462.5 B fp8_ds_mla records of the four KV-source layers + 337.5 B FP8 indexer
keys), 1,440 B allocator padding and 829 B sliding-window admission reserve. vLLM packs all KV groups into one block pool
whose stride is the widest group's page sum (138,240 B per 128 tokens for the ratio-2 group); the ratio-1 group paid that
stride for 46,080 B of data. 0057 pairs its pages, 0060 takes three, which is exactly the stride: no padding left. 0057
and 0060: T=0 texts identical 8/8, needles at 250K 6/6, corruption test (8K–100K, 4 each) 0 hits.

¹ Teacher-forced prompt logprobs over 44K positions (top-5). Two identical requests with FP8 records already differ by
KL ~0.008 / top-1 ~97.5 % from position 128 on (bit-identical before), so that is the floor of this measurement.

**What each step brought** (decode tok/s, each A/B in one window, 2000 MHz, T = 1 unless noted, prose / JSON / code
prompts at 1, 3 and 4 streams). Rows 1–2 against the same image without the new switches (v4i behaviour), row 3
against row 2, rows 4–6 against rows 2 + 3. v5 runs rows 2–6.

| | 1 stream | prose 3 / 4 streams | JSON 3 / 4 | code 3 / 4 | geometric mean |
|---|---|---|---|---|---|
| 1. 0045 + 0046 | +1.6 % | – | – | – | +1.0 % |
| 2. 0045 + 0046 + 0040 (variant B) + 0042 | +1…+2 % | +9.4 / +6.4 % | +3.6 / +2.4 % | −0.6 / +1.2 % | +4.0 % |
| 3. recipe switches `VLLM_MOE_GATE_FIRST`, `VLLM_EXL3_MOE_SMOE_SHARED` | T=0: prose −1.5, code +4.6, JSON +9.5 % ¹ | – | – | – | +2.6 % |
| 4. 0042b ring kernel from 9 rows | T=0: prose −0.2, code −2.1 % ¹ | +1.6 / +6.6 % | +5.1 / +3.4 % | +5.0 / +7.7 % | +2.35 % |
| 5. 0042b ring from 9 rows + 0047 (same window as row 4) | T=0: prose +5.1, code −1.9 % ¹ | −0.3 / +9.3 % | +3.4 / +3.3 % | +3.7 / +5.0 % | +2.01 % |
| 6. 0048 + 0050 | 0048 alone: +0.2…+0.8 % | +0.0 / +4.6 % | +1.3 / −0.1 % | +1.7 / +3.9 % | +1.86 % |

¹ These switches change the rounding order, so T = 0 produces a different text; our reading is that these cells
measure the other text, not a change in speed. "–": not recorded for that run. Corruption test (30K and 100K
contexts): 0 hits after steps 2 and 3 and for the v5 configuration as a whole.

**From v12 to v22** (21.10., 2000 MHz unless noted). In steps 1–4 each candidate ran from a fresh start against a
stored reference run of the setup before it (also a fresh start, the same day): 11 decode cells of 256 tokens (prose /
code / JSON at 1 stream with T = 0 and T = 1 and at 4 streams with T = 1, prose and code at 1 stream after a 64K-token
prefix with T = 0); "sustained" is kit_bench set b with 4 streams for 90 s, measured against the previous setup running
in the same window. We gate on the cost per decode round and on sustained throughput: two reference runs gave sustained
118.1 / 117.8 tok/s but 4-stream concurrent 113.4 / 118.5, and tok/s at T = 1 moves with the accepted drafts. Steps 5
and 6 were switched live through the depth file, steps 7 and 8 measured with a fairness bench (both below).

| step (adopted) | cost per round, geometric mean | tok/s, geometric mean | other |
|---|---|---|---|
| 1. 0051 (`VLLM_DSV41_INDEXER_DEFER=1`) + the RDMA ring over both PCIe functions of the ConnectX port (`EXL3_RDMA_DEVICES=rocep1s0f0,roceP2p1s0f0`) | −1.20 % | +1.31 % | 4 streams: prose −4.2 %, JSON −3.2 % per round; T=0 identical; shares of the two not separated |
| 2. k = 5 also at 3–4 streams (0040 `[[1,4,5]]`, up to 24 verify rows) | +1.23 % | +1.27 % | tokens per round +2.37 %; 4 streams: JSON +5.2 %, code +2.6 %, prose +0.7 % tok/s; 1 stream unchanged; T=0 identical |
| 3. 0064 (`VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63`) + `EXL3_CPU_PIN=5-9,15-19` (image v16) | −0.62 / −0.24 % (two runs) | −2.6 / −3.1 % ¹ | sustained 118 → 129 tok/s (+9 %); corruption test 0 hits; shares of the two not separated |
| 4. 0066 `VLLM_DSV41_MARKOV_FUSED=1` + 0067 `VLLM_DSV41_VOR_FUSED=1` (image v18) | −0.66 % | −0.78 % ² | T=0 identical |
| 5. 0071 `VLLM_DSV41_BG_DEPTH=0` (image v19) | – | – | foreground next to background requests +29…+56 % (below) |
| 6. 0072 `VLLM_DSV41_BG_PAUSE_PRIO=10` (image v20) | – | – | foreground next to background requests another +25…+56 % (below) |
| 7. 0031 cadence `VLLM_PREFILL_EVERY_N_STEPS=100000` (was 24) + 0073 `VLLM_DSV41_FG_WAIT_THROTTLE=1` (image v21) | – | – | foreground next to one / two 60K-token background prefills 31.3 / 22.5 → 57.4 / 56.5 tok/s, first token next to two 3.7–3.8 → 1.3–2.4 s; shares of the two not separated (below) |
| 8. 0074 `VLLM_DSV41_BG_LPT_SOLO=1` (image v22) | – | – | first token next to one background prefill 2.1–4.0 → 1.4 s; a 60K-token background prefill alone 92 → 112 s (below) |

0066's `VLLM_DSV41_MARKOV_TRIM=1` measured −0.42 % per round alone against step 3 (tok/s −0.42 % ²), but +0.04 % on
top of step 4; T=0 identical in both; not adopted. Steps 1 and 2 need no new image (0051 is in v12). Not adopted on
21.10.: 0062, 0063, 0065, 0066's trimmed Markov loop and drafter L2 prefetch, a re-measured depth cost table,
all-reduces up to 3 MB on the RDMA ring, the ring's flag-write mode and a higher X925 clock (see "Tried").

¹ Steps with 33–63 rows (short prompts, last prompt chunks) now round as exl3_smoe does: 7 of 8 T=0 texts identical to
before, two runs of the new setup identical to each other. The JSON cell at T=0 wrote another text (tokens per round
−16.7 % at −2.6 % ms per round); that is the tok/s mean.
² The tok/s loss is in the 64K-context cells (step 4: code at 64K −18.5 % at −3.3 % ms per round). At 64K, T=0 texts
are not reproducible (see "How we A/B"), so these cells compare two different texts.

**Foreground next to background requests (0071, 0072)**: one foreground request (`priority` 0, 384 tokens, T = 0)
next to one or two background streams (`priority` 10, long outputs, T = 0), the switches set live through the depth
file, arms interleaved, two runs per cell. First run on the v19 setup at 2190 MHz (0071: cap off / `bg=0` / `bg=1`,
percentages against cap off), second run on v20 (`bg=0` again, then with 0072's `bgpause=10`, percentages against that
`bg=0`):

| foreground tok/s | cap off | `bg=0` | `bg=1` | `bg=0`, 2nd run | `bg=0` + 0072 |
|---|---|---|---|---|---|
| prose + 1 background stream | 31.3 | 42.5 (+36 %) | 38.0 (+21 %) | 42.9 | 53.9 (+26 %) |
| prose + 2 background streams | 22.2 | 34.7 (+56 %) | 30.3 (+36 %) | 34.4 | 53.8 (+56 %) |
| code + 1 background stream | 50.9 | 65.7 (+29 %) | 61.3 (+20 %) | 66.4 | 83.3 (+25 %) |
| code + 2 background streams | 40.2 | 57.6 (+43 %) | 51.3 (+28 %) | 57.5 | 83.2 (+45 %) |

First token 0.22–0.31 s in every arm (0.23–0.29 s with 0072). Capped at `bg=0`, a background stream gets one token per
round, at more rounds per second (one background stream: 15.7 → 22.2); with 0072 it pauses while the foreground request
decodes (0–0.1 rounds with output per second) and keeps its KV blocks and drafts. No errors in either run. Both patches
need clients that send `priority` (our proxy: interactive 0, subagents 5, memory service 10) and do nothing without it.

**Foreground next to background prefills (0031 cadence, 0073, 0074)**, `tests/fair_bench.py`: one foreground stream
(600 tokens, T = 1) next to 0 / 1 / 2 background streams with 60K-token uncached prompts and 400 output tokens
(`priority` 10), three foreground runs per cell; 0071 (`bg=0`) and 0072 on in every row:

| foreground tok/s (first token next to background) | alone | + 1 background | + 2 background |
|---|---|---|---|
| background prefill every 24th step (0031 as before), v20 | 55.9 | 31.3 (3.4–4.9 s) | 22.5 (3.7–3.8 s) |
| every 96th step, v20 | 56.0 | 45.8 (2.9–4.8 s) | 40.1 (2.5–3.7 s) |
| v21: `VLLM_PREFILL_EVERY_N_STEPS=100000` + 0073 | 57.9 | 57.4 (2.1–4.0 s) | 56.5 (1.3–2.4 s) |
| v22: + 0074 | 57.2 | 58.5 (1.4 s) | 59.1 (1.3–2.4 s) |

With the cadence at 100000, background prefill runs only while no foreground request runs; 0073 keeps it out of the
step that admits a waiting foreground request, 0074 keeps its chunks at the long-prefill threshold (1,024 tokens here)
when it runs alone. The price of 0074: a 60K-token background prompt alone prefills in 112 s instead of 92 s (v21).
Corruption test on v22 (30K/100K, 2 each, with and without thinking): 0 hits. Between the v20 rows and the v21 row both
nodes were rebooted: after ~20 restarts that day the head node had only ~1.9 GiB free after a start instead of
~2.7 GiB, and 3.4 GiB after the reboot, so the 100000 row is not from the same boot as the 24 / 96 rows.

**Against TensorFold's Zig build** (kit_bench set b, the same client in every column). Ours: image v16 (= step 3 above),
2200 MHz, no other clients on the engine. [TensorFold](https://github.com/bertholomus/TensorFold) at `629c524a` with
[Mia-AiLab's 1017 EXL3 EAQ 2.9 bpw quant](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw),
built with the [MiaAI-Lab recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) and run on the
same two nodes the night before (2190 MHz, 512K context). Last column: bertholomus's published TP2 numbers for the Zig
build v0.6.0 (Mia's quant of the non-1017 model; we do not know the clock).

| tok/s | this setup (v16) | Zig + Mia 1017, our nodes | Zig v0.6.0, published |
|---|---|---|---|
| code / prose / structured, 1 stream | 99.8 / 55.5 / 139.4 | 97.0 / 59.6 / 145.5 | 103.1 / 64.1 / 143.2 |
| aggregate 2 / 4 streams ¹ | 77.7 / 119.7 | 87.4 / 125.7 | 83.1 / 118.9 |
| 4 streams, sustained 90 s | 129.6 | 136.8 | 142.0 |
| prefill 8K / 32K / 128K | 2,183 / 2,207 / 2,192 | 901 / 1,631 / 1,211 ² | 2,433 / 2,495 / 2,354 |
| decode after a 128K prompt ³ | 118.8 | 27.5 | 100.1 |

¹ kit_bench divides the tokens of all streams by the time of the slowest one; at 2 streams that is the prose stream
(ours per stream: code 66.2, prose 39.6 tok/s).
² kit_bench's prefill uses fresh random words, so the Engram rows are read cold; the head node held only 2–3 GiB of page
cache during the Zig run. Probably NVMe-bound; not re-measured.
³ The continuation of a 128K random-word prompt without EOS. The number depends on whether the continuation falls into
repetition (Zig accepted 4 of 168 drafts in that run), so it is not an engine number, ours neither.

A run on v20 under the same conditions gave the same numbers within noise: code / prose / structured 99.5 / 55.3 /
138.9, aggregate 2 / 4 streams 77.7 / 122.3, sustained 129.1, prefill 8K / 32K / 128K 2,165 / 2,164 / 2,149 (decode
after a 128K prompt 73.7, see ³). kit_bench sends no `priority`, so 0071–0074 do not act there.

## What is in the image

The serving image of the [sudoingX recipe](https://github.com/sudoingX/dgx-spark-deepseek-v41) v1.1 (vLLM pin
`6e517b15` + patches 0001–0021 + the `vllm_exl3_moe` plugin with TensorFold prompt kernels) plus:

| patch | what it does | effect |
|---|---|---|
| 0022 | from the recipe's `patches/experimental/`, unchanged: DSpark verify rows through the plugin's Triton sparse attention (`VLLM_DSV41_TRITON_VERIFY=1`) | off; not measured here |
| 0030 | 1017 reference parity: wo_a FP8 activation quant-dequant; tool rounds kept together in the encoder | matches the 1017 reference |
| 0031 + 0031c | priority-aware prefill throttling: requests with `priority` ≥ `VLLM_PREFILL_THROTTLE_MIN_PRIO` (request field `priority` or header `X-Vllm-Priority`, default 0) count as background; while a foreground request runs, their prefills run only every `VLLM_PREFILL_EVERY_N_STEPS`-th step, capped at `--long-prefill-token-threshold` | foreground stream next to one / two 60K background prefills ~29 / ~23 tok/s, first token ≤ 3.8 s (stock vLLM in an earlier config: ~5 / ~2.6 tok/s); background prefills take longer. We ran every 24th step until 21.10., since then 100000 (background prefill only while no foreground request runs; with 0073 + 0074: 58.5 / 59.1 tok/s, first token 1.3–2.4 s) |
| 0032 | the 0030 quant-dequant as one Triton kernel instead of 9 eager kernels, bit-exact (`VLLM_DSV41_WOA_QDQ_TRITON=0` = old path) | faster; not measured on its own |
| 0033 | RoPE cache only up to max-model-len + 2048; indexer prefill buffer factor 8 instead of 40 (no change to the math) | ~0.5 GiB less per rank at model load |
| 0034 | backport of vLLM #59894: layer 20 (last KV source) writes its KV for all rows, then runs as the first replay layer on the replay rows only (`VLLM_DSV41_REPLAY_TRIM_CUT=0` = old path) | with 0035/0036: 18.5K TTFT 9.04 → 8.42 s, 314K prefill 171 → 152 s; cold-prefill A/B 5/5 identical texts, corruption test 0 hits |
| 0035 | dense prompt path without three copies (group-major fp8 quant output fed straight to the GEMM), bit-exact (40/40) | part of the above |
| 0036 | backport of vLLM #54951 + #61647: indexer prefill rows split over both TP ranks from `VLLM_INDEXER_TP_SHARD_MIN_ROWS` rows on (we use 2048) | part of the above, grows with context |
| 0040 | configuration only ([patches/0040](patches/0040-dsv41-dynamic-sd-config.md)): vLLM's dynamic speculation `num_speculative_tokens_per_batch_size`; until 21.10. `[[1,2,5],[3,4,4]]` (DSpark k = 5 at 1–2 streams, k = 4 at 3–4 streams, at most 20 verify rows), since then `[[1,4,5]]` (k = 5 at 1–4 streams, at most 24 rows) | see steps 2 and 4–5 above; `[[1,4,5]]`: step 2 of "From v12 to v22" |
| 0041 | joint DSpark depth choice for all requests of a step (`confb:` / `confr:` policy strings, `cal=1`, `ctx=`), live via the depth file | off by default; did not beat the stock rule `conf` (see "Tried") |
| 0042 / 0042b | `exl3_mrow` for 17–32 verify rows (`VLLM_EXL3_MROW_MAX_ROWS=32`) instead of exllamav3's GEMM. 0042b, which is in the image, rewrites the 17–32-row kernel (`EXL3_MROW32_VARIANT`, default `ring`); `EXL3_MROW32_MIN_ROWS=9` sends 9–16 rows to it as well | dense time per round at 18 / 24 rows: 20.7 / 21.0 ms before, 18.4 / 19.6 with 0042, 13.2 / 13.7 with 0042b (micro-benchmark); ring from 9 rows: step 4 |
| 0043 | prompt-lookup drafts layered on DSpark (Triton suffix match; replaces drafts only where its measured acceptance beats the confidence head), `lookup=` | off by default; +0.6…+4.4 % on copy-heavy prompts at T=1, −1…−3 % elsewhere |
| 0045 | the decode round's TP all-gathers (DSpark Markov pairs, Engram rows, logits) over the plugin's RDMA ring instead of NCCL (`EXL3_RDMA_AG=1`, `EXL3_RDMA_AG_MAX_KB=3072`); a pure copy, bit-exact | step 1 (with 0046) |
| 0046 | a row limit of its own for the LM head on `exl3_mrow` (`VLLM_EXL3_HEAD_MROW=1 EXL3_HEAD_MROW_MAX_ROWS=8`): 1 stream on mrow, from 2 streams on exllamav3 as before | head pass at 1 / 6 rows 1,113 / 1,160 → 994 / 1,052 µs (micro-benchmark) |
| 0047 | adaptive verify window with two regimes: for steps with ≤ `VLLM_DSV41_AK_MAX_REQS` requests the scheduler sets k from the requests' accepted drafts (EMA + `VLLM_DSV41_AK_MARGIN`); requests whose EMA reaches `VLLM_DSV41_AK_HI` keep the full window (`VLLM_DSV41_ADAPTIVE_K=1`) | step 5: prose at 1 and 4 streams up, code and JSON at 3–4 streams 1–3 % below step 4 |
| 0048 | mHC sublayer boundary (post + input collapse + RMSNorm) in one launch, bit-identical to the two TileLang kernels (`VLLM_DSV41_MHC_FUSED_POST=2`) | 6.4 → 3.6 µs per boundary in the decode graph; T=0 texts identical (8/8); step 6 |
| 0049 | router GEMV prefetch depth and x staging as build flags (`EXL3_ROUTER_FLAGS`) | no faster build found; we run the default (= as before) |
| 0050 | mHC overlap path up to 24 rows (`VLLM_DSV41_MHC_OVERLAP_MAX_TOKENS=24`), so 4-stream windows (20 rows) take it too | step 6 (4-stream cells) |
| 0051 | the indexer's `weights_proj` GEMV (hidden → 32 heads, 56–61 µs on GB10) no longer makes the main stream wait right after the fused QKV GEMM: the join moves to the first read, and with short contexts, where every candidate is selected and the weights are never read, the GEMV is skipped (`VLLM_DSV41_INDEXER_DEFER=1`); bit-identical | **in use** since 21.10., measured together with the RDMA ring over both PCIe functions (cost per round −1.20 %, step 1 of "From v12 to v22"); on top of 0054 alone it measured +0.2 % (within noise) |
| 0053 | KV cache backed by the GB10 display reservation (1,792 MiB per node, DRM dumb buffer + `cuMemHostRegister`) with pinned ordinary memory behind it (`VLLM_DSV41_DISPLAY_KV_MIB`) | off: KV pool +47 %, but ~1 GB less free memory under load and +1.5 % per round (see "Tried") |
| 0054 | token embedding (631 MiB per rank) and the DSpark Markov cache (505 MiB) in the display reservation, copied bit-exact (`VLLM_DSV41_DISPLAY_TENSORS=1`) | off: decode −1.7 % (geometric mean), 64K code −6.7 % |
| 0055 | draft mode per request from the 0047 regime: requests that plan the full window draft probabilistically, the others greedy with a one-hot draft distribution, so their verification stays exact (`VLLM_DSV41_HYBRID_DRAFT=1`) | off: +0.75 % at T=1 (code at 1 stream +5.0 %), but prose at 1 stream −1.6 % |
| 0056 | CRC-32 sidecar for every block of vLLM's file-system KV offload tier, checked after each load; a bad block is deleted and recomputed (`VLLM_DSV41_KV_CRC=1`) | off; live (21.10., image v12, with 0058): reload of a 57K-token prompt from the file-system tier after a restart 32.3 → 3.6 s, a corrupted block detected and recomputed, decode −0.3 %, corruption test 0 hits; not adopted because the 1 GiB CPU tier left the head node at 1.9 GiB free under load (our gate: 2.0) |
| 0057 | paired KV pages: the ratio-1 compressed-KV group takes 128-token blocks made of two 64-state kernel pages and stops paying the ratio-2 group's block stride (`VLLM_DSV41_KV_PAIR=1`; `VLLM_DSV41_KV_SWA_FILL=1` fills the sliding-window groups up to the stride) | KV pool 989,460 → 1,346,928 tokens (+36 %), live 3,267 → 2,187 B per token, cost per round −0.3 %, T=0 identical 8/8, corruption test 0 hits |
| 0058 | vLLM's OffloadingConnector next to `expandable_segments:True` (`VLLM_DSV41_OFFLOAD_EXPANDABLE_OK=1`) | needed for 0056; works (see 0056) |
| 0059 | NVFP4 KV records on GB10: FlashInfer's 384 B SM120 record (e2m1 with an e4m3 scale per 16, RoPE in BF16) for the window and the compressed KV, written with FlashInfer's own quantize-append; SWA indices padded to 512 for DSpark's 192-wide non-causal verify (`VLLM_DSV41_KV_NVFP4=1`) | off: KV pool +49.5 %, but KV-KLD 0.028 / top-1 95.0 % against FP8 records (see "KV cache capacity") |
| 0060 | three 64-state kernel pages per manager block for the ratio-1 KV group: 3 × 46,080 B = the ratio-2 block stride, no padding left (`VLLM_DSV41_KV_PAIR=3`, on top of 0057) | **in use:** KV pool 1,346,928 → 1,531,258 tokens (+13.7 %), live 2,187 → 1,827 B per token, cost per round −0.5 %, T=0 identical 8/8, corruption test 0 hits |
| 0061 | FP8 sliding window + NVFP4 compressed KV (as vLLM's SM100 format does): two FlashInfer calls with `return_lse`, merged in Triton (base-2 LSE, sink in the merge) (`VLLM_DSV41_KV_NVFP4=2`) | off: KV pool 1,837,489 with 0060, but KV-KLD 0.0146 / top-1 96.6 % and tok/s −2.8 % (fewer drafts accepted) |
| 0062 | the DSpark verify step's sampling phase on the live rows only: the LM head (where it runs on exl3_mrow), the logits all-gather and the rejection sampler skip the rows past each request's depth, same tokens by construction (`VLLM_DSV41_EXACT_VERIFY=1`; `count` only logs the dead rows per round) | off, not in the image: expected 0.03–0.25 % per round, not measured; a forward of exact width does not fit this stack (see "Tried") |
| 0063 | the dense L2 prefetch also for 17–32-row verify windows (`EXL3_L2PF_MAX_ROWS=32`) | off, not in the image: tok/s −2.2 %, at 4 streams JSON −6.2 % and code −3.0 % |
| 0064 | the routed experts of 33–63-row calls (mixed steps under load, short prompts, last prompt chunks) as exl3_smoe row chunks of ≤ 42 rows, instead of exllamav3's prefill tiers with a host read of the expert counts per layer (`VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63`) | **in use** (with `EXL3_CPU_PIN`): sustained 118 → 129 tok/s (+9 %), cost per round −0.24 %, corruption test 0 hits; 33–63-row steps round as exl3_smoe does (deterministic) |
| 0065 | mixed prefill + decode steps of up to 32 tokens (40 with the coop limit raised) as breakable PIECEWISE CUDA graphs next to the FULL decode graphs (`cudagraph_mode` FULL_AND_PIECEWISE, `VLLM_DSV41_PW_SIZES`, `VLLM_DSV41_PW_TWINS`, `VLLM_DSV41_PW_MAX_ROWS`) | off, not in the image: sustained +0.0 % after 0064, first token of short arrivals −30 ms, 446 MiB less free on the head node |
| 0066 | DSpark drafter: each greedy Markov step in three kernels and one all-gather instead of ~13 kernels, same drafts (`VLLM_DSV41_MARKOV_FUSED=1`); the Markov loop only as deep as the next step verifies (`VLLM_DSV41_MARKOV_TRIM=1`); L2 prefetch of the drafter's dense weights (`VLLM_DSV41_DRAFT_L2PF=next`) | `MARKOV_FUSED` **in use** (with 0067: cost per round −0.66 %, T=0 identical); `MARKOV_TRIM` −0.42 % alone but +0.04 % on top of `MARKOV_FUSED`, off; `DRAFT_L2PF` −0.01 %, off |
| 0067 | the eager phase between the drafter graph and the target graph: the decode SWA metadata of all 15 sliding-window groups in one launch instead of 45, the `replay_start` upload only when it changes (`VLLM_DSV41_VOR_FUSED=1`); same bits | **in use**, measured together with 0066's `MARKOV_FUSED` |
| 0071 | background verify-depth cap: while a foreground request decodes in the step, requests with `priority` ≥ `VLLM_DSV41_BG_MIN_PRIO` (default 10) verify at most `VLLM_DSV41_BG_DEPTH` drafts, switchable live through the depth file (`bg=`, `bgprio=`); outputs unchanged | **in use** (`VLLM_DSV41_BG_DEPTH=0`): foreground decode next to one / two background streams +29…+56 %, first token unchanged |
| 0072 | background decode pause: while a request with `priority` < `VLLM_DSV41_BG_PAUSE_PRIO` runs, the scheduler skips decoding requests at or above it; they keep their KV blocks and drafts and continue once no foreground request runs; prefill chunks stay with 0031; live through the depth file (`bgpause=`); outputs unchanged | **in use** (`VLLM_DSV41_BG_PAUSE_PRIO=10`): foreground decode next to one / two background streams +25…+56 % on top of 0071 `bg=0` (prose 34.4 → 53.8, code 57.5 → 83.2 tok/s next to two), first token unchanged |
| 0073 | the 0031 prefill throttle also holds while a foreground request (`priority` below `VLLM_PREFILL_THROTTLE_MIN_PRIO`) waits, so the step that admits it carries no background chunk (`VLLM_DSV41_FG_WAIT_THROTTLE=1`) | **in use**, measured together with the 0031 cadence 100000: foreground next to two 60K-token background prefills 22.5 → 56.5 tok/s, first token 3.7–3.8 → 1.3–2.4 s (shares not separated) |
| 0074 | background prefill chunks keep `--long-prefill-token-threshold` also when they are the only eligible request (vLLM drops it then); a foreground request alone still gets the whole budget (`VLLM_DSV41_BG_LPT_SOLO=1`) | **in use**: first token next to one background prefill 2.1–4.0 → 1.4 s; a 60K-token background prompt alone prefills in 112 instead of 92 s |
| recipe switches | `VLLM_MOE_GATE_FIRST=1` (recipe patch 0018: the router GEMM before the shared-expert fork) and `VLLM_EXL3_MOE_SMOE_SHARED=1` (the shared expert inside the routed-expert kernel as one more pick per token, instead of its own launches); since 21.10. also `EXL3_CPU_PIN=5-9,15-19` (the plugin pins every thread of the vLLM processes to the Cortex-X925 cores) | step 3; `EXL3_CPU_PIN`: with 0064 in step 3 of "From v12 to v22" |

All new switches are off by default; without them the image behaves like v4i. `serve/serve_example.sh` sets the ones we
use. `docker/Dockerfile` rebuilds the image from a locally built recipe image; `patches/` holds the diffs, each with
what, why, measurements and switches in its header.

**Rule for 0031:** keep `long-prefill-token-threshold × (max-num-seqs − 1) < max-num-batched-tokens` (we use
max-num-batched-tokens / 4 with 4 seqs). vLLM stops throttling once a request is left waiting after a release step
(`prefill_capacity_bound`). If background chunks can fill a whole step, a new foreground request starves until the
background prefill is done (we measured 150 s). Without priorities from your clients everything is foreground. Since
19.10. we also start vLLM with `--scheduling-policy priority`, so its waiting queue is ordered by the same `priority`
field (lower first). Since 21.10. we run `VLLM_PREFILL_EVERY_N_STEPS=100000` instead of 24, so background prefill runs
only while no foreground request runs, together with 0073 (no background chunk in the step that admits a waiting
foreground request) and 0074 (background chunks keep the threshold when they run alone); the rule above still sets
how much of a step's budget background chunks can take.

## Where the time goes

**How we A/B.** All variants of a comparison in one window at a locked clock (2000 MHz unless noted), no other clients
(`Running: 0`), T = 1 with a fixed seed, prose / code / JSON prompts at 1, 3 and 4 streams. Variants that need a restart
run one after the other, live-switchable ones (depth policy) interleaved as A/B/…/B/A. Gates are written down before the
run (no cell below −1.5 %, 1 stream not worse on average; ranking by the geometric mean over all cells), then a
corruption test on the winner. Single-stream cells at T = 1 move by several percent between runs because a different
sample writes a different text; for those we compare tokens per round or use T = 0. T = 0 texts are only reproducible for
short contexts: from position 128 on (the first ratio-128 compressed entries) two identical requests to the same
process differ by KL ~0.008 / top-1 ~97.5 % in their prompt logprobs, with the change of the actual token's NLL
centred on zero. Above that we compare logprobs against this floor, not texts.

**Round cost by verify depth** (all requests at the same fixed depth, short context, ms per decode round, before
0040–0050):

| streams | depth 0 | depth 5 | per live row ¹ |
|---|---|---|---|
| 1 | 32.3 | 48.1 | 3.2 |
| 2 | 44.5 | 70.3 | 2.6 |
| 3 | 59.1 | 94.6 | 2.4 |
| 4 | 68.8 | 109.2 | 2.0 |

¹ (depth 5 − depth 0) / (5 × streams).

Now a decode round costs 24.8 ms fixed + 3.6 ms per request + 2.8 ms per live verify row (fit over fixed depths at
1–4 streams, profile of 21.10., image v19, 2000 MHz); the live-row part is almost entirely routed experts. On 19.10.
(the table above) a fit over all cells gave ~9.6 ms of fixed cost per request even at depth 0 (+1.6 ms at 32K context):
each request paid for its whole verify window (k + 1 rows through the dense layers and attention; rows past the depth
only skip the routed experts) and for its drafter rows, while a live row next to other streams was cheap. That is what
0040 (fewer drafts per stream at 3–4 streams) and 0047 (narrower windows for requests that accept little) act on, and
why verifying fewer drafts when more streams ran did not help (0041, see "Tried").

**Profile, 1 stream** (38.9 ms per round, before 0045–0050; kernel time per group, the groups overlap): MoE + router
25.3 ms · dense 11.8 · mHC / norm / Engram 5.4 · attention + indexer 4.9 · DSpark drafter 3.6 · communication 2.4 (of
which NCCL, 10 calls: 0.51) · sampler 1.6 · idle / host 0.4. The round is GPU-bound.

- **Routed experts** are bandwidth-bound: the expert kernels stream at ~225 GB/s, and their 24.2 ms of kernel time
  overlap to ~15.4 ms of wall time per round. By live rows 1 / 2 / 3 / 4 / 6 / 12 / 20 the routed MoE costs 9.1 / 12.1 /
  15.4 / 19.4 / 25.5 / 47.7 / 78.8 ms per round (micro-benchmark); deeper pipelining or folded stages changed this by
  −1 ms at 1 row and by ±2 % elsewhere.
- **Dense layers above 16 verify rows** (micro-benchmark, one rank, the 43 dense layers incl. the drafter, ms per round):

  | rows | 16 | 18 | 20 | 24 |
  |---|---|---|---|---|
  | exllamav3 GEMM above 16 rows (before 0042) | 15.1 | 20.7 | – | 21.0 |
  | 0042 | 15.0 | 18.4 | – | 19.6 |
  | 0042b `ring` (separate run, in which 0042 measured 14.8 / 18.2 / 18.4 / 19.0) | 14.8 (12.7 through ring) | 13.2 | 13.3 | 13.7 |

  Ring from 9 rows on was measured twice in the serve: +2.7 % and +2.35 % geometric mean, most 2–4-stream cells
  between +3 and +8 % (step 4 is the second run); ring for 17–32 rows only: +0.2 %.
- **mHC boundaries** (80 per round): two TileLang kernels on the main stream at each overlapped boundary; 0048 does the
  same work in one launch with the same bits (6.4 → 3.6 µs per boundary in the decode graph). At 4 streams (20 rows)
  the overlap path was not taken (its bound was 16 rows) and mHC cost 3.35 ms per round on the main stream; 0050 raises
  the bound to 24.
- **Sampler phase** (1.6 ms): mostly bytes, the K6 LM head (248 MB per rank) takes ~1.1 ms; by our estimate, fusing the
  small kernels around it would save ≤ 0.1 ms.

**GPU clock** (against 2000 MHz): 2200 / 2300 / 2400 MHz shorten a decode round by 1.1 / 2.0 / 2.2 % and the time to
first token of an 18.5K prompt by 9 / 11 / 12 %. At 2400 MHz the GPU peaked at 76 °C (hottest ACPI sensor 83 °C) at a
median GPU power of 59 W (47 W at 2200 MHz). Deep CPU idle states off (LPI-2/3): −0.09 %, no effect.

**Per round against TensorFold** (greedy, kit_bench set b; TensorFold from Elevatormusic's run logs of 18.10., ~2.4 GHz;
ours from a probe on 17.10. with the image before 0040–0050):

| | tokens per round, TensorFold / this setup | ms per round, TensorFold / this setup |
|---|---|---|
| code | 4.34 / 4.51 | 43.9 / 46.6 |
| prose | 1.92 / 2.13 | 33.4 / 40.5 |
| structured | 5.88 / 5.98 | 42.2 / 45.6 |

More of our drafts are accepted, but TensorFold's rounds are cheaper; the clock explains at most ~2 % of that. Our
reading, not measured directly: TensorFold's rounds are flat and sized to the drafts it verifies, while our fixed window
carries rows past the accepted depth.

**Window width.** A fixed narrower window at one stream (k = 4 / 3 / 2 instead of 5) decoded prose +1.3 / +4.0 / +5.7 %
faster (T = 0) at the same 2.0 tokens per round, while code lost 3 / 7 / 18 % and JSON 6 / 9 / 19 %. 0047 sizes the
window per step from each request's accepted drafts. With one regime (the window always follows the average) prose
gained 7.7 % at 1 stream (T = 0), but code lost 3.8 % (T = 0) and JSON 10.6 % (T = 1). With a threshold of 2.0
accepted drafts, above which a request keeps the full window, code and JSON held at T = 0, but JSON at T = 1 still fell
from 3.98 to 3.49 tokens per round. At 1.6, JSON keeps 4.05 of 4.11 tokens per round and prose gains 5.1 % at 1 stream
(T = 0) and 9.3 % at 4 streams (step 5).

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

- **Batch-aware depth à la TensorFold (0041 `confb`), confidence calibration for T > 0 (`cal=1`) and prompt-lookup drafts
  (0043): all in the image, all off by default, none beat the stock per-request rule** (measured with
  Elevatormusic-style prompts at T=1, 1/3/4 streams, 4 runs per cell, interleaved A/B/C/D-D/C/B/A at 2000 MHz):

  | vs the stock depth rule `conf` | 1 stream | 3 / 4 streams prose | 3 / 4 streams code | copy-heavy prompts |
  |---|---|---|---|---|
  | `confb` (one table of total rows, extrapolated from 1 stream) | ±0.2 % | −1.9 / −2.3 % | −1.9 / −3.0 % | – |
  | `confb` + `cal=1` | −0.5…−1.0 % | −2.0 / −5.4 % | −2.1 / −2.0 % | – |
  | + prompt lookup (`lookup=3,8,0.5`) | −1.5…−3.4 % | −1.7 / −3.7 % | −5.1 / −6.5 % | +0.6…+4.4 % (vs. without) |

  Why (19.10.): with a fixed verify window (k+1 rows per request, rows past the depth only skip the routed experts) live
  rows were *cheap* next to other streams, and every request paid a fixed cost for its whole window (see "Where the time
  goes"; the costs per request and per live row have changed since).
  TensorFold's rule "verify fewer drafts when more streams run" fits its variable-size rounds, not this window;
  calibrating the confidence head for T=1 also made rounds shallower and lost. Prompt lookup is capped by the window: at
  T=1 DSpark already reaches 5.4–5.7 of the maximum 6 tokens per round on code edits and JSON rewrites, and the extra
  search/apply per step costs 1–3 % elsewhere.
- **Fewer drafts at 3–4 streams without 0042:** 0040 variant A (k = 4 at 3 and k = 3 at 4 streams, never above 16 rows):
  prose 3 / 4 streams +8.1 / +7.8 %, but code at 4 streams −5.3 %. **0042 alone** (k = 5 everywhere, 18 / 24 rows on
  `exl3_mrow`): prose 3 / 4 streams +3.9 / +2.5 %, below our gate.
- **0047 with one regime, or with a threshold of 2.0:** lost code and JSON (see "Window width"). We use 1.6.
- **A narrower 0047 window for prose** (margin 0.5 instead of 1.0, or additionally a 1-draft floor): prose at 1 stream
  T=0 +2.4 %, but code at T=1 fell from 3.70 to 3.57 tokens per round and the geometric mean moved only +0.1 / +0.7 %.
  We keep margin 1.0.
- **Ring kernel for 17–32 rows only:** +0.2 % (geometric mean). From 9 rows on it pays (step 4).
- **Routed-expert kernel variants** (deeper pipelining, folded stages; bit-identical): bandwidth-bound, −1 ms at 1 live
  row, ±2 % elsewhere. Not adopted.
- **Router prefetch depth / x staging (0049):** no flag combination built faster than the default.
- **Triton verify attention (0022, from the recipe):** in the image, off. Measured on v5: 1 stream ms per round −0.1 %,
  3–4 streams −2.1 % (geometric mean), and its accumulation order moved the oracle (24 prompts: top-1 99.36 %, truncated
  KL mean 0.0121, p99 0.20; 8 of 24 greedy continuations identical). Not adopted.
- **Higher GPU clock:** 2400 MHz saves 2.2 % per decode round (see "Where the time goes"); all our A/B numbers are at
  2000 MHz.
- **Deep CPU idle states off** (LPI-2/3, aimed at the wake-up latency of NCCL's proxy thread): −0.09 %, no effect.
- **The RDMA ring over both PCIe functions of the ConnectX port** (`EXL3_RDMA_DEVICES` with two devices, to halve the
  wire time of the logits gather in 0045): in use since 21.10., measured together with 0051 (step 1 of "From v12 to
  v22").
- DSpark drafter experts at 3 instead of 2 bpw: +1.9 % tokens per step, −1 % steps/s → decode +0.9 %, 3 streams −2.8 %, 1.3 GiB less headroom. Dropped.
- **Probabilistic drafting** (`draft_sample_method=probabilistic`, measured on v5 at 2000 MHz): tokens per round at T=1
  +0.8 % (prose 1.88 vs 1.93 at 1 stream), but the full-vocabulary draft path costs ~1 ms per round: T=1 −2.8 %, T=0
  −3 %. With block verification on top (`rejection_sample_method=block`): tokens per round +3.95 % at T=1 (code at
  1 stream 3.70 → 4.05, JSON 3.86 → 4.05, prose unchanged), decode −0.1 % because of the same cost. With greedy drafts,
  block verification is provably the same as token-wise checking (P(prefix i accepted) = Π p(x_j) either way).
  Note: in August 2026, probabilistic drafting in an older vLLM build corrupted outputs above ~26.5K context at T > 0;
  our corruption test (8K–100K at T = 1) would be the gate before using it.
- THP `never` / proactive compaction off: not A/B-tested; compact_stall rose by only ~120–180 per node over three benches plus a 314K prefill.
- A second NCCL rail: TTFT 3 % shorter, decode ~1 % slower per step (measured with `Simple`; `^LL128` + cuMem off changed nothing measurable). We keep it.
- **The display reservation as memory (0053, 0054).** GB10 reserves 1,792 MiB per node for the display; with
  `nvidia_drm modeset=1` a DRM dumb buffer maps it and `cuMemHostRegister` makes it device-addressable (the technique of
  Gilesco's patch 0072, Apache-2.0, which 0rand's fork uses). A probe read it at 162 GB/s (ordinary memory 244), scattered 584 B KV records at 12 GB/s (79). KV cache
  fully in the display region (1,792 MiB): decode −0.5…+0.3 % at 1 stream, prefill −1.2 %. KV cache over display +
  pinned ordinary memory (0053, same 3.75 GiB of ordinary memory): pool 1,346,928 → 1,975,467 tokens, T=0 identical,
  corruption test 0 hits, but the lowest free memory under load fell from 3,011 to 2,013 MiB on the head node and the
  cost per round rose 1.5 % (multi-stream cells +1.1…+7.1 %). Embedding + Markov cache in the display region (0054, so
  the KV pool can grow by 1 GiB): decode −1.7 %, 64K code −6.7 %. Not adopted.
- **Hybrid drafting (0055)**, probabilistic drafts only for requests that plan the full window, greedy for the rest:
  +0.75 % at T=1 overall and +5 % for code at 1 stream, but prose at 1 stream −1.6 %. Not adopted.
- **1M context:** max-model-len 1,048,576 starts (pool 1,664,407 tokens with 0057), but the indexers' shared K gather
  buffer (max-model-len × `VLLM_INDEXER_PREFILL_BUFFER_FACTOR` × 132 B) and the longer RoPE cache leave the head node
  with 1.3–1.9 GiB free under the first long prompts, below our 1.5 GiB watchdog, also with buffer factor 2 instead of 8.
  We stay at 320K.
- **NVFP4 KV (0059, 0061):** see "KV cache capacity". Keeping the sliding window in FP8 halves the KLD of NVFP4
  everywhere, but the rest is still above the run-to-run floor and costs accepted drafts.
- **An exact verify width per round** (verify only the drafts the depth policy keeps, the way TensorFold sizes its
  rounds): does not fit this stack. The host would need the round's depth before it launches the target graph, but the
  V4.1 attention metadata is planned on the host from the query lengths, the V4.1 indexer accepts a device/host
  query-length mismatch only on sm90/sm100, and re-planning after a sync would, by our estimate, cost ≥ 1 ms of host
  time per round, more than the dead rows cost at 1 stream (~0.2–0.25 ms each). 0062 trims only the sampling phase
  (expected 0.03–0.25 %, not measured). A re-measured round-cost table for the depth policy (base cost −13 %) chose
  shallower rounds but saved no time: tok/s −0.20 %, prose at T = 0 −2.7 % per round at −5.0 % tokens per round.
- **L2 prefetch up to 32 rows (0063):** tok/s −2.2 %, at 4 streams JSON −6.2 % and code −3.0 % (cost per round +8.1 /
  +5.4 %). Our reading: at 17–32 rows the prefetch competes with the routed experts' DRAM reads. Not adopted.
- **CUDA graphs for mixed prefill + decode steps (0065):** after 0064 a mixed step costs the running streams only
  20–66 ms; PIECEWISE graphs up to 40 rows changed sustained throughput by +0.0 % and the first token of short arrivals
  by −30 ms, and took 446 MiB of free memory on the head node. Not adopted.
- **Drafter L2 prefetch and the trimmed Markov loop (0066):** `VLLM_DSV41_DRAFT_L2PF=next` −0.01 % per round;
  `VLLM_DSV41_MARKOV_TRIM=1` −0.42 % alone, but +0.04 % on top of `MARKOV_FUSED` + 0067. All three 0066 switches
  together with 0067 in one earlier window: +1.32 % per round (1-stream cells +1.7…+2.6 %), although each switch alone
  measured ≤ 0; not resolved.
- **All-reduces up to 3 MB on the RDMA ring** (`EXL3_RDMA_AR_MAX_KB=3072`): cost per round +0.41 %, sustained −0.3 %.
  Together with 0063 and `VLLM_EXL3_MOE_COOP_MAX_TOKENS=42`: tok/s −1.65 %, sustained −0.4 %.
- **The RDMA ring's flag-write mode** (`EXL3_RDMA_FLAGW=1`: each data write is chained with a flag write that the
  peer's GPU polls, no receiving thread on the path): 14.5 µs per all-reduce against 13.9 µs without it ([6, 5120] bf16
  in a graph, both PCIe functions; NCCL 42 µs), 0 wrong results in a stress test of 1M calls, but −0.01 % per round in
  the serve. There a ring call cost 21–35 µs in the profile of 19.10., the rest being one rank waiting for the other,
  not the receive path; in the profile of 21.10. (ring over both PCIe functions, CPU pin) it is 14.4 µs (median).
- **Cortex-X925 cores at 3.9 instead of 2.8 GHz** (since `EXL3_CPU_PIN` all vLLM processes run on them; live, GPU
  2190 MHz): sustained −0.3 %, decode code / prose / structured −0.6 / +0.3 / −0.7 %, hottest ACPI sensor 88.0 →
  90.7 °C. We keep 2.8 GHz.

## Note on untrusted text

The stock tokenizer path turns literal DeepSeek special tokens in user text (for example `<｜end▁of▁sentence｜>` in a
quoted e-mail) into the real special token. If you feed untrusted text, insert a zero-width space after `<` for every
`<｜…｜>` and `<|place_holder_mm_span_N|>` before the request reaches vLLM.

## Licenses and credits

Our patches: Apache-2.0 (`LICENSE`). The image contains vLLM (Apache-2.0), the sudoingX recipe and plugin
(Apache-2.0, see `NOTICE`), exllamav3 kernels (MIT), TensorFold kernels (Apache-2.0) and the NVIDIA CUDA base image of
`vllm/vllm-openai` (NVIDIA license terms). The weights are a derivative of DeepSeek-V4.1-Flash-1017 (MIT).
Thanks to DeepSeek (model and reference code); sudoingX (recipe, quantization pipeline, KLD panel, calibration data,
decode kernels, launcher, the experimental patch 0022 and the MoE switches we turn on); turboderp (EXL3); ashhart,
bertholomus and the TensorFold contributors (RDMA ring and its use for decode gathers, L2 prefetch, EXL3 prompt kernels;
bertholomus's TensorFold multi-stream depth rule is the idea behind 0041, Apache-2.0; bertholomus's kit_bench and Zig
build for the comparison of 21.10.); Mia-AiLab (the 1017 EAQ quant and the recipe we ran that comparison with); 0xSero;
christopher_owen (DSpark experiments, NCCL `^LL128`, measurement method); Elevatormusic (TensorFold test scripts and run
logs used above); coolbho3k (the idea of layering prompt lookup on DSpark; 0043 is an independent implementation, no
code taken); jayleaton (special tokens in user text, `NCCL_CUMEM_ENABLE=0`); Gilesco and 0rand (display reservation as
device memory, the question that led to 0057); siertum (pointer to DeepSeek's DeepInfer KV formats); see_spot_ruminate
(X925 clock cap); shayanx45/Elevatormusic (compaction analysis) and the DGX Spark forum.
