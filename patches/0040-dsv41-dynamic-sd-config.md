# 0040: DSpark draft count by batch size (configuration only, no code)

vLLM's dynamic speculative decoding (`num_speculative_tokens_per_batch_size`, already in the recipe's vLLM pin) sets the
number of DSpark drafts k per step from the number of requests in the batch. Each entry is `[min batch size, max batch
size, k]`. Until 21.10. we used variant B (since then `[[1,4,5]]`, see the end):

```bash
SPEC='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,2,5],[3,4,4]]}'
```

1–2 requests: k = 5 (6 verify rows each, at most 12 rows). 3–4 requests: k = 4 (5 rows each, at most 20 rows).

**Why.** Every request pays for its whole verify window (k + 1 rows in the dense layers and attention), measured on
19.10. at ~9.6 ms per request even at depth 0 (README, "Where the time goes"; the profile of 21.10. gives a round of
24.8 ms fixed, 3.6 ms per request and 2.8 ms per live verify row). And the plugin's `exl3_mrow` dense kernel took at
most 16 rows: 3 / 4 streams at k = 5 are 18 / 24 rows and fell back to exllamav3's GEMM (dense time per round 15.1 ms at
16 rows, 20.7 ms at 18, 21.0 ms at 24; micro-benchmark, one rank, 43 layers incl. the drafter). Variant B therefore runs
together with 0042/0042b (`VLLM_EXL3_MROW_MAX_ROWS=32`), which keeps the 20 rows on `exl3_mrow`.

**Measured** (GB10, 2000 MHz, T = 1, decode tok/s):

| | prose 3 / 4 streams | JSON 3 / 4 streams | code 3 / 4 streams | 1 stream |
|---|---|---|---|---|
| variant A `[[1,2,5],[3,3,4],[4,4,3]]` (never more than 16 rows), alone | +8.1 / +7.8 % | – | – / −5.3 % | – |
| variant B + 0042 + 0045 + 0046, against the same image without these switches | +9.4 / +6.4 % | +3.6 / +2.4 % | −0.6 / +1.2 % | +1…+2 % |

Variant A was dropped for the code loss at 4 streams, where it verifies only k = 3 drafts. Variant B with 0042, 0045 and 0046:
geometric mean over all cells +4.0 %, corruption test 0 hits. "–": not recorded.

**Default:** without `num_speculative_tokens_per_batch_size` vLLM uses `num_speculative_tokens` (k = 5) at every batch
size, as before. 0047 (adaptive verify window) only works on top of such a schedule.

**Since 21.10.** we run `[[1,4,5]]` (k = 5 at 1–4 streams, at most 24 verify rows). Against variant B (2000 MHz): tok/s
+1.27 % (geometric mean), tokens per round +2.37 %, cost per round +1.23 %; at 4 streams JSON +5.2 %, code +2.6 %, prose
+0.7 %; 1 stream unchanged; T=0 texts identical. Our reading: with 0042b's ring kernel up to 32 rows and 0047's window
per request, the full window now pays at 3–4 streams as well.
