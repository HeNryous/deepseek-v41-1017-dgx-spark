# SPDX-License-Identifier: Apache-2.0
"""Routed-expert forward for one layer on one TP rank (torch + exllamav3_ext; no vLLM import).

The vLLM method's apply() is a thin wrapper around routed_moe_forward(), so the kernel tests can run
the exact serving path without vLLM.

Paths, chosen by batch rows m (static per CUDA-graph capture size):
  0. m <= coop.rows_max and bf16 input with an exl3_smoe layer (smoe.py, VLLM_EXL3_MOE_SMOE, default on): our kernel,
     every K group in one launch set (rot, gate/up, down), no side streams. Deterministic, no host sync.
  0b. coop.rows_max < m <= VLLM_EXL3_MOE_SPLIT_MAX_TOKENS (dsv41 patch 0064, default off), bf16 input, exl3_smoe layer:
     exl3_smoe on row chunks of <= 42 rows, eager or captured; no host sync, each row's bits as in one smoe call.
  1. m <= coop.rows_max (decode, DSpark verify, every capture size): exl3_moe_coop per K group,
     chained through sh_out. No sort, no host sync, deterministic.
  2. capturing with m > coop.rows_max, or decode_kernel == "fused": exl3_moe per K group with
     scatter_add counts and num_active = -1 (no host sync); needs m <= fused.rows.
  3. eager prefill: per K group, exl3_moe in 64/32/16-row tiers for experts up to R rows (one host
     read of the counts per group) + reconstruct and fp16 GEMM for hotter experts. Waits for
     `prefill_wait` first (vLLM's shared-expert stream): see run_prefill.
"""

from __future__ import annotations

import json
import logging
import os

import torch

from . import kernels as K
from .bank import Exl3Bank


# VLLM_EXL3_MOE_STATS=/path.jsonl (eager serves only: CUDA-graph replays never run this Python): per decode call, the
# number of tokens and of distinct experts it reads on this rank's layer, flushed as one summary line per 2,000 calls.
# What a DSpark verify step costs in expert reads depends on how many experts its draft tokens share.
_STATS_PATH = os.environ.get("VLLM_EXL3_MOE_STATS")
_STATS = {} if _STATS_PATH else None
# VLLM_EXL3_MOE_FUSED_OPS (default 1): bf16 decode calls use fused_ops.py's prep / sum kernels (bit-identical results)
_FUSED = os.environ.get("VLLM_EXL3_MOE_FUSED_OPS", "1") == "1"
# VLLM_EXL3_MOE_DET_PREFILL=1: the eager prefill path writes every assignment's output into its own fp32 slot and sums
# each token's slots in k order (exllamav3's exl3_moe output_scratch / fused_base + exl3_moe_gather), so identical
# prompts give identical bits run to run; 0 = exllamav3's atomic accumulation (arrival order). Decode is deterministic
# either way (exl3_smoe / coop).
_DET_PREFILL = os.environ.get("VLLM_EXL3_MOE_DET_PREFILL", "1") == "1"
# VLLM_EXL3_MOE_TF_PROMPT=1: prompt chunks of >= 64 rows through TensorFold's grouped EXL3 kernels (tf_prompt.py):
# deterministic too, other rounding than the path below; off until measured on the serve
_TF_PROMPT = os.environ.get("VLLM_EXL3_MOE_TF_PROMPT", "0") == "1"
# VLLM_EXL3_MOE_SMOE_CHECK=N (debug, eager calls only): the first N exl3_smoe calls also run the coop path on the same
# inputs and append the difference to VLLM_EXL3_MOE_SMOE_CHECK_LOG (default /root/.cache/smoe_check.jsonl)
_SMOE_CHECK = int(os.environ.get("VLLM_EXL3_MOE_SMOE_CHECK", "0") or 0)
_SMOE_CHECK_LOG = os.environ.get("VLLM_EXL3_MOE_SMOE_CHECK_LOG", "/root/.cache/smoe_check.jsonl")
_SMOE_CHECKED = [0]
# VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=N (dsv41 patch 0064, default 0 = off; 63 = every row count below TensorFold's prompt
# path): a call with coop.rows_max < m <= N rows, bf16 input and an exl3_smoe layer runs its routed experts as
# exl3_smoe launches on row chunks of at most smoe.MAX_SLOTS // top_k rows (42 at top_k 6), full chunks first, instead
# of exllamav3's prefill tiers (_run_prefill_det: one host read of the expert counts a layer) or, under CUDA graph
# capture, exl3_moe's atomic tier. exl3_smoe computes every row on its own (a run's rows are separate rows of the mma
# fragments, each with a whole-K fp32 chain; the epilogues are per row and a token's picks are summed in pick order:
# csrc/exl3_smoe.cu gu_item / down_item), so each row gets the bits one exl3_smoe call over all m rows would give, and
# the chunks need no host read (graph-safe). Calls of <= coop.rows_max rows are not touched (the branch sits behind
# the decode branch and needs m > rows_max). The shared expert keeps its path for m > rows_max (moe_method.py
# _apply_with_shared: the runner, on vLLM's aux stream); exl3_smoe keeps its own counters, so no join with that stream
# is needed (run_prefill's join guards exllamav3's shared locks buffer). Against exllamav3's tiers those steps' rows
# get exl3_smoe's rounding, as in decode steps. TensorFold's prompt path (VLLM_EXL3_MOE_TF_PROMPT) keeps the eager
# calls it serves (m >= tf_prompt.MIN_ROWS).
_SPLIT_MAX = max(0, int(os.environ.get("VLLM_EXL3_MOE_SPLIT_MAX_TOKENS", "0") or 0))
_SPLIT_LOGGED = [False]


def _record_stats(m: int, phys: torch.Tensor, valid: torch.Tensor) -> None:
    uniq = int(torch.unique(phys[valid]).numel())
    b = _STATS.setdefault(m, [0, 0, 0])
    b[0] += 1
    b[1] += uniq
    b[2] += int(valid.sum())
    if sum(v[0] for v in _STATS.values()) % 2000 == 0:
        with open(_STATS_PATH, "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "calls_by_tokens": {k: v[0] for k, v in sorted(_STATS.items())},
                                "mean_distinct_experts": {k: round(v[1] / v[0], 2) for k, v in sorted(_STATS.items())},
                                "mean_routed_slots": {k: round(v[2] / v[0], 2) for k, v in sorted(_STATS.items())}}) + "\n")


def split_rows(m: int, cap: int) -> list[tuple[int, int]]:
    """dsv41 patch 0064: row ranges [a, b) covering 0..m in chunks of at most cap rows, full chunks first. An expert
    picked in several chunks is read once per chunk; a short last chunk shares the fewest experts with the others."""
    if m < 1 or cap < 1:
        raise ValueError(f"split_rows: m={m}, cap={cap}")
    return [(a, min(a + cap, m)) for a in range(0, m, cap)]


def smoe_split_ok(m: int, x: torch.Tensor, decode_kernel: str, coop: K.CoopScratch, smoe) -> bool:
    """dsv41 patch 0064: whether this call's routed experts run as exl3_smoe row chunks (see _SPLIT_MAX)."""
    if not (_SPLIT_MAX and smoe is not None and decode_kernel == "coop" and x.dtype == torch.bfloat16
            and coop.rows_max < m <= _SPLIT_MAX):
        return False
    if _TF_PROMPT and not torch.cuda.is_current_stream_capturing():
        from . import tf_prompt
        if m >= tf_prompt.MIN_ROWS:
            return False                     # TensorFold's prompt kernels keep the eager calls they serve
    return True


def smoe_split(smoe, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, perm: torch.Tensor,
               act_limit: float, pad: torch.Tensor | None = None) -> torch.Tensor:
    """dsv41 patch 0064: this rank's routed partial sum, bf16 [m, H], as exl3_smoe launches on row chunks of at most
    smoe.max_rows() rows, each writing its rows of one output tensor (no host read, no copy)."""
    m, hidden = x.shape
    cap = smoe.max_rows()
    if not _SPLIT_LOGGED[0]:
        _SPLIT_LOGGED[0] = True
        logging.getLogger(__name__).info(
            "EXL3 routed experts: a %d-row call runs exl3_smoe on row chunks of <= %d rows (dsv41 patch 0064, "
            "VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=%d)", m, cap, _SPLIT_MAX)
    out = torch.empty((m, hidden), dtype=torch.bfloat16, device=x.device)
    for a, b in split_rows(m, cap):
        smoe.into(out[a:b], x[a:b], topk_ids[a:b], topk_weights[a:b], perm, act_limit,
                  None if pad is None else pad[a:b])
    return out


def routed_moe_forward(
    x: torch.Tensor,                 # [m, H] bf16/fp16 (post-norm MoE input)
    topk_weights: torch.Tensor,      # [m, top_k] fp32 (routed_scaling_factor already applied by vLLM's router)
    topk_ids: torch.Tensor,          # [m, top_k] int32/int64 logical expert ids, < 0 = unrouted
    perm: torch.Tensor,              # int64 [E] logical -> physical (K-sorted) slot, on device
    groups: list[K.GroupTables],
    bank: Exl3Bank,
    coop: K.CoopScratch,
    fused: K.FusedScratch,
    act_limit: float,
    mcg: bool,
    mul1: bool,
    decode_kernel: str = "coop",
    par: tuple | None = None,        # (scratch sets, side streams) from K.get_parallel_coop: concurrent K groups
    prefill_wait: torch.cuda.Stream | None = None,   # stream whose queued work the eager prefill path waits for
    smoe=None,                       # smoe.SmoeLayer of this layer, or None
    pad: torch.Tensor | None = None,  # bool [>= m]: rows that take no experts (vLLM's is_padding), smoe path only
) -> torch.Tensor:
    m, hidden = x.shape
    if m == 0:
        return torch.zeros_like(x)
    top_k = topk_ids.shape[-1]
    coop_path = decode_kernel == "coop" and m <= coop.rows_max
    if smoe is not None and coop_path and x.dtype == torch.bfloat16 and smoe.rows_ok(m):
        out = smoe(x, topk_ids, topk_weights, perm, act_limit, pad)
        if _SMOE_CHECK and _SMOE_CHECKED[0] < _SMOE_CHECK and not torch.cuda.is_current_stream_capturing():
            _SMOE_CHECKED[0] += 1
            ref = routed_moe_forward(x, topk_weights, topk_ids, perm, groups, bank, coop, fused, act_limit, mcg, mul1,
                                     decode_kernel, par, prefill_wait, None).float()
            d = out.float() - ref
            rows = (d.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)).tolist()
            with open(_SMOE_CHECK_LOG, "a") as f:
                f.write(json.dumps({"pid": os.getpid(), "call": _SMOE_CHECKED[0], "layer": getattr(smoe, "layer", -1),
                                    "m": m, "rel": round((d.norm() / ref.norm().clamp_min(1e-12)).item(), 6),
                                    "row_rel_max": round(max(rows), 6), "max_abs": round(d.abs().max().item(), 5),
                                    "ref_max": round(ref.abs().max().item(), 3), "x_max": round(x.float().abs().max().item(), 3),
                                    "ids_dtype": str(topk_ids.dtype), "ids_shape": list(topk_ids.shape),
                                    "ids_stride": list(topk_ids.stride()), "w_dtype": str(topk_weights.dtype),
                                    "x_stride": list(x.stride())}) + "\n")
        return out
    if _SPLIT_MAX and smoe_split_ok(m, x, decode_kernel, coop, smoe):
        # dsv41 patch 0064: m > coop.rows_max as exl3_smoe row chunks, no host read (see _SPLIT_MAX)
        return smoe_split(smoe, x, topk_ids, topk_weights, perm, act_limit, pad)
    if coop_path and _FUSED and x.dtype == torch.bfloat16:
        # decode: one prep launch + (parallel groups) one sum launch instead of ~13 tiny torch ops (fused_ops.py)
        from .fused_ops import moe_prep, sum_groups_bf16
        x16, phys, rw16 = moe_prep(x, topk_ids, topk_weights, perm)
        if _STATS is not None and not torch.cuda.is_current_stream_capturing():
            _record_stats(m, phys, topk_ids >= 0)
        if par is not None and len(groups) > 1:
            outs = K.run_decode_coop_parallel(x16, phys, rw16, groups, par[0], par[1], act_limit, mcg, mul1,
                                              reduce=False)
            return sum_groups_bf16(outs, m)
        return K.run_decode_coop(x16, phys, rw16, groups, coop, act_limit, mcg, mul1).to(x.dtype)
    if _TF_PROMPT and not coop_path and not torch.cuda.is_current_stream_capturing():
        from . import tf_prompt
        if m >= tf_prompt.MIN_ROWS:
            if prefill_wait is not None and prefill_wait != torch.cuda.current_stream():
                torch.cuda.current_stream().wait_stream(prefill_wait)
            return tf_prompt.routed(x, topk_weights, topk_ids, perm, bank, act_limit, mcg, mul1).to(x.dtype)
    x16 = x.to(torch.float16).contiguous()
    ids = topk_ids.to(torch.long)
    valid = ids >= 0
    phys = perm[ids.clamp(min=0)].contiguous()
    rw16 = torch.where(valid, topk_weights, torch.zeros_like(topk_weights)).to(torch.float16).contiguous()

    if coop_path:
        if _STATS is not None and not torch.cuda.is_current_stream_capturing():
            _record_stats(m, phys, valid)
        if par is not None and len(groups) > 1:
            out = K.run_decode_coop_parallel(x16, phys, rw16, groups, par[0], par[1], act_limit, mcg, mul1)
            return out.to(x.dtype)                 # already a fresh tensor (the sum), not scratch
        out = K.run_decode_coop(x16, phys, rw16, groups, coop, act_limit, mcg, mul1)
        # the result lives in process-wide scratch that the next layer overwrites: always copy out
        return out.to(x.dtype) if x.dtype != torch.float32 else out.clone()

    out32 = torch.zeros((m, hidden), dtype=torch.float32, device=x.device)
    flat_p = phys.reshape(-1)
    flat_w = rw16.reshape(-1)
    flat_tok = torch.arange(m, device=x.device).repeat_interleave(top_k)

    if torch.cuda.is_current_stream_capturing() or (decode_kernel == "fused" and m <= fused.rows):
        if m > fused.rows:
            raise RuntimeError(
                f"vllm_exl3_moe: CUDA graph capture at {m} tokens exceeds the graph-safe limit "
                f"(coop {coop.rows_max}, fused {fused.rows}); lower --max-cudagraph-capture-size "
                f"or use cudagraph_mode FULL_DECODE_ONLY")
        for g in groups:
            counts, tok, w = K.bucket_rows(flat_p, g, flat_tok, flat_w)
            K.run_exl3_moe(x16, out32, g, counts, tok, w, fused, act_limit, mcg, mul1, num_active=-1)
        return out32.to(x.dtype)

    run_prefill(x16, out32, flat_p, flat_tok, flat_w, groups, bank, fused, act_limit, mcg, mul1, prefill_wait)
    return out32.to(x.dtype)


def run_prefill(x16, out32, flat_p, flat_tok, flat_w, groups, bank: Exl3Bank, fused: K.FusedScratch,
                act_limit: float, mcg: bool, mul1: bool, wait: torch.cuda.Stream | None = None) -> None:
    """Eager only. exllamav3's own tiering (block_sparse_mlp.py:1131-1145, 1261-1281) per K group.
    exl3_moe takes its group barriers and expert tickets from exllamav3's per-device locks buffer, the one every
    exllamav3 GEMM / GEMV launch also uses (exl3_devctx.cu get_locks): two such launches in flight at once corrupt
    each other's counters and the barriers spin forever. vLLM enqueues the shared expert (an EXL3 dense linear in our
    builds) on its aux stream just before the routed experts for batches up to VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
    tokens (256; moe_runner.py _forward_impl), so `wait` (that stream) is joined before the first launch: both TP
    ranks hung in their first prefill over 32 tokens without it (Oct 9). The next layer's shared expert waits
    for this stream's work, so nothing overlaps these launches after the join. The coop decode kernels keep their own
    counters and need no join.
    Prefill on GB10 measures 1,565 tok/s at 8K (kit_bench, the full serve); a faster path would be our own sm_121
    grouped GEMM that decodes each trellis tile once per row block. With VLLM_EXL3_MOE_DET_PREFILL=1 (the default)
    _run_prefill_det below replaces this path: deterministic accumulation."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("run_prefill host-syncs and must not run under CUDA graph capture")
    if wait is not None and wait != torch.cuda.current_stream():
        torch.cuda.current_stream().wait_stream(wait)
    if _DET_PREFILL:
        return _run_prefill_det(x16, out32, flat_p, flat_tok, flat_w, groups, bank, fused, act_limit, mcg, mul1)
    R = fused.rows
    for g in groups:
        counts, tok, w = K.bucket_rows(flat_p, g, flat_tok, flat_w)
        ch = counts.tolist()                      # host sync (eager prefill only)
        n = g.size
        fz = [c for c in ch[:n] if 0 < c <= R]
        t2 = sum(1 for c in fz if c > K.MTILE_T2)
        t1 = sum(1 for c in fz if K.MTILE_T1 < c <= K.MTILE_T2)
        t0 = len(fz) - t1 - t2

        def run(**kw):
            K.run_exl3_moe(x16, out32, g, counts, tok, w, fused, act_limit, mcg, mul1, **kw)

        if mul1 and (t1 or t2):                   # 32/64-row instances exist for mul1 only (exl3_moe.cu:436-448)
            if t2:
                run(num_active=t2, count_lo=K.MTILE_T2 + 1, count_hi=R, m_tile=64)
            if t1:
                run(num_active=t1, count_lo=K.MTILE_T1 + 1, count_hi=K.MTILE_T2, m_tile=32)
            if t0:
                run(num_active=t0, count_lo=1, count_hi=K.MTILE_T1, m_tile=16)
        elif fz:
            run(num_active=len(fz))

        start = 0
        for e_loc in range(n):
            c = ch[e_loc]
            if c > R:
                rows = tok[start:start + c]
                wt = w[start:start + c].float().unsqueeze(1)
                w1, w3, w2 = bank.views(g.lo + e_loc)
                xe = x16.index_select(0, rows)
                gate = K.exl3_linear_reconstruct(xe, *w1, mcg, mul1)
                up = K.exl3_linear_reconstruct(xe, *w3, mcg, mul1)
                a16 = K.swiglu_limit(gate, up, act_limit).to(torch.float16)
                y = K.exl3_linear_reconstruct(a16, *w2, mcg, mul1)
                out32.index_add_(0, rows, y * wt)
            start += c


def _run_prefill_det(x16, out32, flat_p, flat_tok, flat_w, groups, bank: Exl3Bank, fused: K.FusedScratch,
                     act_limit: float, mcg: bool, mul1: bool) -> None:
    """run_prefill with deterministic accumulation: one stable sort of the assignments by physical expert, a slot a
    assignment in one fp32 scratch (the fused tiers write weighted outputs there, the hot experts' reconstruct GEMMs
    too), then one exl3_moe_gather sums each token's slots in k order into out32. Each group's experts are a
    contiguous range of physical slots, so a group's rows are a slice of the global order."""
    import numpy as np
    from .kernels import ext as _k_ext
    m, hidden = x16.shape
    A = flat_p.numel()
    E = max(g.hi for g in groups)
    dev = x16.device
    order = torch.argsort(flat_p, stable=True)
    inv_order = torch.empty_like(order).scatter_(0, order, torch.arange(A, device=dev))
    counts_all = torch.zeros((E,), dtype=torch.long, device=dev).scatter_add_(0, flat_p, torch.ones_like(flat_p))
    ch = counts_all.tolist()                                     # host sync (eager prefill only)
    starts = np.concatenate(([0], np.cumsum(ch[:-1]))).astype(np.int64)
    base = starts.copy()                                         # every assignment its own slot: slot = sorted position
    kind = (np.asarray(ch) > 0).astype(np.int64)                 # 1: slots hold weighted outputs
    tok_sorted = flat_tok[order].contiguous()
    w_sorted = flat_w[order].contiguous()
    scratch = torch.empty((max(A, 1), hidden), dtype=torch.float32, device=dev)
    base_t = torch.from_numpy(base).to(dev)
    start_t = torch.from_numpy(starts).to(dev)
    kind_t = torch.from_numpy(kind).to(dev)
    R = fused.rows
    for g in groups:
        n = g.size
        cg = ch[g.lo:g.hi]
        s0, s1 = int(starts[g.lo]), int(starts[g.lo]) + sum(cg)
        if s1 == s0:
            continue
        counts = torch.tensor(cg + [0], dtype=torch.long, device=dev)   # the sentinel bin (outside the group) is empty
        tok, w = tok_sorted[s0:s1], w_sorted[s0:s1]
        fb = base_t[g.lo:g.hi]
        fz = [c for c in cg if 0 < c <= R]
        t2 = sum(1 for c in fz if c > K.MTILE_T2)
        t1 = sum(1 for c in fz if K.MTILE_T1 < c <= K.MTILE_T2)
        t0 = len(fz) - t1 - t2

        def run(**kw):
            K.run_exl3_moe(x16, out32, g, counts, tok, w, fused, act_limit, mcg, mul1,
                           output_scratch=scratch, fused_base=fb, **kw)

        if mul1 and (t1 or t2):
            if t2:
                run(num_active=t2, count_lo=K.MTILE_T2 + 1, count_hi=R, m_tile=64)
            if t1:
                run(num_active=t1, count_lo=K.MTILE_T1 + 1, count_hi=K.MTILE_T2, m_tile=32)
            if t0:
                run(num_active=t0, count_lo=1, count_hi=K.MTILE_T1, m_tile=16)
        elif fz:
            run(num_active=len(fz))
        start = 0
        for e_loc in range(n):
            c = cg[e_loc]
            if c > R:
                rows = tok[start:start + c]
                wt = w[start:start + c].float().unsqueeze(1)
                w1, w3, w2 = bank.views(g.lo + e_loc)
                xe = x16.index_select(0, rows)
                gate = K.exl3_linear_reconstruct(xe, *w1, mcg, mul1)
                up = K.exl3_linear_reconstruct(xe, *w3, mcg, mul1)
                a16 = K.swiglu_limit(gate, up, act_limit).to(torch.float16)
                y = K.exl3_linear_reconstruct(a16, *w2, mcg, mul1)
                b = int(base[g.lo + e_loc])
                torch.mul(y, wt, out=scratch[b:b + c])
            start += c
    _k_ext().exl3_moe_gather(out32, scratch, flat_p, inv_order, start_t, base_t, kind_t, w_sorted)


def reference_moe(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, bank: Exl3Bank,
                  act_limit: float, mcg: bool, mul1: bool) -> torch.Tensor:
    """Slow per-(token, expert) reference on exllamav3's reconstruct path (tests only), fp32 [m, H]."""
    x16 = x.to(torch.float16)
    out = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            if e < 0:
                continue
            w1, w3, w2 = bank.views(bank.lp.perm[e])
            xt = x16[t:t + 1].contiguous()
            gate = K.exl3_linear_reconstruct(xt, *w1, mcg, mul1)
            up = K.exl3_linear_reconstruct(xt, *w3, mcg, mul1)
            a16 = K.swiglu_limit(gate, up, act_limit).to(torch.float16)
            out[t] += float(topk_weights[t, j]) * K.exl3_linear_reconstruct(a16, *w2, mcg, mul1)[0]
    return out
