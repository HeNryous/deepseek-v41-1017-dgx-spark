# SPDX-License-Identifier: Apache-2.0
"""Prompt chunks' routed experts through TensorFold's grouped EXL3 kernels (VLLM_EXL3_MOE_TF_PROMPT=1, default off).

Why: an 8K prefill (session 10's trace, Oct 12) spent ~1.9 s of rank 0's 4.5 s in the routed experts plus most of
~0.7 s of casts and copies around them: every expert over the fused tier's row limit ran exllamav3's general path (a
gather, three full fp16 reconstructs, six Hadamard passes, three cuBLAS GEMMs with fp32 outputs, SwiGLU as five torch
ops). TensorFold's prompt path (tf_experts.py, csrc/tf_experts/: Apache-2.0, see csrc/NOTICE.tensorfold) groups the
chunk's rows by expert on the GPU, rotates each slot's input once, runs gate and up as one grouped launch that decodes
each weight tile once for up to 64 member rows into mma fragments, applies SwiGLU in an epilogue and adds each row's
weighted slots in a fixed order: about six launches a layer, every row's arithmetic independent of the chunk (greedy
output stays identical run to run). Same-box, its whole 8K prefill took 3.6 s of GPU time against our 4.5 s: we
spent +0.58 s in the routed experts and +0.67 s in casts and copies, and led on communication and mHC.

The tables are built once per layer from the bank's own tensors (trellis pointers and the down's sign vectors in
place; gate's and up's sign vectors copied out of the bank's interleaved layout: ~10 MB a layer). One scratch per device is shared by all layers (calls run in order on one stream).
"""
from __future__ import annotations

import os

import torch

from . import tf_experts as T

ENABLED = os.environ.get("VLLM_EXL3_MOE_TF_PROMPT", "0") == "1"
MIN_ROWS = T.EXACT_ROWS          # below it the decode paths serve the rows (no host read per layer)
# VLLM_EXL3_TF_PROMPT_TRANSIENT=1 (dsv41 patch 0076d, default 0): the prompt scratch's big buffers - gate/up's rotated
# rows xg / xu, the down input xd and the partial sums z: 240 + 240 + 54 + 480 MiB = 1,014 MiB at 4,096 rows on a TP2
# rank, twice that at 8,192 - live only for the duration of one call (torch.empty at the call's size, released after
# it) instead of for the whole run. The attention / indexer / mHC temporaries of a prefill step's other phases then
# reuse that memory from the caching allocator instead of stacking on top of it; the MoE phase's own peak is unchanged.
# Same kernels, same arguments, same bits: every element a prompt call reads is written earlier in the same call
# (rot_in -> xg / xu of the routed slots, grouped gate/up -> z, epilogue -> xd, down -> z, down_combine reads z), and
# the persistent scratch already carries stale data from earlier calls in the slots a call does not use.
TRANSIENT = os.environ.get("VLLM_EXL3_TF_PROMPT_TRANSIENT", "0") == "1"
T.TRANSIENT_ALLOC = TRANSIENT
# VLLM_EXL3_TF_PROMPT_ALIAS=1 (dsv41 patch 0076e, needs TRANSIENT): the down's partials reuse the rotated rows' storage
# (tf_experts.ALIAS_ROT_DOWN): a call's peak 1,014 -> 750 MiB at 4,096 rows. Same bits.
T.ALIAS_ROT_DOWN = TRANSIENT and os.environ.get("VLLM_EXL3_TF_PROMPT_ALIAS", "0") == "1"
_TINY: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

_tables: dict[int, T.Exl3RoutedExperts] = {}
_scratch: dict[tuple, T.Scratch] = {}


def _codebook(mcg: bool, mul1: bool) -> int:
    return T.CB_MUL1 if mul1 else T.CB_MCG if mcg else T.CB_3INST


def tables(bank, mcg: bool, mul1: bool) -> T.Exl3RoutedExperts:
    """TensorFold's per-layer tables over the bank's tensors, in physical slot order. Its prepare() would stack every
    sign vector into a fresh [E, n] copy; the down's are already contiguous [E, n] in the bank (w2_suh, w2_svh) and are
    used in place, so only gate's and up's (interleaved [E, 2, n] in the bank) are copied: ~10 MB a layer, not ~14."""
    ex = _tables.get(id(bank))
    if ex is None:
        t, E = bank.t, bank.num_experts
        views = [bank.views(p) for p in range(E)]
        dev = t["w13_trellis"].device

        def table(j):
            ws = [v[j][0] for v in views]
            for w in ws:
                if w.dtype != torch.int16 or not w.is_contiguous():
                    raise ValueError("trellis views must be contiguous int16")
            k2 = [T.k2_of(w) for w in ws]
            return (torch.tensor([w.data_ptr() for w in ws], dtype=torch.int64, device=dev),
                    torch.tensor(k2, dtype=torch.int32, device=dev), k2, ws)

        (gp, gk, gks, gw), (up_, uk, uks, uw), (dp, dk, dks, dw) = table(0), table(1), table(2)
        D, I = gw[0].shape[0] * 16, gw[0].shape[1] * 16
        tb = torch.tensor([(D * I // 256) * (gks[e] + uks[e] + dks[e]) * 16 for e in range(E)], dtype=torch.int64)
        s13, v13 = t["w13_suh"], t["w13_svh"]
        ex = T.Exl3RoutedExperts(
            gp, up_, dp, gk, uk, dk, s13[:, 0].contiguous(), s13[:, 1].contiguous(), v13[:, 0].contiguous(),
            v13[:, 1].contiguous(), t["w2_suh"], t["w2_svh"], E, D, I, _codebook(mcg, mul1),
            (min(gks + uks), max(gks + uks)), (min(dks), max(dks)), tb, gw + uw + dw,
            all(w.data_ptr() % 16 == 0 for w in gw + uw + dw))
        _tables[id(bank)] = ex
    return ex


def scratch(ex: T.Exl3RoutedExperts, rows: int, slots: int, device: torch.device) -> T.Scratch:
    """The device's prompt scratch for at least ``rows`` rows (grown in steps of 2,048 rows)."""
    key = (str(device), slots, ex.dims, ex.width, ex.count)
    s = _scratch.get(key)
    if s is None or s.rows < rows:
        s = T.Scratch(ex, -(-rows // 2048) * 2048, slots, device=device, prompt=True)
        if TRANSIENT:
            # dsv41 patch 0076d: the constructor's xd / z go straight back to the caching allocator
            s.xd, s.z = _tiny(device)
        _scratch[key] = s
    return s


def routed(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, perm: torch.Tensor, bank,
           act_limit: float, mcg: bool, mul1: bool) -> torch.Tensor:
    """fp32 [m, H]: the routing-weighted sum of each row's routed experts (ids < 0: no expert)."""
    ex = tables(bank, mcg, mul1)
    m, k = topk_ids.shape
    ids = topk_ids.long()
    pick = torch.where(ids >= 0, perm[ids.clamp(min=0)], ex.count).to(torch.int32).contiguous()
    wts = topk_weights.float().contiguous()
    s = scratch(ex, m, k, x.device)
    xb = x if x.is_contiguous() else x.contiguous()
    if not TRANSIENT:
        return T.routed(xb, pick, wts, ex, s, None, m, limit=float(act_limit) if act_limit else float("inf"),
                        act_mode=T.ACT_F32, group=True, prompt=True)
    # dsv41 patch 0076d: this call's buffers only (routed() makes xg / xu and grows z with torch.empty)
    s.xd = torch.empty((m * k, ex.width), dtype=torch.float16, device=x.device)
    s.xg = s.xu = None
    s.z = _tiny(x.device)[1]
    try:
        return T.routed(xb, pick, wts, ex, s, None, m, limit=float(act_limit) if act_limit else float("inf"),
                        act_mode=T.ACT_F32, group=True, prompt=True)
    finally:
        tiny16, tiny32 = _tiny(x.device)
        s.xg = s.xu = None
        s.xd, s.z = tiny16, tiny32


def _tiny(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """dsv41 patch 0076d: placeholders the scratch keeps between calls (fp16 [1, 1], fp32 [1])."""
    key = str(device)
    t = _TINY.get(key)
    if t is None:
        t = _TINY[key] = (torch.empty((1, 1), dtype=torch.float16, device=device),
                          torch.empty((1,), dtype=torch.float32, device=device))
    return t
