# SPDX-License-Identifier: Apache-2.0
"""dsv41 patch 0059 (VLLM_DSV41_KV_NVFP4=1, default off): FlashInfer's 384-byte NVFP4 DSV4 cache record on SM12x.

FlashInfer's SM120 sparse-MLA kernels read a group-16 NVFP4 record for the sliding-window and the compressed KV
caches (``trtllm_batch_decode_sparse_mla_dsv4(kv_cache_format="nvfp4")``): 448 NoPE values as e2m1 pairs with one
e4m3 scale per 16 values, the 64 RoPE values copied as BF16, 384 bytes per token instead of the 584 of fp8_ds_mla.
Inside a page the data rows (352 B) come first and the scale rows (32 B) after them, so the layout is FlashInfer's
own: rows are written with its ``nvfp4_quantize_append_sparse_mla_cache`` after GPT-J RoPE on dims 448..511 here,
which keeps writer and reader bit-compatible by construction.
"""

import os

import torch

from vllm.triton_utils import tl, triton

# patch 0061: VLLM_DSV41_KV_NVFP4=2 keeps the sliding-window cache in the fp8_ds_mla record (584 B) and stores only the
# compressed cache as NVFP4, the split DeepSeek's own FP4 KV format and vLLM's nvfp4_ds_mla (SM100) use. Mode 1 (patch
# 0059) quantizes both; window F5 (20.10.2026) measured KL 0.028 against FP8 for it, 0.15 in the first 128 positions,
# where only the window is read.
_MODE = os.environ.get("VLLM_DSV41_KV_NVFP4", "0")
KV_NVFP4 = _MODE in ("1", "2")  # compressed cache NVFP4
KV_NVFP4_SWA = _MODE == "1"  # sliding-window cache NVFP4 too
NVFP4_RECORD_BYTES = 384
FP8_SWA_PAGE_ALIGNMENT = 576  # fp8_ds_mla pages: rows of 576 data bytes (448 FP8 + 64 BF16), the SM120 kernels' stride


@triton.jit
def _rope_rows_kernel(
    rows,
    positions,
    cos_sin,
    out,
    ROW_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    # Same rotation as fused_compress_quant_cache.py's insert kernels: GPT-J pairs, NoPE pairs load (cos, sin) =
    # (1, 0), the compressed state takes the position of its group's first token.
    t = tl.program_id(0)
    position = tl.load(positions + t)
    d = tl.arange(0, 512)
    x = tl.load(rows + t.to(tl.int64) * ROW_STRIDE + d).to(tl.float32)
    even, odd = tl.split(tl.reshape(x, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * s, odd * c + even * s)
    tl.store(out + t.to(tl.int64) * 512 + d, rotated.to(tl.bfloat16))


_module = None


def prebuild() -> None:
    """Build/load FlashInfer's NVFP4 module before CUDA graphs are captured (the writer runs inside captured regions)."""
    global _module
    if _module is None:
        from flashinfer.mla._sparse_mla_nvfp4_sm120 import get_sparse_mla_nvfp4_sm120_module

        _module = get_sparse_mla_nvfp4_sm120_module()


def nvfp4_rope_insert(
    rows: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int = 1,
) -> None:
    """GPT-J RoPE on the RoPE dims of BF16 latent rows, then FlashInfer's NVFP4 quantize-append by cache slot.

    ``rows`` holds one 512-wide BF16 row per slot (KV already normalized); slots < 0 are padding. With
    ``compress_ratio`` > 1 only rows ending a compress group are written, as in rope_quant_insert.
    """
    n = slot_mapping.numel()
    if n == 0:
        return
    assert rows.dtype == torch.bfloat16 and rows.shape[-1] == 512 and rows.stride(-1) == 1
    assert cache.dtype == torch.uint8 and cache.shape[-1] == NVFP4_RECORD_BYTES
    roped = torch.empty((n, 512), dtype=torch.bfloat16, device=rows.device)
    _rope_rows_kernel[(n,)](
        rows,
        positions,
        cos_sin_cache,
        roped,
        ROW_STRIDE=rows.stride(0),
        COS_STRIDE=cos_sin_cache.stride(0),
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
    )
    slots = slot_mapping
    if compress_ratio > 1:
        slots = torch.where(
            (positions[:n] + 1) % compress_ratio == 0, slot_mapping, torch.full_like(slot_mapping, -1)
        )
    nvfp4_append(roped, slots, cache)


def nvfp4_append(rows: torch.Tensor, slot_mapping: torch.Tensor, cache: torch.Tensor) -> None:
    """FlashInfer's NVFP4 quantize-append of already-roped BF16 rows (one per slot; slots < 0 are skipped)."""
    from flashinfer.mla._sparse_mla_nvfp4_sm120 import nvfp4_quantize_append_sparse_mla_cache

    m = slot_mapping.numel()
    if m == 0:
        return
    assert rows.shape[0] >= m and rows.shape[-1] == 512 and rows.dtype == torch.bfloat16
    nvfp4_quantize_append_sparse_mla_cache(rows[:m].contiguous(), slot_mapping.contiguous(), cache)


@triton.jit
def _lse_merge_kernel(o1, l1, o2, l2, sink, out, H: tl.constexpr, D: tl.constexpr, HAS_SINK: tl.constexpr):
    # out = (2^l1 o1 + 2^l2 o2) / (2^l1 + 2^l2 + e^sink) per (row, head). FlashInfer's SM120 sparse-MLA kernels return
    # the LSE of the scaled scores in base 2 (unit test: split + merge matches one call to 0.25 % with base 2, 9-13 %
    # off with base e). Both calls run WITHOUT the attention sink and the sink is added here: how the kernels fold it
    # into their LSE differs between FP8/NVFP4 and decode/prefill (2-4 % off when one call took it). The sink logit is
    # in natural units like the scores (FlashMLA V4: lse' = log(exp(lse) + exp(sink))). An empty segment (no
    # compressed entries yet, LSE ~ -1e30) has weight 0, whatever its output holds.
    t = tl.program_id(0)
    h = tl.program_id(1)
    a = tl.load(l1 + t * H + h)
    b = tl.load(l2 + t * H + h)
    LOG2E: tl.constexpr = 1.4426950408889634
    m = tl.maximum(a, b)
    if HAS_SINK:
        s2 = tl.load(sink + h).to(tl.float32) * LOG2E
        m = tl.maximum(m, s2)
    m = tl.where(m < -1e29, 0.0, m)
    wa = tl.where(a < -1e29, 0.0, tl.exp2(a - m))
    wb = tl.where(b < -1e29, 0.0, tl.exp2(b - m))
    den = wa + wb
    if HAS_SINK:
        den += tl.exp2(s2 - m)
    d = tl.arange(0, D)
    base = (t * H + h).to(tl.int64) * D
    x = tl.load(o1 + base + d).to(tl.float32)
    y = tl.load(o2 + base + d).to(tl.float32)
    x = tl.where(wa > 0, x, 0.0)
    y = tl.where(wb > 0, y, 0.0)
    r = tl.where(den > 0, (x * wa + y * wb) / tl.where(den > 0, den, 1.0), 0.0)
    tl.store(out + base + d, r.to(tl.bfloat16))


def _pad_nvfp4_primary(indices: torch.Tensor) -> torch.Tensor:
    width = indices.shape[-1]
    if width in (128, 512):
        return indices
    target = 128 if width < 128 else 512
    assert width <= 512, f"index width {width} > 512 has no NVFP4 kernel"
    return torch.nn.functional.pad(indices, (0, target - width), value=-1)


def split_sparse_mla(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    comp_cache: torch.Tensor,
    comp_indices: torch.Tensor,
    comp_lens: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    sinks: torch.Tensor | None,
    workspace: torch.Tensor,
) -> None:
    """patch 0061: sparse MLA over an FP8 sliding window and an NVFP4 compressed segment. FlashInfer's SM120 kernels
    read both pools of one call in one format, so the window runs as one call (fp8) and the compressed segment as
    another (nvfp4), both without the sink and with their LSE, merged by log-sum-exp (sink included) into ``out``.
    ``q``/``out``: [tokens, heads, 512] BF16; caches in the NHD view of _as_sparse_cache; indices [tokens, 1, width]."""
    from flashinfer.mla._core import _SparseMLASegment, _trtllm_batch_decode_sparse_mla_sm120

    t, h, d = q.shape
    if t == 0:
        return
    q4 = q.unsqueeze(1)
    lse_w = torch.empty((t, h), dtype=torch.float32, device=q.device)
    _trtllm_batch_decode_sparse_mla_sm120(
        query=q4, kv_cache=swa_cache, workspace_buffer=workspace,
        sparse_mla_segments=[_SparseMLASegment(indices=swa_indices, lengths=swa_lens)],
        out=out.view(t, 1, h, d), sm_scale=scale, sinks=None, lse=lse_w, return_lse=True,
        kv_scale_format="auto", kv_cache_format="fp8",
    )
    o_c = torch.empty((t, 1, h, d), dtype=torch.bfloat16, device=q.device)
    lse_c = torch.empty((t, h), dtype=torch.float32, device=q.device)
    _trtllm_batch_decode_sparse_mla_sm120(
        query=q4, kv_cache=comp_cache, workspace_buffer=workspace,
        sparse_mla_segments=[_SparseMLASegment(indices=_pad_nvfp4_primary(comp_indices), lengths=comp_lens)],
        out=o_c, sm_scale=scale, sinks=None, lse=lse_c, return_lse=True,
        kv_scale_format="auto", kv_cache_format="nvfp4",
    )
    sink = sinks if sinks is not None else lse_w  # placeholder pointer when there is no sink
    _lse_merge_kernel[(t, h)](out, lse_w, o_c, lse_c, sink, out, H=h, D=d, HAS_SINK=sinks is not None, num_warps=4)
