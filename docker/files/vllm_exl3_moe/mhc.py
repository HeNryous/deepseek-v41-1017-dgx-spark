# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 mHC: post + the next sublayer's carried-pre input collapse + RMSNorm in one launch (csrc/exl3_mhc.cu).

vLLM's V4.1 decoder runs these as two one-block-per-token TileLang kernels at every sublayer boundary (mhc_post, then
mhc_pre_big_fuse_with_norm in split mode "input"), ~12 + ~9 us at a 6-token verify step, 80 boundaries a round on
GB10. post_collapse returns (residual_out, layer_input) like vLLM's fused all-reduce + mHC hook, so patch 0021 can use
it where that hook (TP=4 NVLink only) is unavailable; the mix statistics still run on vLLM's side stream.

VLLM_DSV41_MHC_FUSED_POST: 0 off (default, TileLang pair), 1 the chunked kernel (rounding class: its RMSNorm sum of
squares is added in another order), 2 post_collapse_exact (patch 0048): one block a token in TileLang's thread layout,
the same FMAs and the same sum-of-squares order, so both outputs keep TileLang's bits (H 5120 or 7168).
"""

from __future__ import annotations

import os

import torch

MODE = os.environ.get("VLLM_DSV41_MHC_FUSED_POST", "0")
ENABLED = MODE in ("1", "2")
EXACT = MODE == "2"
MAX_TOKENS = 256
_ext = None
_scratch: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def ext():
    global _ext
    if _ext is None:
        from .jit import load
        _ext = load("exl3_mhc", "EXL3_MHC_FLAGS")
    return _ext


def supported(x: torch.Tensor, residual: torch.Tensor) -> bool:
    n = x.shape[0]
    if EXACT and residual.dim() == 3 and residual.shape[2] not in (5120, 7168):
        return False
    return (0 < n <= MAX_TOKENS and residual.dim() == 3 and residual.shape[1] == 4 and residual.shape[2] % 256 == 0
            and residual.shape[2] <= 16384 and x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16)


def post_collapse(x: torch.Tensor, residual: torch.Tensor, comb: torch.Tensor, post: torch.Tensor, pre: torch.Tensor,
                  norm_weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """x [n, H] bf16, residual [n, 4, H] bf16, comb [n, 4, 4] f32, post [n, 4(, 1)] f32, pre [n, 4] f32 (the carried
    pre-mix), norm_weight [H] bf16 -> (residual_out [n, 4, H], layer_input [n, H]) bf16."""
    n, hc, h = residual.shape
    dev = x.device
    if EXACT:
        residual_out = torch.empty_like(residual)
        layer_input = torch.empty((n, h), dtype=torch.bfloat16, device=dev)
        w = norm_weight if norm_weight.dtype == torch.bfloat16 else norm_weight.to(torch.bfloat16)
        ext().post_collapse_exact(x.reshape(n, h).contiguous(), residual.contiguous(),
                                  comb.reshape(n, hc, hc).contiguous(), post.reshape(n, hc).contiguous(),
                                  pre.reshape(n, hc).contiguous(), w.contiguous(), residual_out, layer_input, float(eps))
        return residual_out, layer_input
    if dev not in _scratch:       # persistent: the counters must survive (and stay zero between) graph replays
        _scratch[dev] = (torch.empty((MAX_TOKENS, 64), dtype=torch.float32, device=dev),
                         torch.zeros((MAX_TOKENS,), dtype=torch.int32, device=dev))
    part, counter = _scratch[dev]
    residual_out = torch.empty_like(residual)
    layer_input = torch.empty((n, h), dtype=torch.bfloat16, device=dev)
    w = norm_weight if norm_weight.dtype == torch.bfloat16 else norm_weight.to(torch.bfloat16)
    ext().post_collapse(x.reshape(n, h).contiguous(), residual.contiguous(), comb.reshape(n, hc, hc).contiguous(),
                        post.reshape(n, hc).contiguous(), pre.reshape(n, hc).contiguous(), w.contiguous(),
                        residual_out, layer_input, part, counter, float(eps))
    return residual_out, layer_input


def reference(x, residual, comb, post, pre, norm_weight, eps):
    """vLLM's two TileLang kernels in torch, same operation order (tests): mhc_post_tilelang, then the "input" split
    of mhc_pre_big_fuse_with_norm with a carried pre-mix (save_pre_mix: bf16 boundary before the RMSNorm)."""
    # a += b * c compiles to one fused multiply-add (one rounding) in both TileLang's CUDA and ours: emulated here by
    # the exact product in fp64 rounded once to fp32
    def fma(b, c, a):
        return (b.double() * c.double() + a.double()).float()

    n, hc, h = residual.shape
    xf, rf = x.float(), residual.float()
    out = post.reshape(n, hc, 1).float() * xf.reshape(n, 1, h)
    for i in range(hc):
        out = fma(comb.reshape(n, hc, hc)[:, i, :].float().reshape(n, hc, 1), rf[:, i, :].reshape(n, 1, h), out)
    out = out.to(torch.bfloat16)
    col = torch.zeros((n, h), dtype=torch.float32, device=x.device)
    for o in range(hc):
        col = fma(pre.reshape(n, hc)[:, o:o + 1].float(), out[:, o, :].float(), col)
    rounded = col.to(torch.bfloat16).float()
    scale = torch.rsqrt((rounded * rounded).sum(-1, keepdim=True) / h + eps)
    return out, (rounded * scale * norm_weight.float()).to(torch.bfloat16)
