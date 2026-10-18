# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
import torch.nn as nn

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum, get_tma_aligned_size


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor | QuantizedActivation,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization. ``wo_b`` is any callable over
    the flattened ``z``: the projection module itself, or a wrapper that also
    reduce-scatters its output (DeepSeek-V4.1 GEMM-RS).

    A QuantizedActivation ``o`` was already rotated and cast by the attention
    kernel, one ``wo_a`` group per slot with the live groups first, so only
    the padding groups are sliced off.
    """
    grouped_bmm = getattr(wo_a, "grouped_bmm", None)
    if grouped_bmm is not None and not isinstance(o, QuantizedActivation):
        # A quantization method that runs wo_a group by group itself (e.g. EXL3
        # trellis weights): o_rot [tokens, groups, K] -> [tokens, groups, N].
        o_rot, _ = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=einsum_recipe[2],
            tma_aligned_scales=tma_aligned_scales,
            quantize=False,
        )
        if _WOA_FP8_ACT:
            # DeepSeek-V4.1-Flash reference since Oct 12 2026 (1017 ships it): o is block-wise FP8
            # quantized and dequantized (32 activations per ue8m0 scale) before the wo_a einsum.
            o_rot = _fp8_qdq_ue8m0(o_rot)
        return wo_b(grouped_bmm(o_rot).flatten(1))
    if getattr(wo_a, "sm12x_bmm_groups", None) and not isinstance(
        o, QuantizedActivation
    ):
        # SM12x: FP8 wo_a through one Marlin MXFP8 GEMM per group (DeepGEMM's
        # fp8_einsum is datacenter-Blackwell only; the weight is Marlin-packed).
        from vllm.model_executor.kernels.linear.mxfp8.sm12x_bmm import (
            sm12x_marlin_bmm,
        )

        o_rot, _ = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=einsum_recipe[2],
            tma_aligned_scales=tma_aligned_scales,
            quantize=False,
        )
        n, k = wo_a.sm12x_bmm_nk
        z = sm12x_marlin_bmm(o_rot, wo_a.weight, wo_a.weight_scale, n_groups, n, k)
        return wo_b(z.flatten(1))
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    if isinstance(o, QuantizedActivation):
        assert use_fp8
        o_proj_input, o_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
    else:
        o_proj_input, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=einsum_recipe[2],
            tma_aligned_scales=tma_aligned_scales,
            quantize=use_fp8,
        )
    z = torch.empty(
        (o_proj_input.shape[0], n_groups, o_lora_rank),
        device=o_proj_input.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (wo_a.weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    return wo_b(z.flatten(1))


# TRTLLM-gen's DSv4 RopeQuant epilogue (flashinfer-ai/flashinfer#4918) has one
# fixed schedule: 128 query heads in groups of 8 512-wide heads, and FP8 with
# one packed UE8M0 scale per 128 elements.
_ROPE_QUANT_NUM_HEADS = 128
_ROPE_QUANT_HEADS_PER_GROUP = 8
_ROPE_QUANT_HEAD_DIM = 512
_ROPE_QUANT_BLOCK = 128


def rope_quant_unsupported_reason(layer: torch.nn.Module) -> str | None:
    """Why ``layer`` cannot take FlashInfer's fused inverse RoPE + FP8 output.

    DSv4 always has 8 heads per ``wo_a`` group at any TP size, and the rest of
    the kernel's fixed shape (head/RoPE dims, per-128 einsum scales, FP32 RoPE
    cache) is what the SM100 layer uses anyway. What varies is the local head
    count and the configs the layer also serves unquantized.
    """
    if layer.n_local_heads != _ROPE_QUANT_NUM_HEADS:
        # Padding fewer heads up to the kernel's fixed 128 costs more
        # attention than the fused epilogue saves.
        return f"RopeQuant needs {_ROPE_QUANT_NUM_HEADS} local query heads"
    if layer.kv_cache_torch_dtype != torch.float8_e4m3fn:
        return "RopeQuant needs the per-tensor FP8 KV cache"
    if layer.wo_a.weight.dtype != torch.float8_e4m3fn:
        return "RopeQuant needs an FP8 wo_a"
    return None


def rope_quant_attn_out(num_tokens: int, device: torch.device) -> QuantizedActivation:
    """Allocate the output pair the RopeQuant kernel writes.

    Values are group-major, ``[num_tokens, 16, 8 * 512]`` over physical
    ``[group, token, K]``. Scales are MN-major packed UE8M0, one int32 per
    head over its four 128-wide blocks. Both are what ``fp8_einsum`` consumes.
    """
    n_groups = _ROPE_QUANT_NUM_HEADS // _ROPE_QUANT_HEADS_PER_GROUP
    group_width = _ROPE_QUANT_HEADS_PER_GROUP * _ROPE_QUANT_HEAD_DIM
    data = torch.empty(
        (n_groups, num_tokens, group_width), dtype=torch.float8_e4m3fn, device=device
    ).transpose(0, 1)
    aligned = get_tma_aligned_size(num_tokens, torch.int32.itemsize)
    scale = torch.empty(
        (n_groups, group_width // (_ROPE_QUANT_BLOCK * 4), aligned),
        dtype=torch.int32,
        device=device,
    ).permute(2, 0, 1)[:num_tokens]
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=kFp8Dynamic128Sym,
    )

# --- DSV4.1-1017 reference parity (dsv41 recipe patch 0030) --------------------------------------------
import os as _os

_WOA_FP8_ACT = _os.environ.get("VLLM_DSV41_WOA_FP8_ACT", "1") == "1"


# Patch 0032 (18.10.2026): the round trip as ONE Triton kernel. The torch version below ran ~14 eager kernels with fp32
# temporaries (~58 B per element; ~0.2 s per 2048-token prefill chunk on 2x GB10, ~15 % of TTFT). The scale uses the
# reference's own IEEE-bit ceil(log2) (fast_log2_ceil / fast_pow2 in inference/kernel.py), not log2f/exp2f, so it is
# exact where the torch version could round at power-of-two boundaries. VLLM_DSV41_WOA_QDQ_TRITON=0 = torch version.
import triton as _triton
import triton.language as _tl

_QDQ_TRITON = _os.environ.get("VLLM_DSV41_WOA_QDQ_TRITON", "1") == "1"


@_triton.jit
def _qdq_ue8m0_kernel(x_ptr, y_ptr, n_groups, BLOCK_G: _tl.constexpr, GS: _tl.constexpr):
    g = _tl.program_id(0) * BLOCK_G + _tl.arange(0, BLOCK_G)
    offs = g[:, None] * GS + _tl.arange(0, GS)[None, :]
    mask = g[:, None] < n_groups
    x = _tl.load(x_ptr + offs, mask=mask, other=0.0).to(_tl.float32)
    amax = _tl.maximum(_tl.max(_tl.abs(x), axis=1), 1e-4)
    bits = (amax * (1.0 / 448.0)).to(_tl.uint32, bitcast=True)
    e = ((bits >> 23) & 0xFF).to(_tl.int32) - 127 + _tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    s = ((e + 127) << 23).to(_tl.uint32).to(_tl.float32, bitcast=True)
    q = _tl.clamp(x / s[:, None], -448.0, 448.0).to(_tl.float8e4nv)
    _tl.store(y_ptr + offs, (q.to(_tl.float32) * s[:, None]).to(y_ptr.dtype.element_ty), mask=mask)


def _fp8_qdq_ue8m0(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """act_quant(x, 32, "ue8m0", e8m0, inplace=True) of DeepSeek's inference/kernel.py:
    per 32 values amax (min 1e-4), scale = 2**ceil(log2(amax / 448)), e4m3 round trip, back to x.dtype."""
    if _QDQ_TRITON and x.is_cuda and group_size == 32 and x.shape[-1] % 32 == 0:
        x = x.contiguous()
        y = torch.empty_like(x)
        n_groups = x.numel() // 32
        _qdq_ue8m0_kernel[(_triton.cdiv(n_groups, 64),)](x, y, n_groups, BLOCK_G=64, GS=32)
        return y
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax * (1.0 / 448.0))))
    y = (xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return y.reshape(shape).to(x.dtype)
