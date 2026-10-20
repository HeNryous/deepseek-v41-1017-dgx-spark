# SPDX-License-Identifier: Apache-2.0
"""EXL3 dense linears for DeepSeek-V4.1: attention wq_a + wkv / wq_b / wo_a / wo_b and the shared expert's gate_up / down.

The checkpoint's dense linears are block-FP8 (8.25 bits per weight with scales); on GB10 they are ~34% of a TP=2 decode
step and bandwidth-bound (Marlin at ~240 GB/s), so storing them as EXL3 trellis codes at K bits (5 by default: ~1%
relative error vs the FP8 weights, measured by scripts/quantize/p4_dense.py) cuts their bytes by ~40%. Checkpoint layout
(the EXL3 pack convention, also used by Mia-AiLab's 2.9 bpw pack), full matrices, cut per rank at load:
    layers.L.attn.wq_a / wkv.{trellis,suh,svh,mul1}  (-> fused_wqa_wkv shards 0 / 1)  replicated (in 5120, out 1280 / 512)
    layers.L.attn.wq_b.{trellis,suh,svh,mul1}                    column-parallel   (in 1280, out 32768)
    layers.L.attn.wo_a.slice.G.{...}  G = 0..7, one per head group column-parallel by group (in 4096, out 1024 each)
    layers.L.attn.wo_b.{...}                                     row-parallel      (in 8192, out 5120)
    layers.L.ffn.shared_experts.w1 / w3.{...}  (-> gate_up_proj shards 0 / 1, each with its own suh)   column-parallel
    layers.L.ffn.shared_experts.w2.{...}       (-> down_proj)                                           row-parallel
TP cuts are exact because EXL3's input / output transforms are 128-wide block Hadamards with per-channel signs: a column
cut at a multiple of 128 keeps whole output blocks (svh cut, suh whole), a row cut keeps whole input blocks (suh cut,
svh whole) and each rank's output transform is linear, so the partial sums still add up to the full product.

vLLM wiring (main @ 6e517b15 + patch 0006): DeepseekV41Exl3Config.get_quant_method returns this method for the module
paths above; V4.1's load_weights maps w1 / w3 to gate_up_proj with shard ids 0 / 1 and w2 to down_proj (no source change),
patch 0006 lets _pad_shared_expert_weight pass 1-D tensors and lets deep_gemm_fp8_o_proj hand wo_a's head groups to the
`grouped_bmm` set here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.utils import set_weight_attrs

from . import l2pf
from .hostcopy import copy_from_host

logger = init_logger(__name__)

# VLLM_EXL3_DENSE_MROW (default 1; "groups" = multi-matrix modules only, the Oct 9 v1 wiring; 0 = off): at decode sizes
# (<= MROW_MAX_ROWS rows) every dense EXL3 module runs as ONE exl3_mrow launch (mrow.py: bf16 in and out, deterministic):
# the multi-matrix modules (fused wq_a + wkv, the shared expert's gate + up, wo_a's head groups) and, since the kernel
# streams weights with TMA bulk copies, the single matrices too (wq_b, wo_b, the shared down: 1.13-1.31x exllamav3 at
# m=4-6, tests/tune_mrow.py). Prefill sizes stay on exllamav3.
# dsv41 patch 0042 (draft): VLLM_EXL3_MROW_MAX_ROWS=32 runs 17..32 rows (3-4-stream verify windows) on exl3_mrow's
# second plan (two m16 tiles, csrc/exl3_mrow.cu) instead of exllamav3's GEMM + casts; 16 = before.
MROW_MAX_ROWS = 32 if __import__('os').environ.get('VLLM_EXL3_MROW_MAX_ROWS', '16') == '32' else 16


# L2 prefetch points (l2pf.py) that follow a dense module's launch at decode sizes.
_L2PF_AFTER = {"attn.wq_b": "attn", "attn.wo_b": "moe"}

# dsv41 patch 0035 (19.10.2026, draft): prompt rows (> PROMPT_ROWS, exllamav3's reconstruct range) skip three copies
# that change no value: wo_a's input is cast to fp16 in one pass straight from the inverse-RoPE kernel's group-major
# [G, T, in] buffer (each head group then a contiguous [T, in] slab, no per-group .contiguous()), wo_a's group outputs
# and a merged module's part outputs are converted from fp32 straight into their column blocks of the bf16 result (no
# torch.stack / torch.cat of fp32). The same exllamav3 calls on the same values, the same fp32 -> bf16 rounding (RNE in
# copy_ as in .to): bit-identical. Decode graphs and replay batches up to PROMPT_ROWS keep the old code.
# VLLM_EXL3_DENSE_PROMPT_NOCOPY=0 restores it everywhere.
PROMPT_ROWS = 144


def _prompt_nocopy() -> bool:
    import os
    return os.environ.get("VLLM_EXL3_DENSE_PROMPT_NOCOPY", "1") == "1"


_PROMPT_NOCOPY = _prompt_nocopy()


def _discard_fp8(param, loaded_weight, *args, **kwargs) -> bool:
    """Weight loader of the Engram wkv's FP8 sink parameters: the EXL3 tensors carry the layer, the FP8 ones are dropped."""
    return True


def _mrow_mode() -> str:
    import os
    return os.environ.get("VLLM_EXL3_DENSE_MROW", "1")


def _mrow_grouped() -> bool:
    """EXL3_MROW_GROUPED=1: wo_a's mrow launch reads each head group where the inverse-RoPE kernel left it (its
    [G, T, D] buffer, seen as [T, G, D]) instead of after a copy to [T, G * D]: one copy kernel less a layer at verify
    sizes (4.3 us in the v7 profile), same inputs, bit-identical outputs."""
    import os
    return os.environ.get("EXL3_MROW_GROUPED", "0") == "1"

HAD = 128
DENSE_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(attn\.fused_wqa_wkv|attn\.wq_b|attn\.wo_a|attn\.wo_b|ffn\.shared_experts\.gate_up_proj"
    r"|ffn\.shared_experts\.down_proj)$")
# checkpoint module name (and K key in the config) of each vLLM module
CKPT_NAME = {"attn.fused_wqa_wkv": "attn.wq_a", "attn.wq_b": "attn.wq_b", "attn.wo_a": "attn.wo_a", "attn.wo_b": "attn.wo_b",
             "ffn.shared_experts.gate_up_proj": "ffn.shared_experts.w1",
             "ffn.shared_experts.down_proj": "ffn.shared_experts.w2"}


def dense_module(prefix: str) -> tuple[int, str] | None:
    """(layer index, vLLM module path) for the dense linears this method serves, else None."""
    m = DENSE_RE.search(prefix)
    return (int(m.group(1)), m.group(2)) if m else None


@dataclass(frozen=True)
class Cut:
    """What one rank keeps of a full EXL3 matrix [in, out]: rows (input) or columns (output), in channels."""
    row_parallel: bool
    lo: int
    hi: int

    def trellis(self, t: torch.Tensor) -> torch.Tensor:
        dim = 0 if self.row_parallel else 1
        return t.narrow(dim, self.lo // 16, (self.hi - self.lo) // 16)

    def suh(self, v: torch.Tensor) -> torch.Tensor:
        return v[self.lo:self.hi] if self.row_parallel else v

    def svh(self, v: torch.Tensor) -> torch.Tensor:
        return v if self.row_parallel else v[self.lo:self.hi]


def rank_cut(full: int, tp_size: int, tp_rank: int, row_parallel: bool) -> Cut:
    if full % tp_size or (full // tp_size) % HAD:
        raise ValueError(f"EXL3 dense: {full} channels do not split into {tp_size} whole {HAD}-wide Hadamard blocks")
    n = full // tp_size
    return Cut(row_parallel, tp_rank * n, (tp_rank + 1) * n)


def _holder(in_f: int, out_f: int, k: int, n: int = 1) -> dict[str, torch.Tensor]:
    """Empty per-rank tensors for n EXL3 matrices of the same shape."""
    lead = (n,) if n > 1 else ()
    return {"trellis": torch.empty(*lead, in_f // 16, out_f // 16, 16 * k, dtype=torch.int16),
            "suh": torch.empty(*lead, in_f, dtype=torch.float16),
            "svh": torch.empty(*lead, out_f, dtype=torch.float16),
            "mul1": torch.empty(n, dtype=torch.int32)}


class Exl3DenseLinearMethod(LinearMethodBase):
    """One method instance per vLLM linear module."""

    def __init__(self, module: str, k: int, layer_idx: int = -1, prefetch: bool = True):
        self.module = module
        self.k = int(k)
        self.layer_idx = layer_idx
        self.prefetch = prefetch          # backbone modules join l2pf's per-layer chain; the drafter's do not

    # ----------------------------------------------------------------------------------------------- weights
    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        extra_weight_attrs.pop("weight_loader", None)
        row = isinstance(layer, RowParallelLinear)
        tp_size, tp_rank = int(getattr(layer, "tp_size", 1)), int(getattr(layer, "tp_rank", 0))
        if isinstance(layer, ReplicatedLinear):   # vLLM still sets tp_size on it; its weights are whole on every rank
            tp_size, tp_rank = 1, 0
        k = self.k
        layer.exl3_k = k
        if self.module == "attn.wo_a":
            groups = int(getattr(layer, "bmm_batch_size", 0)) or 8 // tp_size   # attention.py sets it after init
            n_groups = groups * tp_size
            out_g = output_size // n_groups
            local = range(tp_rank * groups, (tp_rank + 1) * groups)
            layer.slice = torch.nn.ModuleList()
            for g in range(n_groups):
                holder = torch.nn.Module()
                shapes = _holder(input_size, out_g, k) if g in local else {
                    "trellis": torch.empty(0, dtype=torch.int16), "suh": torch.empty(0, dtype=torch.float16),
                    "svh": torch.empty(0, dtype=torch.float16), "mul1": torch.empty(0, dtype=torch.int32)}
                for name, t in shapes.items():
                    p = Parameter(t, requires_grad=False)
                    holder.register_parameter(name, p)
                    set_weight_attrs(p, {"weight_loader": self._loader(name, Cut(False, 0, out_g), g in local, k,
                                                                        (input_size, out_g))})
                layer.slice.append(holder)
            layer.exl3_groups = list(local)
            layer.exl3_shape = (input_size, out_g)
            # V4.1 code reads wo_a.weight.dtype to pick FP8 paths (o_proj.py, RopeQuant check): a bf16 placeholder
            layer.weight = torch.empty(0, dtype=torch.bfloat16)
            return
        n = len(output_partition_sizes)
        if len(set(output_partition_sizes)) != 1:
            if tp_size != 1 or row:
                raise ValueError(f"EXL3 dense {self.module}: unequal output partitions {output_partition_sizes} are "
                                 f"supported for replicated linears only")
            self._create_flat(layer, input_size, [int(o) for o in output_partition_sizes], k)
            return
        out_loc = output_partition_sizes[0]
        full_in = input_size
        full_out_each = out_loc * (1 if row else tp_size)
        cut = (rank_cut(full_in, tp_size, tp_rank, True) if row else rank_cut(full_out_each, tp_size, tp_rank, False))
        in_loc = input_size_per_partition
        for name, t in _holder(in_loc, out_loc, k, n).items():
            p = Parameter(t, requires_grad=False)
            layer.register_parameter(name, p)
            full_shape = (full_in, full_out_each)
            set_weight_attrs(p, {"weight_loader": self._loader(name, cut, True, k, full_shape, stacked=n > 1)})
        layer.exl3_n = n
        layer.exl3_shape = (in_loc, out_loc)
        if self.module == "engram.wkv":
            # An Engram variant keeps V4.1's shards 47 / 48 as hardlinks (they hold the 101.5 GB embedding tables next
            # to the FP8 wkv), so the FP8 wkv tensors still stream in, and V4.1's load_weights wants a parameter for
            # each (.scale maps to weight_scale or weight_scale_inv): empty sinks that drop them (LOG Oct 11 12:40)
            for name in ("weight", "weight_scale", "weight_scale_inv"):
                p = Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
                layer.register_parameter(name, p)
                set_weight_attrs(p, {"weight_loader": _discard_fp8})

    def _create_flat(self, layer: torch.nn.Module, in_f: int, outs: list[int], k: int) -> None:
        """A replicated merged linear with unequal shards (fused_wqa_wkv: wq_a 1280 + wkv 512 outputs): each piece is one
        flat buffer holding shard i at its offset; V4.1's stacked_params_mapping hands the loader shard ids 0..n-1."""
        for o in outs:
            if in_f % HAD or o % HAD:
                raise ValueError(f"EXL3 dense {self.module}: {in_f} x {o} is not in whole {HAD}-wide Hadamard blocks")
        tr = [(in_f // 16) * (o // 16) * 16 * k for o in outs]
        tr_off = [sum(tr[:i]) for i in range(len(outs))]
        sv_off = [sum(outs[:i]) for i in range(len(outs))]
        pieces = {"trellis": torch.empty(sum(tr), dtype=torch.int16),
                  "suh": torch.empty(len(outs), in_f, dtype=torch.float16),
                  "svh": torch.empty(sum(outs), dtype=torch.float16),
                  "mul1": torch.empty(len(outs), dtype=torch.int32)}
        for name, t in pieces.items():
            p = Parameter(t, requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, {"weight_loader": self._flat_loader(name, in_f, outs, k, tr_off, sv_off)})
        layer.exl3_flat = [(o, tr_off[i], sv_off[i]) for i, o in enumerate(outs)]
        layer.exl3_n = len(outs)
        layer.exl3_shape = (in_f, sum(outs))

    def _flat_loader(self, name: str, in_f: int, outs: list[int], k: int, tr_off: list[int], sv_off: list[int]):
        module = self.module

        def load(param: Parameter, w: torch.Tensor, shard_id=None, *args, **kwargs):
            if shard_id is None:
                raise ValueError(f"EXL3 dense {module}.{name}: a fused module's pieces need a shard id")
            i = int(shard_id)
            o = outs[i]
            if name == "trellis":
                want = (in_f // 16, o // 16, 16 * k)
                if w.dtype != torch.int16 or tuple(w.shape) != want:
                    raise ValueError(f"EXL3 dense {module}.trellis shard {i}: got {tuple(w.shape)} {w.dtype}, the "
                                     f"config's K={k} means {want} int16 (config and checkpoint disagree)")
                copy_from_host(param.data[tr_off[i]: tr_off[i] + w.numel()], w.reshape(-1))
            elif name in ("suh", "svh"):
                want = in_f if name == "suh" else o
                if w.dtype != torch.float16 or tuple(w.shape) != (want,):
                    raise ValueError(f"EXL3 dense {module}.{name} shard {i}: got {tuple(w.shape)} {w.dtype}, "
                                     f"want ({want},) fp16")
                copy_from_host(param.data[i] if name == "suh" else param.data[sv_off[i]: sv_off[i] + o], w)
            elif name == "mul1":
                copy_from_host(param.data[i: i + 1], w.reshape(-1)[:1].to(torch.int32))
            else:
                raise ValueError(f"EXL3 dense {module}: unexpected tensor {name}")
            return True
        return load

    def _loader(self, name: str, cut: Cut, local: bool, k: int, full_shape: tuple[int, int], stacked: bool = False):
        module = self.module

        def load(param: Parameter, w: torch.Tensor, shard_id=None, *args, **kwargs):
            if not local:
                return True
            slot = param.data[int(shard_id)] if stacked else param.data
            if stacked and shard_id is None:
                raise ValueError(f"EXL3 dense {module}.{name}: stacked parameter loaded without a shard id")
            if name == "trellis":
                want = (full_shape[0] // 16, full_shape[1] // 16, 16 * k)
                if w.dtype != torch.int16 or tuple(w.shape) != want:
                    raise ValueError(f"EXL3 dense {module}.trellis: got {tuple(w.shape)} {w.dtype}, the config's "
                                     f"K={k} means {want} int16 (config and checkpoint disagree)")
                src = cut.trellis(w)
            elif name in ("suh", "svh"):
                want = full_shape[0] if name == "suh" else full_shape[1]
                if w.dtype != torch.float16 or tuple(w.shape) != (want,):
                    raise ValueError(f"EXL3 dense {module}.{name}: got {tuple(w.shape)} {w.dtype}, want ({want},) fp16")
                src = cut.suh(w) if name == "suh" else cut.svh(w)
            elif name == "mul1":
                src = w.reshape(-1)[:1].to(torch.int32)
                slot = slot.view(-1)[:1]
            else:
                raise ValueError(f"EXL3 dense {module}: unexpected tensor {name}")
            if tuple(slot.shape) != tuple(src.shape):
                raise ValueError(f"EXL3 dense {module}.{name}: rank slice {tuple(src.shape)} vs param {tuple(slot.shape)}")
            copy_from_host(slot, src)
            return True
        return load

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from exllamav3.modules.quant.exl3 import LinearEXL3

        def lin(t, s_u, s_v, m, in_f, out_f):
            return LinearEXL3(None, in_f, out_f, suh=s_u.contiguous(), svh=s_v.contiguous(), trellis=t.contiguous(),
                              mul1=m.reshape(()).contiguous())
        if self.module == "attn.wo_a":
            in_f, out_f = layer.exl3_shape
            lins = [lin(layer.slice[g].trellis, layer.slice[g].suh, layer.slice[g].svh, layer.slice[g].mul1, in_f, out_f)
                    for g in layer.exl3_groups]
            layer._exl3 = lins

            mrow = self._mrow_group([(layer.slice[g].trellis, layer.slice[g].suh, layer.slice[g].svh, i * in_f, i * out_f)
                                     for i, g in enumerate(layer.exl3_groups)])

            grouped_in_place = _mrow_grouped()

            def grouped_bmm(o_rot: torch.Tensor) -> torch.Tensor:          # [T, G_loc, in] -> [T, G_loc, out]
                t, g_loc = o_rot.shape[0], o_rot.shape[1]
                if mrow is not None and 0 < t <= MROW_MAX_ROWS and o_rot.dtype == torch.bfloat16:
                    y = torch.empty((t, g_loc * out_f), dtype=torch.bfloat16, device=o_rot.device)
                    if (grouped_in_place and o_rot.stride(2) == 1 and o_rot.stride(0) % 4 == 0 and o_rot.stride(1) % 4 == 0
                            and o_rot.data_ptr() % 16 == 0):
                        return mrow(o_rot, y).view(t, g_loc, out_f)
                    x2 = o_rot.reshape(t, g_loc * in_f)
                    if x2.stride(1) != 1 or x2.data_ptr() % 16:  # the kernel reads 4-element vectors, row-major
                        x2 = x2.contiguous().clone()
                    return mrow(x2, y).view(t, g_loc, out_f)
                if _PROMPT_NOCOPY and t > PROMPT_ROWS:
                    og = o_rot.transpose(0, 1)                 # [G_loc, T, in]
                    if og.is_contiguous():
                        # the inverse-RoPE kernel's (or patch 0035's qdq's) group-major buffer: one cast, contiguous
                        # slabs; .to() returns og itself when it already is fp16 (patch 0035's VLLM_DSV41_WOA_QDQ_FP16)
                        xg = og.to(torch.float16)
                        srcs = [xg[i] for i in range(g_loc)]
                    else:
                        xs = o_rot.to(torch.float16)
                        srcs = [xs[:, i].contiguous() for i in range(g_loc)]
                    out_dtype = torch.bfloat16 if o_rot.dtype == torch.float16 else o_rot.dtype
                    y = torch.empty((t, g_loc, out_f), dtype=out_dtype, device=o_rot.device)
                    for i, g_lin in enumerate(lins):
                        y[:, i].copy_(g_lin.forward(srcs[i], {}, out_dtype=torch.float32))
                    return y
                xs = o_rot.to(torch.float16)
                return torch.stack([g_lin.forward(xs[:, i].contiguous(), {}, out_dtype=torch.float32)
                                    for i, g_lin in enumerate(lins)], 1).to(o_rot.dtype)
            layer.grouped_bmm = grouped_bmm
            if self.prefetch:
                l2pf.register(self.layer_idx, self.module, mrow.trellis if mrow is not None else [])
            return
        if getattr(layer, "exl3_flat", None):
            in_f = layer.exl3_shape[0]
            parts = [(layer.trellis[t_off: t_off + (in_f // 16) * (o // 16) * 16 * self.k]
                      .view(in_f // 16, o // 16, 16 * self.k), layer.suh[i], layer.svh[s_off: s_off + o])
                     for i, (o, t_off, s_off) in enumerate(layer.exl3_flat)]
            layer._exl3 = [lin(t, su, sv, layer.mul1[i], in_f, o)
                           for i, ((t, su, sv), (o, _, _)) in enumerate(zip(parts, layer.exl3_flat))]
            col = [sum(o for o, _, _ in layer.exl3_flat[:i]) for i in range(len(parts))]
            layer._mrow = self._mrow_group([(t, su, sv, 0, c) for (t, su, sv), c in zip(parts, col)])
            if self.prefetch:
                l2pf.register(self.layer_idx, self.module, layer._mrow.trellis if layer._mrow is not None else [])
            return
        in_f, out_f = layer.exl3_shape
        if layer.exl3_n == 1:
            layer._exl3 = [lin(layer.trellis, layer.suh, layer.svh, layer.mul1, in_f, out_f)]
            layer._mrow = self._mrow_group([(layer.trellis, layer.suh, layer.svh, 0, 0)])
        else:
            layer._exl3 = [lin(layer.trellis[i], layer.suh[i], layer.svh[i], layer.mul1[i], in_f, out_f)
                           for i in range(layer.exl3_n)]
            layer._mrow = self._mrow_group([(layer.trellis[i], layer.suh[i], layer.svh[i], 0, i * out_f)
                                            for i in range(layer.exl3_n)])
        if self.prefetch:
            l2pf.register(self.layer_idx, self.module, layer._mrow.trellis if layer._mrow is not None else [])

    def _mrow_group(self, parts):
        """One exl3_mrow launch for the module's matrices on one input, or None (disabled, unsupported K or shape)."""
        mode = _mrow_mode()
        if mode == "0" or (mode == "groups" and len(parts) < 2) or not 2 <= self.k <= 6:
            return None
        if any(t.shape[0] % 8 or t.shape[1] % 8 for t, _, _, _, _ in parts):   # k and n must be multiples of 128
            return None
        from .mrow import MrowGroup
        g = MrowGroup([(t.contiguous(), su.contiguous(), sv.contiguous(), xc, oc) for t, su, sv, xc, oc in parts], self.k)
        logger.debug_once("EXL3 dense: %s runs as one exl3_mrow launch at <= %d rows", self.module, MROW_MAX_ROWS)
        return g

    # ----------------------------------------------------------------------------------------------- forward
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.module == "attn.wo_a":
            raise RuntimeError("EXL3 wo_a runs through grouped_bmm (patch 0006), not as a plain linear")
        mrow = getattr(layer, "_mrow", None)
        rows = x.numel() // x.shape[-1]
        if mrow is not None and 0 < rows <= MROW_MAX_ROWS and x.dtype == torch.bfloat16:
            x2 = x.reshape(rows, x.shape[-1]).contiguous()
            if x2.data_ptr() % 16:                       # the kernel reads 4-element vectors
                x2 = x2.clone()
            y = torch.empty((rows, sum(m.out_features for m in layer._exl3)), dtype=torch.bfloat16, device=x.device)
            y = mrow(x2, y).reshape(*x.shape[:-1], -1)
            if self.prefetch and self.module in _L2PF_AFTER:
                l2pf.point(_L2PF_AFTER[self.module], self.layer_idx, rows)
            return y if bias is None else y + bias
        xs = x.reshape(-1, x.shape[-1])
        sc = None
        if self.module == "main_proj":
            # the target's raw residual stream (|x| up to ~1.6e5) overflows exllamav3's fp16 input: divide by a power
            # of two that brings the max under 2^14 (exact in any float format), multiply the output back (linear)
            sc = torch.exp2(torch.ceil(torch.log2(xs.abs().amax().float().clamp_min(1.0) / 16384.0)).clamp_min(0.0))
            xs = xs.float() / sc
        xs = xs.to(torch.float16).contiguous()
        outs = [m.forward(xs, {}, out_dtype=torch.float32) for m in layer._exl3]
        if _PROMPT_NOCOPY and len(outs) > 1 and sc is None and rows > PROMPT_ROWS:
            # patch 0035: each part's fp32 output straight into its column block of the result (no fp32 cat)
            y = torch.empty((rows, sum(o.shape[-1] for o in outs)), dtype=x.dtype, device=x.device)
            c = 0
            for o in outs:
                y[:, c:c + o.shape[-1]].copy_(o)
                c += o.shape[-1]
            y = y.reshape(*x.shape[:-1], -1)
            return y if bias is None else y + bias
        y = outs[0] if len(outs) == 1 else torch.cat(outs, -1)
        if sc is not None:
            y = y * sc
        y = y.to(x.dtype).reshape(*x.shape[:-1], -1)
        return y if bias is None else y + bias
