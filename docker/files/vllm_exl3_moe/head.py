# SPDX-License-Identifier: Apache-2.0
"""Opt-in MXFP8 lm_head for DeepSeek-V4.1 (`exl3_moe.head: "mxfp8"` or VLLM_EXL3_HEAD=mxfp8; needs patch 0005, which
hands the head to the quant config).

The checkpoint's head is BF16 (vocab 129,280 x hidden 5,120): 0.66 GB per forward per rank at TP=2, read by every
target forward and every DSpark draft forward (the drafter shares the module), and cuBLAS BF16 at decode sizes runs well
under the GB10's 273 GB/s. At load the head is quantized with vLLM's own MXFP8 quantizer (e4m3, one power-of-two
scale per 32 inputs, the format of the checkpoint's other linears) and Marlin-packed; apply runs one Marlin MXFP8
(W8A16) GEMM, so activations stay BF16 and the only change is the weight rounding. Its quality cost is measured on the
KLD panel first (scripts/quantize/p4_kld.py --head-mxfp8).
"""

from __future__ import annotations

import os

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

from .dense import MROW_MAX_ROWS
from .hostcopy import copy_from_host

# dsv41 patch 0046: the head's own row cap for exl3_mrow (with VLLM_EXL3_HEAD_MROW=1). mb_dense_rows (19.10.2026, K6
# head, 5120 x 64640 a rank): mrow 994 / 1052 us at 1 / 6 rows against exl3 1113 / 1160 us, but 1257-1308 us against
# 1130-1136 us at 12-16 rows; so mrow only up to EXL3_HEAD_MROW_MAX_ROWS rows (default MROW_MAX_ROWS = as before).
HEAD_MROW_MAX_ROWS = min(MROW_MAX_ROWS, int(os.environ.get("EXL3_HEAD_MROW_MAX_ROWS") or MROW_MAX_ROWS))

logger = init_logger(__name__)


class Mxfp8MarlinHeadMethod(UnquantizedEmbeddingMethod):
    """Loads the BF16 head like the unquantized method, then keeps only an MXFP8 Marlin copy."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_mxfp8_layer_for_marlin
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

        w = layer.weight.data                                  # [vocab per rank (padded), hidden] bf16
        n, k = w.shape
        q, s = mxfp8_e4m3_quantize(w.to(torch.bfloat16), is_sf_swizzled_layout=False)
        holder = torch.nn.Module()
        holder.weight = Parameter(q, requires_grad=False)
        holder.weight_scale = Parameter(s, requires_grad=False)
        holder.output_size_per_partition, holder.input_size_per_partition = n, k
        prepare_mxfp8_layer_for_marlin(holder)
        layer.mxfp8_weight = Parameter(holder.weight.data, requires_grad=False)
        layer.mxfp8_scale = Parameter(holder.weight_scale.data, requires_grad=False)
        layer.mxfp8_nk = (n, k)
        # Drop the BF16 copy (-0.66 GB per rank). Only the logits path reads this module (DSpark shares it without
        # touching .weight); an empty tensor makes any other reader fail loudly instead of reading stale data.
        layer.weight = Parameter(torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False)
        logger.info("lm_head: MXFP8 Marlin (%d x %d per rank)", n, k)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_mxfp8_marlin_linear

        n, k = layer.mxfp8_nk
        out = apply_mxfp8_marlin_linear(input=x, weight=layer.mxfp8_weight, weight_scale=layer.mxfp8_scale,
                                        workspace=None, size_n=n, size_k=k)
        return out if bias is None else out + bias

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("the MXFP8 head is an output projection only (V4.1 does not tie embed and head)")


class Exl3HeadMethod(UnquantizedEmbeddingMethod):
    """lm_head stored as EXL3 (`exl3_moe.head: "exl3"`, `head_k`; needs patches 0005 + 0007): checkpoint tensors
    head.{trellis,suh,svh,mul1} hold the full [hidden -> vocab] matrix and each rank keeps its vocab range, a column cut
    at whole 128-wide Hadamard blocks (V4.1: 129,280 = 2 x 505 x 128). At K6 it reads 6 bits per weight (MXFP8: 8.25)
    and stays within ~0.4% of the BF16 head (Hessian-aware encode from the final hidden states, p4_dense.py --head)."""

    def __init__(self, k: int):
        self.k = int(k)

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        extra_weight_attrs.pop("weight_loader", None)
        n_loc = output_partition_sizes[0]
        si = getattr(layer, "shard_indices", None)
        lo = si.org_vocab_start_index if si is not None else int(getattr(layer, "tp_rank", 0)) * n_loc
        hi = si.org_vocab_end_index if si is not None else lo + n_loc
        if lo % 128 or (hi - lo) % 128 or hi - lo != n_loc:
            raise ValueError(f"EXL3 head: vocab range [{lo}, {hi}) of {n_loc} per rank is not whole 128-wide blocks "
                             f"(padding or uneven vocab split is not supported)")
        k, d = self.k, input_size_per_partition
        shapes = {"trellis": ((d // 16, n_loc // 16, 16 * k), torch.int16), "suh": ((d,), torch.float16),
                  "svh": ((n_loc,), torch.float16), "mul1": ((1,), torch.int32)}
        for name, (shape, dt) in shapes.items():
            p = Parameter(torch.empty(*shape, dtype=dt), requires_grad=False)
            layer.register_parameter(name, p)
            p.weight_loader = self._loader(name, lo, hi, d, k)
        layer.exl3_shape = (d, n_loc)

    @staticmethod
    def _loader(name: str, lo: int, hi: int, d: int, k: int):
        def load(param, w, *args, **kwargs):
            if name == "trellis":
                if w.dtype != torch.int16 or w.dim() != 3 or w.shape[0] != d // 16 or w.shape[2] != 16 * k:
                    raise ValueError(f"EXL3 head trellis {tuple(w.shape)} {w.dtype}: want [{d // 16}, vocab/16, {16 * k}] "
                                     f"int16 (config head_k={k})")
                src = w.narrow(1, lo // 16, (hi - lo) // 16)
            elif name == "suh":
                src = w
            elif name == "svh":
                src = w[lo:hi]
            else:
                src = w.reshape(-1)[:1].to(torch.int32)
            if tuple(src.shape) != tuple(param.shape):
                raise ValueError(f"EXL3 head {name}: rank slice {tuple(src.shape)} vs param {tuple(param.shape)}")
            copy_from_host(param.data, src)
            return True
        return load

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from exllamav3.modules.quant.exl3 import LinearEXL3

        d, n = layer.exl3_shape
        layer._exl3 = LinearEXL3(None, d, n, suh=layer.suh, svh=layer.svh, trellis=layer.trellis,
                                 mul1=layer.mul1.reshape(()))
        # VLLM_EXL3_HEAD_MROW=1: decode sizes run as one exl3_mrow launch (bf16 in and out), the kernel of every dense
        # module; the head is read by every target forward and every DSpark draft forward.
        layer._mrow = None
        if os.environ.get("VLLM_EXL3_HEAD_MROW", "0") == "1" and 2 <= self.k <= 6 and d % 128 == 0 and n % 128 == 0:
            from .mrow import MrowGroup
            layer._mrow = MrowGroup([(layer.trellis.data, layer.suh.data, layer.svh.data, 0, 0)], self.k)
        logger.info("lm_head: EXL3 K%d (%d x %d per rank)%s", self.k, d, n,
                    f", exl3_mrow at <= {HEAD_MROW_MAX_ROWS} rows" if layer._mrow else "")

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        mrow = getattr(layer, "_mrow", None)
        if mrow is not None and 0 < rows <= HEAD_MROW_MAX_ROWS and x.dtype == torch.bfloat16:
            x2 = x.reshape(rows, x.shape[-1]).contiguous()
            if x2.data_ptr() % 16:
                x2 = x2.clone()
            y = mrow(x2, torch.empty((rows, layer.exl3_shape[1]), dtype=torch.bfloat16, device=x.device))
            y = y.reshape(*x.shape[:-1], -1)
            return y if bias is None else y + bias
        xs = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
        y = layer._exl3.forward(xs, {}, out_dtype=torch.float32).to(x.dtype).reshape(*x.shape[:-1], -1)
        return y if bias is None else y + bias

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("the EXL3 head is an output projection only (V4.1 does not tie embed and head)")
