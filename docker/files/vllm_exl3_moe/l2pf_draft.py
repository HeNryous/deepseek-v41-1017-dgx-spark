# SPDX-License-Identifier: Apache-2.0
"""dsv41 patch 0066: paced L2 prefetch inside the DSpark drafter pass (VLLM_DSV41_DRAFT_L2PF, default off).

l2pf.py's chain serves the backbone only: the drafter's dense modules are built with prefetch=False (config.py),
because a drafter layer (index >= num_hidden_layers) in that chain would move its "last layer", so the backbone's last
MoE would fork a prefetch of drafter weights that nothing inside the captured target forward joins (a graph that ends
with side-stream work does not capture), and the drafter's own pass is a separate graph. This module is a second,
independent chain over the drafter's three blocks, with the kernel, chunking and pacing of l2pf.py (csrc/exl3_l2pf.cu,
TensorFold's design): a point first joins the previous prefetch, then forks the next one on its own side stream; the
last block's "next" point and the end of the drafter forward join without forking, so the drafter graph ends with
nothing in flight. TensorFold's batched drafter graph forks the same way (dspark.py BatchDraftGraph, l2_fork).

Points (comma list in VLLM_DSV41_DRAFT_L2PF):
  next  after block j's MoE -> block j+1's mHC attention mix (EXL3_L2PF_HC=1), fused wq_a+wkv and wq_b. Window: the
        MoE all-reduce, the mHC boundary and block j+1's hc_pre (no DRAM reader of size runs there; ~60-100 us at one
        stream in the F15 trace).
  attn  after block j's wq_b -> block j's wo_a (FP8 Marlin, 4 head groups) and wo_b. Window: Q/KV insert and the sparse
        attention (~22 us); the rest streams under the wo_a GEMMs, which read the same bytes in the same order.
Why it does not compete with the target MoE: the drafter graph runs after the target forward and the sampler on the
same stream, its forks wait for the drafter's own preceding kernels, and everything is joined before the drafter graph
ends; the target's MoE readers (patch 0063's 4-stream slowdown) never run at the same time. Inside the drafter the
"next" fork starts after the drafter MoE (it waits for it), so it does not share DRAM with the drafter's experts
either. A prefetch writes nothing: every output keeps its bits.
EXL3_DRAFT_L2PF_MAX_ROWS (default 16, l2pf.MAX_ROWS): drafter passes with more rows (4 streams = 20) prefetch nothing.
Budget, pacing and delay: EXL3_L2PF_MB / EXL3_L2PF_RATE_GBPS / EXL3_L2PF_DELAY_NS as for the backbone chain.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

POINTS = {p for p in os.environ.get("VLLM_DSV41_DRAFT_L2PF", "0").replace(" ", "").split(",")
          if p and p != "0"} & {"attn", "next"}
MAX_ROWS = int(os.environ.get("EXL3_DRAFT_L2PF_MAX_ROWS") or 16)
HC = os.environ.get("EXL3_L2PF_HC", "0") == "1"

# What each point prefetches: (block offset, module) in the order the drafter reads them.
PLAN = {
    "attn": ((0, "attn.wo_a"), (0, "attn.wo_b")),
    "next": ((1, "hc_attn_fn"),) * HC + ((1, "attn.fused_wqa_wkv"), (1, "attn.wq_b")),
}

_reg: dict[tuple[int, str], list[torch.Tensor]] = {}
_first_layer = 1 << 30
_last_layer = -1
_stream: torch.cuda.Stream | None = None
_pending = False
_announced = False


def enabled() -> bool:
    return bool(POINTS)


def register(layer: int, module: str, tensors) -> None:
    """The weight tensors a drafter module streams, in the order its kernel reads them (largest first for FP8)."""
    global _first_layer, _last_layer
    if not enabled():
        return
    _reg[(layer, module)] = [t for t in tensors if t is not None and t.is_cuda and t.is_contiguous()]
    _first_layer = min(_first_layer, layer)
    _last_layer = max(_last_layer, layer)


def register_drafter(model, first_layer: int) -> None:
    """After the drafter's weights load (DSparkDeepseekV4ForCausalLM): each block's wo_a (FP8, not an EXL3 module,
    so dense.py does not register it) and, with EXL3_L2PF_HC=1, its mHC attention mix. Blocks are numbered
    first_layer + i, the layer index their EXL3 modules carry."""
    if not enabled():
        return
    n = 0
    for i, layer in enumerate(model.layers):
        idx = first_layer + i
        wo_a = getattr(getattr(layer, "attn", None), "wo_a", None)
        if wo_a is not None:
            ts = sorted((p.data for p in wo_a.parameters() if p.is_cuda), key=lambda t: -t.numel() * t.element_size())
            register(idx, "attn.wo_a", ts)
            n += 1
        t = getattr(layer, "hc_attn_fn", None)
        if HC and t is not None:
            register(idx, "hc_attn_fn", [t.data])
            n += 1
    logger.info("dsv41 0066 drafter L2 prefetch: points %s, %d drafter tensors registered (blocks %d-%d), rows <= %d",
                ",".join(sorted(POINTS)), n, first_layer, first_layer + len(model.layers) - 1, MAX_ROWS)


def _ranges(layer: int, point: str) -> tuple[list[int], list[int]]:
    from . import l2pf

    ptrs, sizes, left = [], [], l2pf.BUDGET
    for off, module in PLAN[point]:
        for t in _reg.get((layer + off, module), ()):
            n = min(t.numel() * t.element_size(), left) // 16 * 16
            p = t.data_ptr()
            if n <= 0 or p % 16:
                continue
            ptrs.append(p)
            sizes.append(n)
            left -= n
            if left < 16:
                break
    return ptrs[:24], sizes[:24]                   # PMAX in csrc/exl3_l2pf.cu


def join() -> None:
    global _pending
    if _pending:
        torch.cuda.current_stream().wait_stream(_stream)
        _pending = False


def point(name: str, layer: int, rows: int) -> None:
    """Prefetch point `name` after drafter block `layer` (a drafter pass with `rows` rows); no-op when off."""
    global _stream, _pending, _announced
    if not POINTS or layer < _first_layer:
        return                                     # off, or a backbone layer (moe_method.py calls every MoE layer)
    join()
    if name not in POINTS or not 0 < rows <= MAX_ROWS:
        return
    if name == "next" and layer >= _last_layer:
        return                                     # the last block: the drafter pass ends with nothing in flight
    ptrs, sizes = _ranges(layer, name)
    if not ptrs:
        return
    from . import l2pf

    cur = torch.cuda.current_stream()
    if _stream is None:
        _stream = torch.cuda.Stream()
    _stream.wait_stream(cur)
    with torch.cuda.stream(_stream):
        wave_ns = int(128 * l2pf.CHUNK / l2pf.RATE_GBPS) if l2pf.RATE_GBPS > 0 else 0
        l2pf.ext().l2pf(ptrs, sizes, l2pf.CHUNK, 4, l2pf.DELAY_NS, wave_ns)
    _pending = True
    if not _announced:
        _announced = True
        logger.info("dsv41 0066 drafter L2 prefetch on: %d MiB a fork, %.0f GB/s pacing, %d tensors registered",
                    l2pf.BUDGET >> 20, l2pf.RATE_GBPS, sum(len(v) for v in _reg.values()))
