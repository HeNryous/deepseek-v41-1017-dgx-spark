# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel all-reduce over RoCE RDMA writes for decode-size bf16 tensors (csrc/exl3_rdma.cu).

vLLM's TP all-reduce between two GB10 nodes goes through NCCL's network proxy: ~25 us median per decode all-reduce in
our serve, ~96 a decode step. Here a staging kernel puts the tensor in registered host memory and rings a doorbell, a
pinned CPU thread RDMA-writes it into the peer's receive ring, and a summing kernel waits for the peer's flag and adds
the ranks' parts in rank order (fp32, one rounding), so both ranks hold the same bits. The kernel pair captures in CUDA
graphs like any other. The design, the rings and the proxy are TensorFold's one-hop gather (csrc/NOTICE.tensorfold);
the sum and the vLLM hook are ours.

EXL3_RDMA_AR=1 turns it on for TP groups of 2+ ranks (default off until measured): bf16 tensors up to
EXL3_RDMA_AR_MAX_KB (256) go over RDMA, everything else stays on NCCL. Devices: EXL3_RDMA_DEVICES (comma list), else
NCCL_IB_HCA's list; GID index: EXL3_RDMA_GID, else NCCL_IB_GID_INDEX; EXL3_RDMA_CPU pins the proxy thread to a core (default: the last).
EXL3_RDMA_FLAGW=1 (the same on every rank): each part's data write is chained with an 8-byte flag write the peer's GPU
polls, so no receiving thread is on the path (csrc/exl3_rdma.cu, P2 stage 0; default off until tested on both nodes).
The setup is collective: the ranks exchange their queue pairs over the group's CPU group on the first eligible
all-reduce outside a graph capture, and if any rank cannot open RDMA, every rank stays on NCCL.

dsv41 patch 0045, EXL3_RDMA_AG=1 (default off; the same on every rank): vLLM's TP all-gathers of up to
EXL3_RDMA_AG_MAX_KB (256) a rank also go over the ring (csrc/exl3_rdma.cu all_gather: the same staging, writes and
flags, a copying kernel instead of the sum), any dtype, any dim, the bytes NCCL plus vLLM's movedim give. In decode
these are the drafter's Markov (value, index) pairs (5 a round, nvidia/dspark.py draft_argmax_local), the Engram rows
(common/engram.py Engram.embed) and, with EXL3_RDMA_AG_MAX_KB >= 3200, the verify logits ([rows, vocab / 2] bf16,
LogitsProcessor._gather_logits; 776 KB at 6 rows, 3.1 MB at 24). Without it each of them is an NCCL call: in a CUDA
graph a host node plus NCCL's proxy thread, which sleeps on a condition variable whenever no NCCL op is pending (all
decode all-reduces run on the ring), so each wakes from an idle core. The path depends on dtype, shape and dim only
(the same on every rank); misaligned inputs are copied first (a local decision). The ring's slots grow to
max(EXL3_RDMA_AR_MAX_KB, EXL3_RDMA_AG_MAX_KB); the all-reduce threshold stays EXL3_RDMA_AR_MAX_KB.
"""

from __future__ import annotations

import math
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_BYTES = int(os.environ.get("EXL3_RDMA_AR_MAX_KB") or 256) * 1024
SLOTS = int(os.environ.get("EXL3_RDMA_AR_SLOTS") or 4)
FLAGW = os.environ.get("EXL3_RDMA_FLAGW", "0") == "1"
# dsv41 patch 0045: all-gathers over the ring
AG = os.environ.get("EXL3_RDMA_AG", "0") == "1"
AG_MAX_BYTES = int(os.environ.get("EXL3_RDMA_AG_MAX_KB") or 256) * 1024
RING_BYTES = -(-max(MAX_BYTES, AG_MAX_BYTES if AG else 0) // 64) * 64    # slot size (a multiple of 64)

_ext = None
_rings: dict[str, "RdmaAllReduce | None"] = {}      # communicator unique_name -> ring (None: NCCL for good)


def enabled() -> bool:
    return os.environ.get("EXL3_RDMA_AR", "0") == "1" or AG


def ext():
    global _ext
    if _ext is None:
        from .jit import load
        _ext = load("exl3_rdma", "EXL3_RDMA_FLAGS", ldflags=["-libverbs"])
    return _ext


def device_names() -> list[str]:
    named = os.environ.get("EXL3_RDMA_DEVICES") or os.environ.get("NCCL_IB_HCA") or ""
    return [d.strip().lstrip("=^").split(":")[0] for d in named.split(",") if d.strip()]


class RdmaAllReduce:
    def __init__(self, cpu_group, rank: int, world: int):
        import torch.distributed as dist
        e = ext()
        # EXL3_RDMA_GID overrides NCCL's index: on our Sparks only rocep1s0f0 carries the IPv4 GID (index 3), while
        # index 1 (RoCE v2, link-local) exists on both PCIe functions of the port, so two devices need 1
        gid = int(os.environ.get("EXL3_RDMA_GID") or os.environ.get("NCCL_IB_GID_INDEX") or 3)
        cpu = int(os.environ.get("EXL3_RDMA_CPU") or (os.cpu_count() or 1) - 1)
        devices = device_names()
        self.rank, self.world = rank, world
        error, info = "", None
        try:
            if not devices:
                raise ValueError("no RDMA device named (EXL3_RDMA_DEVICES / NCCL_IB_HCA)")
            self.h = e.create(rank, world, RING_BYTES, SLOTS, devices, gid, cpu, int(FLAGW))
            info = e.local_info(self.h)
        except Exception as exc:  # noqa: BLE001  (every rank refuses together)
            error = f"{type(exc).__name__}: {exc}"
        infos = [None] * world
        dist.all_gather_object(infos, (error, info, devices), group=cpu_group)
        bad = [f"rank {r}: {x[0]}" for r, x in enumerate(infos) if x[0]]
        if not bad and len({len(x[2]) for x in infos}) != 1:
            bad = ["ranks open different device counts"]
        if bad:
            raise RuntimeError("; ".join(bad))
        try:
            e.connect(self.h, [infos[r][1][rank] for r in range(world)])
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        ups = [None] * world
        dist.all_gather_object(ups, error, group=cpu_group)    # every QP can receive before anyone writes
        bad = [f"rank {r}: {u}" for r, u in enumerate(ups) if u]
        if bad:
            raise RuntimeError("; ".join(bad))
        e.start(self.h)
        import atexit
        atexit.register(e.stop, self.h)
        logger.info("EXL3 RDMA all-reduce: rank %d of %d on %s, gid %d, proxy on cpu %d, up to %d KB "
                    "(slots %d KB, all-gathers %s)", rank, world, ",".join(devices), gid, cpu, MAX_BYTES // 1024,
                    RING_BYTES // 1024, f"up to {AG_MAX_BYTES // 1024} KB" if AG else "off")

    @staticmethod
    def fits(x: torch.Tensor) -> bool:
        # dtype and size only: the ranks must take the same path for every call (alignment is fixed up below)
        return x.dtype == torch.bfloat16 and x.is_cuda and x.numel() % 8 == 0 and 0 < x.numel() * 2 <= MAX_BYTES

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous() or x.data_ptr() % 16:
            x = x.contiguous().clone()
        out = torch.empty_like(x)
        ext().all_reduce(self.h, x, out)
        return out

    # dsv41 patch 0045
    @staticmethod
    def fits_gather(x: torch.Tensor, dim: int) -> bool:
        # dtype, shape and dim only: every rank takes the same path
        if not (AG and x.is_cuda and x.dim() > 0 and x.numel() > 0):
            return False
        if dim < 0:
            dim += x.dim()
        nbytes = x.numel() * x.element_size()
        outer = math.prod(x.shape[:dim])
        return nbytes <= AG_MAX_BYTES and (nbytes // outer) % 4 == 0

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """torch.cat of every rank's x along dim, in rank order (what CudaCommunicator.all_gather returns)."""
        if dim < 0:
            dim += x.dim()
        if not x.is_contiguous() or x.data_ptr() % 16:
            x = x.contiguous().clone()
        shape = list(x.shape)
        shape[dim] *= self.world
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        ext().all_gather(self.h, x, out, math.prod(x.shape[:dim]))
        return out


def ring_for(comm) -> RdmaAllReduce | None:
    """The ring of a vLLM device communicator (built on its first eligible call), or None (NCCL)."""
    name = getattr(comm, "unique_name", "") or ""
    if name in _rings:
        return _rings[name]
    if not name.startswith("tp") or getattr(comm, "world_size", 1) < 2:
        _rings[name] = None
        return None
    if torch.cuda.is_current_stream_capturing():
        return None                                   # set up on an eager call (both ranks reach the same one)
    try:
        r = RdmaAllReduce(comm.cpu_group, comm.rank_in_group, comm.world_size)
    except Exception as exc:  # noqa: BLE001
        logger.warning("EXL3 RDMA all-reduce off for %s, NCCL stays: %s", name, exc)
        r = None
    _rings[name] = r
    return r


def install() -> None:
    """Route vLLM's CUDA communicator all-reduce (EXL3_RDMA_AR) and all-gather (EXL3_RDMA_AG, 0045) through the
    ring for the tensors it fits."""
    if not enabled():
        return
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    if getattr(CudaCommunicator.all_reduce, "_exl3_rdma", False):
        return
    orig = CudaCommunicator.all_reduce

    def all_reduce(self, input_):
        if RdmaAllReduce.fits(input_):
            r = ring_for(self)
            if r is not None:
                return r.all_reduce(input_)
        return orig(self, input_)

    all_reduce._exl3_rdma = True
    if os.environ.get("EXL3_RDMA_AR", "0") == "1":
        CudaCommunicator.all_reduce = all_reduce
        logger.info("EXL3 RDMA all-reduce: vLLM's TP all-reduce routed for bf16 tensors up to %d KB",
                    MAX_BYTES // 1024)
    if AG and not getattr(CudaCommunicator.all_gather, "_exl3_rdma", False):
        orig_ag = CudaCommunicator.all_gather

        def all_gather(self, input_, dim=-1):
            if RdmaAllReduce.fits_gather(input_, dim):
                r = ring_for(self)
                if r is not None:
                    return r.all_gather(input_, dim)
            return orig_ag(self, input_, dim)

        all_gather._exl3_rdma = True
        CudaCommunicator.all_gather = all_gather
        logger.info("EXL3 RDMA all-gather (0045): vLLM's TP all-gather routed for tensors up to %d KB",
                    AG_MAX_BYTES // 1024)
