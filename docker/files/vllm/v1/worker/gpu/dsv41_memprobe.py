# SPDX-License-Identifier: Apache-2.0
"""dsv41 patch 0076a (Entwurf 22.10.2026): memory probe for the head-node budget (measurement windows only).

VLLM_DSV41_MEMPROBE=/root/.cache/memprobe  (prefix; empty = off, the default: every hook is one module-level bool test)
writes one JSON line per logged step to <prefix>_r<rank>.jsonl from each worker:
  step  : the step kind (prefill / mixed / decode / dummy), tokens, requests, torch allocated / reserved / peak since the
          last logged step (MiB), allocator retries, MemAvailable and this process's RSS (MiB)
  phase : with VLLM_DSV41_MEMPROBE_LAYERS="2,14,20" the attention / ffn / engram phases of those decoder layers in eager
          steps of at least VLLM_DSV41_MEMPROBE_MIN_TOKENS rows (default 512): allocated before, peak and net growth of
          the phase (MiB) - which phase of a prefill step sets the activation peak the caching allocator keeps.
  engram: in every logged step record, per Engram disk table the reads served since the last record, their summed and
          the running maximum serve time (ms) from the reader's own counters (engram_disk spin_stats): how long a
          step's lookups took against how long the layers before them ran (layer 1's lookup is waited for after layer 0)
Logged: every step of >= MIN_TOKENS tokens, the first 64 steps, then every 512th step; at most VLLM_DSV41_MEMPROBE_MAX
lines (default 20000). Host bookkeeping only (torch's allocator statistics, /proc reads), no device sync, never under
CUDA graph capture. Numerics untouched.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

import torch

_PREFIX = os.environ.get("VLLM_DSV41_MEMPROBE", "").strip()
ENABLED = bool(_PREFIX)
_LAYERS = {int(x) for x in os.environ.get("VLLM_DSV41_MEMPROBE_LAYERS", "").replace(" ", "").split(",") if x}
_MIN_TOKENS = int(os.environ.get("VLLM_DSV41_MEMPROBE_MIN_TOKENS", "512") or 512)
_MAX_LINES = int(os.environ.get("VLLM_DSV41_MEMPROBE_MAX", "20000") or 20000)
_MIB = 1 << 20
_state = {"f": None, "lines": 0, "steps": 0, "peak": 0, "eng": None, "eng_last": None}


def _file():
    f = _state["f"]
    if f is None:
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except Exception:  # noqa: BLE001
            pass
        f = _state["f"] = open(f"{_PREFIX}_r{rank}.jsonl", "a", buffering=1)
    return f


def _meminfo_mib() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                if k in ("MemAvailable", "MemFree", "Cached", "AnonPages", "Shmem", "SUnreclaim", "Mlocked"):
                    out[k] = int(v.split()[0]) // 1024
        with open("/proc/self/statm") as fh:
            out["RSS"] = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // _MIB
    except OSError:
        pass
    return out


def _emit(rec: dict) -> None:
    if _state["lines"] >= _MAX_LINES:
        return
    _state["lines"] += 1
    rec["t"] = round(time.time(), 3)
    _file().write(json.dumps(rec) + "\n")


def _engram_tables(model):
    if _state["eng"] is None:
        tabs = []
        try:
            for mod in model.modules():
                t = getattr(mod, "_disk", None)
                if t is not None and hasattr(t, "spin_stats"):
                    tabs.append(t)
        except Exception:  # noqa: BLE001
            pass
        _state["eng"] = tabs
        _state["eng_last"] = [list(t.spin_stats()) for t in tabs]
    return _state["eng"]


def _engram_delta(model) -> list:
    tabs = _engram_tables(model) if model is not None else []
    out = []
    for i, t in enumerate(tabs):
        cur = list(t.spin_stats())                 # served, serve_ns, serve_max_ns, errors
        last = _state["eng_last"][i]
        out.append([cur[0] - last[0], round((cur[1] - last[1]) / 1e6, 2), round(cur[2] / 1e6, 2), cur[3]])
        _state["eng_last"][i] = cur
    return out


def step(kind: str, num_tokens: int, num_reqs: int, model=None) -> None:
    """End of a worker step (after the forward is enqueued)."""
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return
    n = _state["steps"] = _state["steps"] + 1
    peak = max(_state["peak"], torch.cuda.max_memory_allocated())
    if num_tokens >= _MIN_TOKENS or n <= 64 or n % 512 == 0:
        stats = torch.cuda.memory_stats()
        rec = {"ev": "step", "n": n, "kind": kind, "toks": int(num_tokens), "reqs": int(num_reqs),
               "alloc": torch.cuda.memory_allocated() // _MIB, "reserved": torch.cuda.memory_reserved() // _MIB,
               "peak": peak // _MIB, "retries": int(stats.get("num_alloc_retries", 0)),
               "segments": int(stats.get("segment.all.current", 0))}
        rec.update(_meminfo_mib())
        eng = _engram_delta(model)
        if eng:
            rec["eng"] = eng
        _emit(rec)
        torch.cuda.reset_peak_memory_stats()
        _state["peak"] = 0
    else:
        _state["peak"] = peak


@contextmanager
def phase(layer_idx: int, tag: str, num_tokens: int):
    """Peak and net growth of one phase of a selected decoder layer in a large eager step."""
    if (not ENABLED or layer_idx not in _LAYERS or num_tokens < _MIN_TOKENS
            or torch.cuda.is_current_stream_capturing()):
        yield
        return
    _state["peak"] = max(_state["peak"], torch.cuda.max_memory_allocated())
    a0 = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        pk = torch.cuda.max_memory_allocated()
        _state["peak"] = max(_state["peak"], pk)
        _emit({"ev": "phase", "layer": layer_idx, "tag": tag, "toks": int(num_tokens), "before": a0 // _MIB,
               "peak_over": (pk - a0) // _MIB, "net": (torch.cuda.memory_allocated() - a0) // _MIB,
               "reserved": torch.cuda.memory_reserved() // _MIB})
