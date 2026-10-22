# SPDX-License-Identifier: Apache-2.0
"""dsv41 patch 0080 (draft 22.10.2026): reserve at start-up what long contexts, mixed steps and the first image would
add to a worker's memory later, so that the free memory after start-up is the floor.

Where memory goes today (code inspection, per rank, details in loops/PLAN2.md): everything sized by max_model_len is
allocated at start-up already - the indexer K-gather workspace (F x max_model_len x 132 B, reserved by the profile run
and locked after graph capture), the RoPE caches (0033), the decode-graph pools, the token history. What grows later
lives in torch's caching allocator, which keeps a peak once it has served it (expandable segments; nothing in the serve
calls empty_cache): the activation peak of a long-context prefill chunk (indexer logits up to their
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB cap, candidate-block temporaries; graph capture's empty_cache releases the profile
run's peak), eager decode-indexer logits of mixed steps ([rows, max_model_len / ratio] fp32), the vision encoder's
activations at the first image, and drift of the allocator's layout over long prefills; plus pinned staging for long
prompts on the host.

VLLM_DSV41_PREALLOC=1 (all steps) or a comma list of ctx,mixed,vision,uva. Run once per worker at the end of
compile_or_warm_up_model, after graph capture (its empty_cache and workspace lock) - like vLLM's V1 runner, which runs
its sampler dummy pass "after capture_model on purpose" against fragmentation:
  ctx     one prefill-shaped dummy step of max_num_batched_tokens rows whose requests sit at the longest context
          (_dummy_run(context_len=max_model_len - rows), the set_dummy_context path vLLM's adaptive-verification
          profiling takes after capture): the long-context activation peak stays reserved. A few seconds.
  mixed   vLLM's run_mixed_prefill_decode_warmup (one decode + one prefill request through execute_model and
          sample_tokens): eager mixed-step allocations, among them the decode-indexer logits (6 rows of them; a
          full 24-row batch needs up to 4x that: VLLM_DSV41_PREALLOC_HEADROOM_MB).
  vision  the encoder's profile pass at the maximum feature size again (profile_run's is released by capture).
  uva     the pinned staging pool for new prompt tokens grown to max_model_len per slot (host, 2 x 4 B x max len).
VLLM_DSV41_PREALLOC_HEADROOM_MB=<n> (default 0): last, one n-MiB block allocated and freed on the main stream; the
          allocator keeps it mapped and serves later growth up to n MiB from it.
VLLM_DSV41_PREALLOC_WATCH=1 (default on when the patch is on): a daemon thread logs every growth of torch's reserved
          memory (>= 32 MiB) or of this process's RSS (>= 64 MiB) over the level after start-up (5 s period).
Numerics: dummy inputs before the API serves anything; dummy batches write KV only to PAD slots (the compressor ring
keeps them out of the null block, PR #58560); the KV pool holds no request and the prefix cache is empty; block 0 is
zeroed at the end as after vLLM's warmup. Default (unset) = nothing runs.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001  (CPU tests without vLLM)
    import logging

    logger = logging.getLogger(__name__)

ALL_STEPS = ("ctx", "mixed", "vision", "uva")
_MIB = 1 << 20


def parse_steps(spec: str) -> tuple[str, ...]:
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "off", "none"):
        return ()
    if spec in ("1", "on", "all"):
        return ALL_STEPS
    return tuple(s for s in ALL_STEPS if s in {x.strip() for x in spec.split(",")})


STEPS = parse_steps(os.environ.get("VLLM_DSV41_PREALLOC", ""))
try:
    HEADROOM_MB = max(0, int(os.environ.get("VLLM_DSV41_PREALLOC_HEADROOM_MB", "0") or 0))
except ValueError:
    HEADROOM_MB = 0
ENABLED = bool(STEPS) or HEADROOM_MB > 0
WATCH = ENABLED and os.environ.get("VLLM_DSV41_PREALLOC_WATCH", "1") == "1"


def _mem_mib() -> tuple[int, int]:
    """(MemAvailable, this process's RSS) in MiB; -1 where /proc is not readable."""
    avail = rss = -1
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                    break
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // _MIB
    except OSError:
        pass
    return avail, rss


def _ctx(worker, runner) -> str:
    rows = int(runner.max_num_tokens)
    context = max(0, int(runner.max_model_len) - rows)
    # is_profile: eager, and the dummy rows are routed to experts (not padding), so the MoE prompt path's buffers
    # (the TensorFold scratch, transient with 0076d) are reserved too, not only the attention / indexer peak;
    # random ids spread them over the experts like text
    runner._dummy_run(rows, context_len=context, skip_eplb=True, is_profile=True, randomize_inputs=True)
    return f"{rows} rows at context {context}"


def _mixed(worker, runner) -> str:
    from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

    num_tokens = min(int(runner.max_num_tokens), 256)
    done = run_mixed_prefill_decode_warmup(
        runner, worker.execute_model, worker.sample_tokens, num_tokens, req_id_prefix="_dsv41_prealloc"
    )
    return "1 decode + 1 prefill" if done else "skipped by vLLM"


def _vision(worker, runner) -> str:
    if not (getattr(runner, "supports_mm_inputs", False) and getattr(runner, "is_first_pp_rank", True)):
        return "no multimodal inputs"
    mm_config = runner.model_config.multimodal_config
    if mm_config is None or getattr(mm_config, "skip_mm_profiling", False):
        return "encoder profiling off"
    from vllm.multimodal.encoder_budget import MultiModalBudget

    budget = MultiModalBudget(runner.vllm_config, runner.mm_registry, enable_cache=False)
    runner.model_state.encoder_runner.profile_encoder_cache(budget.get_dummy_encoder_profile_inputs(), budget)
    runner.reset_encoder_cache()
    return "encoder at the maximum feature size"


def _uva(worker, runner) -> str:
    pool = getattr(runner.req_states.all_token_ids, "write_contents", None)
    if pool is None:
        return "no UVA staging"
    n = int(runner.max_model_len)
    zeros = np.zeros(n, dtype=np.int32)
    for _ in range(pool.max_concurrency):
        pool.copy_to_uva(zeros)  # round robin: every slot grows to the next power of two >= n
    return f"{pool.max_concurrency} slots x {(1 << (n - 1).bit_length()) * 4 // _MIB} MiB pinned"


_STEP_FNS = {"ctx": _ctx, "mixed": _mixed, "vision": _vision, "uva": _uva}


def run(worker) -> None:
    """Worker hook (end of compile_or_warm_up_model, after graph capture). Never raises."""
    runner = worker.model_runner
    device = runner.device
    torch.accelerator.synchronize()
    reserved0 = torch.cuda.memory_reserved(device)
    avail0, rss0 = _mem_mib()
    notes = []
    for step in STEPS:
        before = torch.cuda.memory_reserved(device)
        t0 = time.monotonic()
        try:
            note = _STEP_FNS[step](worker, runner)
            torch.accelerator.synchronize()
        except Exception as exc:  # noqa: BLE001  (a reservation step must not take the start-up down)
            logger.warning("dsv41 0080: step %s failed (%r); continuing without it", step, exc)
            note = "failed"
        grown = (torch.cuda.memory_reserved(device) - before) // _MIB
        notes.append(f"{step} +{grown} MiB ({note}, {time.monotonic() - t0:.1f} s)")
    if HEADROOM_MB:
        block = torch.empty(HEADROOM_MB * _MIB, dtype=torch.uint8, device=device)
        del block  # freed, but the caching allocator keeps it mapped for later allocations
        notes.append(f"headroom {HEADROOM_MB} MiB")
    zeroer = getattr(runner, "kv_block_zeroer", None)
    if zeroer is not None and "ctx" in STEPS:
        zeroer.zero_block_ids([0])  # as after vLLM's warmup
    torch.accelerator.synchronize()
    reserved1 = torch.cuda.memory_reserved(device)
    avail1, rss1 = _mem_mib()
    logger.info(
        "dsv41 0080: torch reserved %d -> %d MiB (%s); MemAvailable %d -> %d MiB, RSS %d -> %d MiB",
        reserved0 // _MIB, reserved1 // _MIB, "; ".join(notes), avail0, avail1, rss0, rss1,
    )
    if WATCH:
        threading.Thread(
            target=_watch, args=(device, reserved1, rss1), name="dsv41-prealloc-watch", daemon=True
        ).start()


def _watch(device, reserved_base: int, rss_base: int, period_s: float = 5.0) -> None:
    last_r, last_rss = reserved_base, rss_base
    while True:
        time.sleep(period_s)
        try:
            r = torch.cuda.memory_reserved(device)
            avail, rss = _mem_mib()
        except Exception:  # noqa: BLE001
            return
        if r - last_r >= 32 * _MIB or (rss >= 0 and rss - last_rss >= 64):
            logger.info(
                "dsv41 0080 watch: torch reserved %d MiB (+%d since start-up), RSS %d MiB (+%d), MemAvailable %d MiB",
                r // _MIB, (r - reserved_base) // _MIB, rss, rss - rss_base, avail,
            )
            last_r, last_rss = max(last_r, r), max(last_rss, rss)
