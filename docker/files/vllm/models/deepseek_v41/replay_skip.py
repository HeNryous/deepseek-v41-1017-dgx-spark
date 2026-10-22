# SPDX-License-Identifier: Apache-2.0
"""dsv41 patch 0075 (Entwurf 22.10.2026): skip the decoder replay on intermediate prefill chunks.

SWA bounded replay (decoder_replay_layers.py, patch 0034) runs the layers past the last KV source (20-39 on V4.1-Flash)
on each prefill request's last ``window`` (= sliding window, 128) rows in every eager step. Those layers own only
sliding-window KV, and a trimmed chunk (query_len > window) reads replay-layer window KV only from its own kept rows
(model_state._gather_replay_batch_kernel: kv_start = max(replay_start, seq_len - kept_len)). So the replay of chunk N is
read later only by an UNtrimmed chunk N+1 (query_len <= window, it reads up to window-1 positions back into chunk N),
by the first decode tokens when chunk N is the last chunk, and by nothing else: no logits are kept for partial
prefills, the DSpark drafter attends over a 128-token window of its own context KV, and the layers before the cut
(incl. layer 20's KV for every row, write_batch_kv) run as before.

The patch therefore keeps a single row (the chunk's last, with a window of itself) instead of ``window`` rows for a
prefill request whose chunk
  - trims (query_len > window),
  - is not its last and leaves more than a window (remaining_after > window), and
  - is text only and asks no prompt logprobs (those requests keep every row anyway),
and the scheduler guarantees (min_chunk below) that a chunk which leaves more than a window always trims, so the chunk
after a skipped one never reads the skipped window. Final outputs, the final chunk's window KV and everything the decode
reads stay what they are without the patch; an intermediate chunk saves layers 20-39 on 127 of its 128 replay rows
(MoE streaming of ~86 % of 20 layers' experts, attention, 5 long-context indexers: ~0.1 s of ~1.9 s per 4096-token chunk
on GB10, estimate). Switch: VLLM_DSV41_REPLAY_SKIP=1 (default 0 = unchanged).

Pure numpy, no vLLM import (CPU tests import this file directly).
"""

from __future__ import annotations

import os

import numpy as np

REPLAY_SKIP = os.environ.get("VLLM_DSV41_REPLAY_SKIP", "0") == "1"


def skip_mask(
    query_lens: np.ndarray,
    is_prefilling: np.ndarray,
    remaining_after: np.ndarray,
    keeps_rows: np.ndarray,
    has_mm: np.ndarray,
    window: int,
) -> np.ndarray:
    """[num_reqs] bool: requests whose replay rows of this step nothing will read (see module docstring).

    query_lens: rows of each request in this step; remaining_after: prompt (prefill) tokens left after this step;
    keeps_rows: requests that keep every row (prompt logprobs); has_mm: requests with multimodal inputs (their chunks
    can end at encoder boundaries the scheduler rule below does not cover)."""
    q = np.asarray(query_lens)
    return (
        np.asarray(is_prefilling, dtype=bool)
        & ~np.asarray(keeps_rows, dtype=bool)
        & ~np.asarray(has_mm, dtype=bool)
        & (q > window)
        & (np.asarray(remaining_after) > window)
    )


def kept_lens(
    query_lens: np.ndarray, keeps_rows: np.ndarray, window: int, skip: np.ndarray | None = None
) -> np.ndarray:
    """Rows each request keeps for the replay layers: all (prompt logprobs), the last ``window`` (as before), or the
    last one for a skipped request."""
    q = np.asarray(query_lens)
    kept = np.where(np.asarray(keeps_rows, dtype=bool), q, np.minimum(q, window))
    if skip is not None:
        kept = np.where(skip, np.minimum(kept, 1), kept)
    return kept


def min_chunk(
    num_new_tokens: int, remaining: int, window: int, enabled: bool = REPLAY_SKIP, has_encoder_inputs: bool = False
) -> int:
    """Scheduler side: a prefill chunk that leaves more than a window must trim (more than ``window`` rows); otherwise
    the request waits a step (0). The final chunk (num_new_tokens == remaining) is never touched, nor are decodes
    (remaining <= window). Requests with encoder inputs keep vLLM's chunking (their worker side never skips)."""
    if not enabled or window <= 0 or has_encoder_inputs or num_new_tokens <= 0:
        return num_new_tokens
    if remaining > window and num_new_tokens <= window:
        return 0
    return num_new_tokens
