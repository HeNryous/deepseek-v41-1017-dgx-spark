# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41 patch 0067: a shorter eager "before" phase of a decode round.

VLLM_DSV41_VOR_FUSED=1 turns on both parts below, =swa or =copy one of them; default 0 = off.

swa:  the decode SWA metadata of all sliding-window KV cache groups of the target in one launch.
      Per group the SWA builder runs torch.ge(slot_mapping, 0), decode_swa_lens[n:] = 0 and
      _compute_swa_indices_and_lens_kernel (sparse_swa.py); DeepSeek-V4.1-Flash has 15 such groups,
      so 45 launches, ~95 us of GPU time a round in trace F15 (kernels ~1-3 us, launch gaps ~0.5 us).
      _swa_decode_groups_kernel writes the same three buffers of every group in one grid; its per-row
      code is the upstream kernel's, the tail work is the torch.ge / fill of the rows past the decode
      tokens.
copy: DeepseekV41ModelState uploads replay_start every round, zeros in decode-only rounds; the copy
      engine round trip costs ~9 us of GPU time a round (F15). The upload now runs only when the values
      differ from what the buffer holds (the model state is its only writer).
"""

import os
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_MODE = os.environ.get("VLLM_DSV41_VOR_FUSED", "0").strip().lower()
SWA_FUSED = _MODE in ("1", "swa")
REPLAY_COPY_SKIP = _MODE in ("1", "copy")

# Pointer table columns, one int64 row per group (the kernel reads rows of 8).
_SLOT, _VALID, _TABLE, _INDICES, _LENS, _TABLE_STRIDE, _BLOCK_SIZE = range(7)
_COLS = 8
_BLOCK = 1024  # = the TRITON_BLOCK_SIZE of the upstream kernel


@triton.jit(do_not_specialize=["num_tokens", "num_valid"])
def _swa_decode_groups_kernel(
    table_ptr,  # int64 [num_groups, _COLS]: see the column names above
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    replay_start_ptr,
    num_tokens,  # decode rows of the step, the same in every group
    num_valid,  # slot_mapping entries of a group (>= num_tokens)
    lens_capacity,  # decode_swa_lens entries of a group
    window_size,
    index_width,
    swa_indices_stride,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = table_ptr + tl.program_id(1) * 8
    slot_mapping_ptr = tl.cast(tl.load(row + 0), tl.pointer_type(tl.int64))
    is_valid_token_ptr = tl.cast(tl.load(row + 1), tl.pointer_type(tl.int8))
    block_table_ptr = tl.cast(tl.load(row + 2), tl.pointer_type(tl.int32))
    swa_indices_ptr = tl.cast(tl.load(row + 3), tl.pointer_type(tl.int32))
    swa_lens_ptr = tl.cast(tl.load(row + 4), tl.pointer_type(tl.int32))
    block_table_stride = tl.cast(tl.load(row + 5), tl.int32)
    block_size = tl.cast(tl.load(row + 6), tl.int32)

    if pid >= num_tokens:
        # Past the decode rows: torch.ge(slot_mapping, 0, out=is_valid_token) and
        # decode_swa_lens[num_tokens:] = 0. Masked-off lanes keep in-range addresses.
        offs = num_tokens + (pid - num_tokens) * TRITON_BLOCK_SIZE
        offs = offs + tl.arange(0, TRITON_BLOCK_SIZE)
        in_valid = offs < num_valid
        safe = tl.where(in_valid, offs, 0)
        slot = tl.load(slot_mapping_ptr + safe, mask=in_valid, other=-1)
        tl.store(is_valid_token_ptr + safe, tl.cast(slot >= 0, tl.int8), mask=in_valid)
        in_lens = offs < lens_capacity
        tl.store(swa_lens_ptr + tl.where(in_lens, offs, 0), 0, mask=in_lens)
        return

    # Decode row `pid`: torch.ge for its token, then the body of
    # _compute_swa_indices_and_lens_kernel with token_offset 0 and HAS_IMAGE False.
    token_idx = pid
    is_valid = tl.load(slot_mapping_ptr + token_idx) >= 0
    tl.store(is_valid_token_ptr + token_idx, tl.cast(is_valid, tl.int8))
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        # Clear the row so a padded token cannot gather through stale indices.
        for i in range(0, index_width, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                swa_indices_ptr + pid * swa_indices_stride + offset,
                -1,
                mask=offset < index_width,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    left = 0
    right = 0
    left_add = tl.maximum(left - (window_size - 1), 0)
    start_pos = tl.maximum(pos - (window_size - 1) - left_add, 0)
    # SWA bounded replay: no window KV exists below the request's replay start.
    start_pos = tl.maximum(start_pos, tl.load(replay_start_ptr + req_idx))
    end_pos = pos + right + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < index_width,
        )


def _same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


def swa_plan_entries(
    attn_groups: list[list[Any]],
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
    builder_cls: type,
    ubatch_idx: int = 0,
) -> list[tuple[Any, torch.Tensor, torch.Tensor]]:
    """(builder, block table, slot-mapping row) of every KV cache group whose only attention
    group has a ``builder_cls`` builder, in the order build_attn_metadata builds them, with the
    tensors it hands them (block_tables[i] itself, a view of slot_mappings[i]). Groups that
    also hold other builders stay out: those could change the shared block table first."""
    entries = []
    for i, groups in enumerate(attn_groups):
        if len(groups) != 1:
            continue
        builder = groups[0].get_metadata_builder(ubatch_idx)
        if isinstance(builder, builder_cls):
            entries.append((builder, block_tables[i], slot_mappings[i]))
    return entries


class SWADecodePlan:
    """Decode SWA metadata of several sliding-window groups in one launch.

    The model state arms it around one build_attn_metadata call. The first entry's builder
    launches the kernel for every entry from its own build(), once its token_to_req_indices
    exists, where the upstream code launches that builder's kernels. A builder skips its own
    three launches only if its inputs are the ones the launch read (same block table object,
    same slot-mapping view, same shared tensors and sizes); otherwise it launches them as
    before, after the plan's launch, and so overwrites whatever the plan wrote for it.
    """

    def __init__(self) -> None:
        self.armed = False
        self.disabled = False
        self.entries: list[tuple[Any, torch.Tensor, torch.Tensor]] = []
        self._pos: dict[int, int] = {}
        self._launched: tuple | None = None
        self._table_key: tuple | None = None
        self._table: torch.Tensor | None = None
        self._table_host: torch.Tensor | None = None
        self._warm = False
        self.launches = 0  # diagnostics: fused launches so far

    def arm(self, entries: list[tuple[Any, torch.Tensor, torch.Tensor]]) -> None:
        self.entries = entries
        self._pos = {id(builder): i for i, (builder, _, _) in enumerate(entries)}
        self._launched = None
        self.armed = bool(entries) and not self.disabled
        for builder, _, _ in entries:
            builder.dsv41_swa_plan = self
        if self.armed and not self._warm:
            self._warmup(entries[0][0])

    def disarm(self) -> None:
        self.armed = False
        self.entries = []
        self._pos = {}
        self._launched = None

    def claim(
        self,
        builder: Any,
        common_attn_metadata: Any,
        num_decode_tokens: int,
        num_prefill_tokens: int,
        token_to_req_indices: torch.Tensor,
        replay_start: torch.Tensor | None,
    ) -> bool:
        """True if this builder's is_valid_token, decode_swa_lens and decode_swa_indices hold
        what its own three launches would write; the first entry launches for all."""
        if not self.armed:
            return False
        i = self._pos.get(id(builder))
        if i is None:
            return False
        _, block_table, slot_mapping = self.entries[i]
        cm = common_attn_metadata
        if cm.block_table_tensor is not block_table or not _same_view(
            cm.slot_mapping, slot_mapping
        ):
            return False
        key = (
            num_decode_tokens,
            num_prefill_tokens,
            cm.causal is True,
            slot_mapping.shape[0],
            token_to_req_indices.data_ptr(),
            cm.query_start_loc.data_ptr(),
            cm.seq_lens.data_ptr(),
            None if replay_start is None else replay_start.data_ptr(),
        )
        if i == 0:
            launched = self._launch(
                builder,
                cm,
                num_decode_tokens,
                num_prefill_tokens,
                token_to_req_indices,
                replay_start,
            )
            self._launched = key if launched else None
            return launched
        return self._launched is not None and key == self._launched

    def _eligible(
        self,
        cm: Any,
        num_decode_tokens: int,
        num_prefill_tokens: int,
        token_to_req_indices: torch.Tensor,
        replay_start: torch.Tensor,
    ) -> bool:
        if num_prefill_tokens != 0 or num_decode_tokens <= 0 or cm.causal is not True:
            return False
        first = self.entries[0][0]
        num_valid = self.entries[0][2].shape[0]
        if num_valid < num_decode_tokens:
            return False
        for t in (cm.query_start_loc, cm.seq_lens, token_to_req_indices, replay_start):
            if t.dtype != torch.int32 or t.dim() != 1 or t.stride(0) != 1:
                return False
        if token_to_req_indices.shape[0] < num_decode_tokens:
            return False
        capacity = first.decode_swa_lens.shape[0]
        if capacity < num_valid:
            return False
        for builder, block_table, slot_mapping in self.entries:
            indices, lens, valid = (
                builder.decode_swa_indices,
                builder.decode_swa_lens,
                builder.is_valid_token,
            )
            if not (
                builder.window_size == first.window_size
                and indices.dtype == torch.int32
                and indices.is_contiguous()
                and indices.shape[1:] == first.decode_swa_indices.shape[1:]
                and indices.shape[0] >= num_decode_tokens
                and lens.dtype == torch.int32
                and lens.dim() == 1
                and lens.is_contiguous()
                and lens.shape[0] == capacity
                and valid.dtype == torch.bool
                and valid.is_contiguous()
                and valid.shape[0] >= num_valid
                and block_table.dtype == torch.int32
                and block_table.dim() == 2
                and block_table.stride(1) == 1
                and slot_mapping.dtype == torch.int64
                and slot_mapping.dim() == 1
                and slot_mapping.stride(0) == 1
                and slot_mapping.shape[0] == num_valid
            ):
                return False
        return True

    def _table_for(self, device: torch.device) -> torch.Tensor:
        rows = []
        for builder, block_table, slot_mapping in self.entries:
            row = [0] * _COLS
            row[_SLOT] = slot_mapping.data_ptr()
            row[_VALID] = builder.is_valid_token.data_ptr()
            row[_TABLE] = block_table.data_ptr()
            row[_INDICES] = builder.decode_swa_indices.data_ptr()
            row[_LENS] = builder.decode_swa_lens.data_ptr()
            row[_TABLE_STRIDE] = block_table.stride(0)
            row[_BLOCK_SIZE] = builder.block_size
            rows.append(tuple(row))
        key = tuple(rows)
        if key != self._table_key:
            host = torch.tensor(rows, dtype=torch.int64)
            if device.type == "cuda":
                host = host.pin_memory()
            # The pinned source stays referenced until the next rebuild, after
            # the copy (same stream) has run.
            self._table = host.to(device, non_blocking=True)
            self._table_host = host
            self._table_key = key
        assert self._table is not None
        return self._table

    def _launch(
        self,
        builder: Any,
        cm: Any,
        num_decode_tokens: int,
        num_prefill_tokens: int,
        token_to_req_indices: torch.Tensor,
        replay_start: torch.Tensor | None,
    ) -> bool:
        if self.disabled:
            return False
        if replay_start is None:
            replay_start = builder.no_replay_start
        if not self._eligible(
            cm, num_decode_tokens, num_prefill_tokens, token_to_req_indices, replay_start
        ):
            return False
        device = builder.decode_swa_lens.device
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return False
        try:
            table = self._table_for(device)
            num_valid = self.entries[0][2].shape[0]
            capacity = builder.decode_swa_lens.shape[0]
            num_tail = triton.cdiv(capacity - num_decode_tokens, _BLOCK)
            _swa_decode_groups_kernel[(num_decode_tokens + num_tail, len(self.entries))](
                table,
                cm.query_start_loc,
                cm.seq_lens,
                token_to_req_indices,
                replay_start,
                num_decode_tokens,
                num_valid,
                capacity,
                builder.window_size,
                builder.decode_swa_indices.shape[-1],
                builder.decode_swa_indices.stride(0),
                TRITON_BLOCK_SIZE=_BLOCK,
            )
        except Exception:  # first launch compiles; stay on the upstream path if that fails
            logger.exception("dsv41 patch 0067: fused SWA metadata off, launch failed")
            self.disabled = True
            self.armed = False
            return False
        self.launches += 1
        return True

    def _warmup(self, builder: Any) -> None:
        """Compile at the first arm (startup dummy runs) instead of in the first request;
        the signature specializes like the step's launches (int32 vectors, int64 table)."""
        self._warm = True
        device = builder.decode_swa_lens.device
        if device.type != "cuda":
            return
        try:
            i32 = torch.zeros(16, dtype=torch.int32, device=device)
            _swa_decode_groups_kernel.warmup(
                torch.zeros((1, _COLS), dtype=torch.int64, device=device),
                i32,
                i32,
                i32,
                i32,
                2,
                3,
                builder.decode_swa_lens.shape[0],
                builder.window_size,
                builder.decode_swa_indices.shape[-1],
                builder.decode_swa_indices.stride(0),
                TRITON_BLOCK_SIZE=_BLOCK,
                grid=(1, 1),
            )
        except Exception:  # compiles at the first launch instead
            logger.warning("dsv41 patch 0067: warmup compile skipped", exc_info=True)


class ReplayStartMirror:
    """Host copy of the values DeepseekV41ModelState._replay_start holds on the device; the
    upload runs only if the step's values differ (copies are stream-ordered, the model state
    is the buffer's only writer)."""

    def __init__(self, size: int) -> None:
        self.values = np.zeros(size, dtype=np.int32)
        self.skipped = 0  # diagnostics

    def holds(self, values: Any) -> bool:
        n = len(values)
        same = bool((self.values[:n] == values).all())
        self.skipped += same
        return same

    def wrote(self, values: Any) -> None:
        self.values[: len(values)] = values
