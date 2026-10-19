# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: running the replay layers on their batch.

Once the last KV-source layer has written every row's KV, it and the layers past
it (which own only SWA KV) run on each request's last ``window`` rows only, in
eager prefill steps. ``DeepseekV41ModelState`` prepares those rows as a sub-batch
with attention metadata and a forward context of its own, like a microbatch;
``DecoderReplayLayers`` gathers the layer inputs by its rows, runs the layers
under that context and scatters the outputs back to batch rows. Steps that run
in a CUDA graph keep the layers on the whole batch, inside the graph.

dsv41 patch 0034: backport of vLLM #59894 (eager path only; #59532's replay
graphs are not taken). With ``VLLM_DSV41_REPLAY_TRIM_CUT=0`` the model builds
this class the #58132 way: no ``write_batch_kv``, the replay layers start past
the cut and ``row_buffers`` are compacted to the replay rows.
"""

from collections.abc import Callable

import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput


class DecoderReplayLayers:
    """Runs the replay layers on the step's replay batch.

    ``run_layers`` takes a batch's layer inputs and returns its per-row
    outputs. ``write_batch_kv`` takes them too and writes the first replay
    layer's KV for every row; ``first_swa_prefix`` keys that layer's SWA
    metadata, which the model state builds with the batch's window starts.
    ``row_buffers`` (only without ``write_batch_kv``) hold per-row results the
    source layer's indexer publishes for the layers after it; they are
    compacted to the replay rows in place.
    """

    def __init__(
        self,
        window: int,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        row_buffers: list[torch.Tensor] | None = None,
        write_batch_kv: Callable[..., None] | None = None,
        first_swa_prefix: str | None = None,
    ) -> None:
        assert (write_batch_kv is None) == (first_swa_prefix is None)
        assert write_batch_kv is None or not row_buffers
        self.window = window
        self.run_layers = run_layers
        self.row_buffers = row_buffers or []
        self.write_batch_kv = write_batch_kv
        self.first_swa_prefix = first_swa_prefix
        # The replay batch, set by the model state every step: its rows of the
        # batch and its forward context. None runs the layers on the batch.
        self.rows: torch.Tensor | None = None
        self.forward_context: ForwardContext | None = None

    def __call__(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        rows = self.rows
        if rows is None:
            return self.run_layers(hidden_states, *states)
        # A trimming step holds a prefill longer than the window, more tokens
        # than any step whose MoE leaves its finalize to the next layer.
        assert isinstance(hidden_states, torch.Tensor)
        if self.write_batch_kv is not None:
            # Under the batch's forward context: every row's KV of the first
            # replay layer, which then reruns on the replay rows below.
            self.write_batch_kv(hidden_states, *states)
        num_rows = rows.shape[0]
        for buf in self.row_buffers:
            buf[:num_rows].copy_(buf.index_select(0, rows))
        with override_forward_context(self.forward_context):
            row_outputs = self.run_layers(
                hidden_states.index_select(0, rows),
                *(None if t is None else t.index_select(0, rows) for t in states),
            )
        # The trimmed rows' outputs stay zero; nothing reads them.
        num_tokens = hidden_states.shape[0]
        return tuple(
            out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(0, rows, out)
            for out in row_outputs
        )
