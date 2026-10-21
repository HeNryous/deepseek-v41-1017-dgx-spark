# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

import functools
import os
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.depth_policy import depth_policy_requested
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)

# dsv41 patch 0066 (21.10.2026), both off by default (unset: every line below is inert, the speculator runs as before):
# VLLM_DSV41_MARKOV_FUSED=1: the greedy local-argmax Markov loop (VLLM_DSPARK_LOCAL_ARGMAX=1) in fused launches,
#   bias -> chunk maxima -> one all-gather -> pick, instead of ~13 kernels a step (ops/markov_step.py). Same drafts.
# VLLM_DSV41_MARKOV_TRIM=1 (or a comma list of step counts): the Markov loop only as deep as the next step verifies.
#   The scheduler's next draft count (SchedulerOutput.num_spec_tokens_to_schedule: dynamic schedule + the 0047 window,
#   2..5) reaches the speculator through gpu_worker.py; under async scheduling no request verifies more drafts than
#   that next step. A FULL graph holds a fixed loop, so the DSpark graphs are captured once more for each shorter
#   depth (1: VLLM_DSV41_AK_MIN .. k-1) and replayed by depth; the backbone, the head and the drafts of the steps that
#   run are those of the full graph, the confidences past the depth are 0 (the depth policy cannot pick rows the
#   window does not have). With the switch on, the 12 never-dispatched DSpark graphs of the 0047 widths (uniform 2-4
#   tokens a request: the drafter always dispatches 5) are not captured, so the graph count stays the same.
_DSV41_MK_FUSED = os.environ.get("VLLM_DSV41_MARKOV_FUSED", "0") == "1"
_DSV41_MK_TRIM = os.environ.get("VLLM_DSV41_MARKOV_TRIM", "0").strip()


def dsv41_trim_steps(spec: str, num_steps: int, ak_min: int) -> list[int]:
    """Shorter Markov depths to capture for VLLM_DSV41_MARKOV_TRIM=spec: "1" = ak_min .. num_steps-1 (at least 1),
    a comma list = those depths below num_steps, "" / "0" = none."""
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "off"):
        return []
    if spec in ("1", "on"):
        return list(range(max(1, ak_min), num_steps))
    out = set()
    for x in spec.split(","):
        x = x.strip()
        if x:
            k = int(x)
            if 1 <= k < num_steps:
                out.add(k)
    return sorted(out)


# Patch 0055 (VLLM_DSV41_HYBRID_DRAFT=1, default off; needs draft_sample_method=probabilistic): only requests in the
# full-window regime of 0047 (scheduled drafts == the batch's k, i.e. accepted-draft average >= VLLM_DSV41_AK_HI: code,
# JSON) draft by sampling; the others draft greedily, and their cached draft distribution becomes one-hot at the drafted
# token. The rejection sampler then verifies those rows exactly like greedy drafts (q(x) = 1, residual = p without x),
# also under block verification, where a point-mass draft gives the token-wise acceptance. Prose keeps greedy drafting
# (sampled drafts lowered its acceptance, F26/F27); code and JSON get sampled drafts + block verification.
@triton.jit
def _dsv41_onehot_kernel(
    cache_ptr, stride0, stride1, col_ptr, idx_map_ptr, mode_ptr, ids_ptr, vocab, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    state = tl.load(idx_map_ptr + row)
    if state < 0:
        return
    if tl.load(mode_ptr + state):
        return
    tok = tl.load(ids_ptr + row)
    col = tl.load(col_ptr)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    val = tl.where(j == tok, 0.0, float("-inf"))
    ptr = cache_ptr + state.to(tl.int64) * stride0 + col.to(tl.int64) * stride1 + j
    tl.store(ptr, val.to(cache_ptr.dtype.element_ty), mask=j < vocab)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.use_confidence_head: bool = False
        # The model reads cached Markov bias rows by token id when it has them.
        self._markov_by_ids: bool = False
        # Greedy drafts reduced per vocabulary shard (VLLM_DSPARK_LOCAL_ARGMAX); set in
        # load_draft_model once the model has its local bias rows.
        self._local_argmax: bool = False
        # patch 0055: per request state, True = draft by sampling (full-window regime), False = greedy
        self._hybrid = os.environ.get("VLLM_DSV41_HYBRID_DRAFT", "0") == "1"
        self._sampled_mode = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=device)
        # patch 0066: fused Markov loop (decided in load_draft_model), trimmed depths and their graph managers
        self._mk_fused = False
        self._trim_steps: list[int] = []
        self._trim_attr: dict[int, str] = {}
        self._next_drafts: int | None = None
        self._draft_steps_now = self.num_speculative_steps

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        # The per-round depth policy (depth_policy.py) reads the confidences too.
        self.use_confidence_head = (
            self.enable_adaptive_verification or depth_policy_requested()
        ) and model.model.confidence_head is not None
        if self.use_confidence_head:
            # The acceptance estimator is not needed when a trained confidence head
            # is available.
            self.use_acceptance_estimator = False
        self._markov_by_ids = hasattr(model, "markov_bias_for_ids")
        # The local argmax never materializes full draft logits: greedy drafting only, and
        # no acceptance estimator (it reads the full logits; the confidence head replaces it).
        self._local_argmax = (
            getattr(model, "supports_local_argmax", lambda: False)()
            and self.draft_logits is None
            and not self.use_acceptance_estimator
            and self._draft_topk is None
        )
        if self._local_argmax:
            logger.info("DSpark: greedy drafts argmax per vocabulary shard (local argmax)")
        # patch 0066: both need the greedy local-argmax loop (no draft logits, no top-k candidates)
        self._mk_fused = _DSV41_MK_FUSED and self._local_argmax
        if _DSV41_MK_FUSED and not self._mk_fused:
            logger.warning("DSpark: VLLM_DSV41_MARKOV_FUSED=1 ignored (needs the local-argmax greedy loop)")
        self._trim_steps = []
        if _DSV41_MK_TRIM not in ("", "0"):
            steps = dsv41_trim_steps(
                _DSV41_MK_TRIM,
                self.num_speculative_steps,
                int(os.environ.get("VLLM_DSV41_AK_MIN", "2")),
            )
            async_sched = getattr(self.vllm_config.scheduler_config, "async_scheduling", False)
            if not self._local_argmax:
                logger.warning("DSpark: VLLM_DSV41_MARKOV_TRIM ignored (needs the local-argmax greedy loop)")
            elif async_sched is not True:
                # the synchronous scheduler takes every returned draft, not num_spec_tokens_to_schedule
                logger.warning("DSpark: VLLM_DSV41_MARKOV_TRIM ignored (needs async scheduling)")
            else:
                self._trim_steps = steps
                logger.info(
                    "DSpark: Markov loop trimmed to the next step's draft count (patch 0066), "
                    "extra graphs for depths %s", steps,
                )
        if self._mk_fused:
            logger.info("DSpark: fused Markov loop (patch 0066, VLLM_DSV41_MARKOV_FUSED=1)")
        return model

    # --- dsv41 patch 0066: trimmed Markov loop ------------------------------------------------------------------

    def dsv41_set_next_drafts(self, num_drafts: int) -> None:
        """gpu_worker.py, before each step: the scheduler's draft count for the next step (VLLM_DSV41_MARKOV_TRIM)."""
        self._next_drafts = int(num_drafts)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        self._trim_attr = {}
        mgr = self.query_cudagraph_manager
        if not self._trim_steps or mgr is None or mgr.cudagraph_mode == CUDAGraphMode.NONE:
            return
        q = self.num_query_per_req

        def dispatched_only(m: DFlashCudaGraphManager) -> None:
            # the speculator always dispatches uniform_token_count = num_query_per_req (dflash propose)
            for mode in list(m._capture_descs):
                m._capture_descs[mode] = [
                    d
                    for d in m._capture_descs[mode]
                    if d.cg_mode != CUDAGraphMode.FULL or d.uniform_token_count == q
                ]

        dispatched_only(mgr)
        for k in self._trim_steps:
            m = DFlashCudaGraphManager(
                self.vllm_config, self.device, mgr.cudagraph_mode, decode_query_len=q
            )
            dispatched_only(m)
            # a direct attribute, as query_cudagraph_manager: profile_cudagraph_memory finds and drops it
            name = f"query_cudagraph_manager_s{k}"
            setattr(self, name, m)
            self._trim_attr[k] = name

    def capture(self) -> None:
        super().capture()
        for k in sorted(self._trim_attr):
            m = getattr(self, self._trim_attr[k], None)
            if m is None:
                continue
            m.capture(
                functools.partial(self._generate_draft, steps=k),
                self.input_buffers,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                self.max_model_len,
                causal=self._group_causal,
                precompute_context_kv=lambda num_reqs: self._precompute_context_kv(
                    0, self._num_graph_context_tokens(num_reqs)
                ),
                progress_bar_desc=f"Capturing dspark CUDA graphs ({k} Markov steps)",
            )

    def _trim_manager(self, steps: int) -> tuple[int, DFlashCudaGraphManager] | None:
        """The captured variant with the fewest Markov steps >= steps, or None (the full graph)."""
        for k in sorted(self._trim_attr):
            if k >= steps:
                m = getattr(self, self._trim_attr[k], None)
                if m is not None:
                    return k, m
        return None

    def _replay_draft_graph(self, batch_desc) -> None:
        steps = self._draft_steps_now
        if steps < self.num_speculative_steps and self._trim_attr:
            hit = self._trim_manager(steps)
            if hit is not None and batch_desc in hit[1].graphs:
                hit[1].run_fullgraph(batch_desc)
                return
        super()._replay_draft_graph(batch_desc)

    def _dsv41_cap(self, num_reqs: int) -> int:
        for lo, hi, k in self.speculative_config.num_speculative_tokens_per_batch_size or ():
            if lo <= num_reqs <= hi:
                return int(k)
        return self.num_speculative_steps

    def propose(self, input_batch, *args, **kwargs):
        if self._trim_steps:
            # patch 0066: this step's drafts serve the next step, which verifies at most the scheduler's count;
            # dummy / profile runs and a missing count (never set) keep the full loop
            nxt, self._next_drafts = self._next_drafts, None
            n_spec = self.num_speculative_steps
            if kwargs.get("dummy_run", False) or nxt is None:
                self._draft_steps_now = n_spec
            else:
                self._draft_steps_now = max(1, min(nxt, n_spec))
        if self._hybrid and self.draft_logits is not None:
            # patch 0055: this step's window per request decides the next block's draft mode (0047 gives the full
            # window exactly to requests whose accepted-draft average is >= AK_HI, and to new ones)
            flags = np.zeros(self.max_num_reqs, dtype=np.bool_)
            per = getattr(input_batch, "num_draft_tokens_per_req", None)
            n = input_batch.num_reqs
            if per is not None and n > 0:
                flags[np.asarray(input_batch.idx_mapping_np[:n])] = np.asarray(per[:n]) >= self._dsv41_cap(n)
            self._sampled_mode.copy_(torch.from_numpy(flags).pin_memory(), non_blocking=True)
        return super().propose(input_batch, *args, **kwargs)

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            draft_ids = logits.argmax(dim=-1)
            return self.model.map_draft_to_target(draft_ids)

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sampler = (
            gumbel_sample
            if self.draft_watermarker is None
            else self.draft_watermarker.sample
        )
        drafted = sampler(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )
        if not self._hybrid:
            return drafted
        # patch 0055: greedy rows take the argmax (lowest index on ties, like the local-argmax path) and get a
        # one-hot cached distribution; padding rows (state -1) are skipped by the kernel
        greedy = logits.argmax(dim=-1)
        mode = self._sampled_mode[idx_map.clamp(min=0).long()]
        out = torch.where(mode, drafted, greedy)
        vocab = self.draft_logits.shape[-1]
        block = 4096
        _dsv41_onehot_kernel[(logits.shape[0], triton.cdiv(vocab, block))](
            self.draft_logits,
            self.draft_logits.stride(0),
            self.draft_logits.stride(1),
            self._step_cols[step : step + 1],
            idx_map,
            self._sampled_mode,
            greedy,
            vocab,
            BLOCK=block,
        )
        return out

    def _sample_sequential(
        self, num_reqs: int, head_hidden: torch.Tensor, steps: int | None = None
    ) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        # patch 0066: steps < n_spec only from a trimmed graph (local argmax); drafts past it are never verified
        steps = n_spec if steps is None else steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        if self._local_argmax:
            # this rank's vocabulary shard of the base logits, never gathered
            base_logits = self.model.compute_draft_logits_local(sample_hidden)
        else:
            # Draft-vocab logits; sampled ids are remapped to target vocab below.
            base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        if self._mk_fused:
            # patch 0066: the same drafts from fused launches; returns the Markov embeddings [n, n_spec, rank]
            embeds = self.model.draft_loop_local_fused(
                base_logits,
                self.input_buffers.input_ids,
                self._anchor_idx[:num_reqs],
                self.draft_tokens,
                steps,
                self.max_num_reqs,
            )
            if self.use_confidence_head:
                self._write_confidence(num_reqs, sample_hidden, embeds.flatten(0, 1), steps)
            return

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(steps):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            if self._local_argmax:
                draft_sampled_i = self.model.draft_argmax_local(
                    base_logits[:, i], prev, markov_embed
                )
                self.draft_tokens[:num_reqs, i] = draft_sampled_i
                prev = draft_sampled_i
                continue
            if self._markov_by_ids:
                bias = self.model.markov_bias_for_ids(prev, markov_embed)
            else:
                bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            if steps < n_spec:
                # patch 0066: the confidence head keeps its n_spec rows a request (same GEMM shape, so the rows
                # that ran keep their bits); the padded rows are zeroed in _write_confidence
                confidence_markov_embeds += [confidence_markov_embeds[-1]] * (n_spec - steps)
            self._write_confidence(
                num_reqs,
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
                steps,
            )

    def _write_confidence(
        self,
        num_reqs: int,
        sample_hidden: torch.Tensor,
        markov_embeds: torch.Tensor,
        steps: int,
    ) -> None:
        n_spec = self.num_speculative_steps
        confidence = self.model.compute_confidence(sample_hidden, markov_embeds)
        self.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)
        if steps < n_spec:
            # patch 0066: positions past the next step's window have no draft: confidence 0, so the depth policy
            # never prices them (survival 0 adds no expected token, a wider row costs more)
            self.draft_token_confidence_probs[:num_reqs, steps:] = 0.0

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        steps: int | None = None,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        # patch 0066: steps = the Markov depth a trimmed graph is captured with (None: num_speculative_steps)
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden, steps)
