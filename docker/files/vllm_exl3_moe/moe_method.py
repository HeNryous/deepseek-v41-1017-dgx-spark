# SPDX-License-Identifier: Apache-2.0
"""FusedMoEMethodBase for DeepSeek-V4.1 routed experts stored as EXL3 trellis banks.

Interface (vLLM main @ 6e517b15, model_executor/layers/fused_moe/fused_moe_method_base.py:28-187):
  create_weights(layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **attrs)
      called from RoutedExperts.__init__ (fused_moe/routed_experts.py:186-198)
  get_fused_moe_quant_config(layer)          abstract, :102-106
  process_weights_after_loading(layer)       model_loader/utils.py:146-170
  apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
      called by RoutedExperts.forward_modular (routed_experts.py:1258-1293) inside the opaque
      torch.ops.vllm.moe_forward custom op (fused_moe/runner/moe_runner.py:117-213). The runner adds
      the shared expert and does the TP all-reduce afterwards (moe_runner.py:480-514: moe_kernel is
      None here, so the fused output counts as un-reduced), so apply() returns this rank's partial sum.

Weight loading: the V4.1 loop maps checkpoint `layers.L.ffn.experts.E.w{1,3}.<suffix>` to
`...experts.routed_experts.w13_<suffix>` and `w2.<suffix>` to `w2_<suffix>`
(routed_experts.py:1124-1127, 1164-1180 build_expert_params_mapping; models/deepseek_v41/nvidia/model.py:1246-1282)
and calls `param.weight_loader(param, w, name, shard_id=..., expert_id=..., return_success=True)`.
Registering w13_trellis / w13_suh / w13_svh / w13_mul1 / w13_mcg and the w2_ twins is all the model
needs: no source patch to vLLM.
"""

from __future__ import annotations

import ctypes
import gc
from typing import Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import aux_stream

from . import kernels as K
from . import l2pf, l2pf_draft
from .bank import TENSOR_NAMES, Exl3Bank
from .forward import routed_moe_forward
from .plan import CODEBOOK_MCG, CODEBOOK_MUL1, PROJ_DOWN, PROJ_GATE, PROJ_UP, Exl3MoEPlan

logger = init_logger(__name__)

_PROJ_INDEX = {PROJ_GATE: 0, PROJ_DOWN: 1, PROJ_UP: 2}
# VLLM_EXL3_MOE_SMOE_SHARED=1 (default 0 until measured on the serve): the layer's shared expert runs inside exl3_smoe's
# grids as one more pick per token (smoe.py attach_shared) instead of its own EXL3 dense launches on vLLM's aux stream.
_SMOE_SHARED = K.env_int("VLLM_EXL3_MOE_SMOE_SHARED", 0)
# VLLM_EXL3_MOE_FUSED=1 (default 0 until measured on the serve; needs patch 0016): decode steps of <= 8 tokens run the
# whole MoE phase (router GEMV, top-k, routed experts, shared expert) as one exl3_smoe launch (smoe.py fused), called
# from DeepseekV4MoE before vLLM's runner, which then does not run (Program P stage P2).
_SMOE_FUSED = K.env_int("VLLM_EXL3_MOE_FUSED", 0)
# below this many tokens the runner path stays (GB10, the fused launch vs router_topk + smoe: 0.98x at 1 token, 1.02-1.03x
# at 4-6; LOG Oct 11 03:23)
_SMOE_FUSED_MIN_ROWS = K.env_int("VLLM_EXL3_MOE_FUSED_MIN_ROWS", 2)
_KINDS = ("trellis", "suh", "svh")
# VLLM_EXL3_MOE_SHARED_ZEROS=1 (dsv41 patch 0079c, default 0): when exl3_smoe computes routed + shared experts in one
# launch, _apply_with_shared hands vLLM's runner a zero tensor as the shared output (the runner adds shared + routed).
# Off: torch.zeros_like(out) every MoE layer = one fill kernel a layer (40 a decode round). On: a [m, hidden] view of
# one zero buffer per (device, dtype, hidden), allocated outside graph capture and never written (the runner only
# reads it or reduces it in place, which keeps zeros), so the same add sees the same zeros without the fill.
_SHARED_ZEROS = K.env_int("VLLM_EXL3_MOE_SHARED_ZEROS", 0)
_SHARED_ZEROS_ROWS = 64  # >= coop rows_max (<= 32): the only calls that take this path
_ZERO_BUFFERS: dict[tuple, torch.Tensor] = {}  # kept for the run: captured graphs read these addresses


def _shared_zero_output(out: torch.Tensor) -> torch.Tensor:
    """dsv41 patch 0079c: zeros shaped like `out` from the persistent buffer (see _SHARED_ZEROS)."""
    if out.dim() != 2 or out.shape[0] > _SHARED_ZEROS_ROWS or not out.is_contiguous():
        return torch.zeros_like(out)  # same shape and strides as before in every case the buffer cannot serve
    key = (out.device, out.dtype, out.shape[1])
    buf = _ZERO_BUFFERS.get(key)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            return torch.zeros_like(out)  # a buffer made inside a capture would hold zeros only after a replay
        buf = _ZERO_BUFFERS[key] = torch.zeros(
            (_SHARED_ZEROS_ROWS, out.shape[1]), dtype=out.dtype, device=out.device
        )
    return buf[: out.shape[0]]


def _unquantized_moe_quant_config():
    try:
        from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG

        return FUSED_MOE_UNQUANTIZED_CONFIG
    except ImportError:
        return None


def _is_dummy_load() -> bool:
    try:
        from vllm.config import get_current_vllm_config

        fmt = get_current_vllm_config().load_config.load_format
        return str(getattr(fmt, "value", fmt)).lower() == "dummy"
    except Exception:  # noqa: BLE001
        return False


def _malloc_trim() -> None:
    # GB10 is unified memory: glibc arenas still holding host staging count against the GPU pool too
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _padding_rows(m: int):
    """vLLM's is_padding for this forward (the persistent buffer: a CUDA graph reads it at replay), or None. vLLM's
    DSv4 top-k kernel drops it on the non-hash layers, so exl3_smoe applies it itself."""
    import vllm.envs as envs
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not envs.VLLM_MOE_SKIP_PADDING or not is_forward_context_available():
        return None
    pad = get_forward_context().is_padding
    if pad is None or pad.dtype != torch.bool or pad.numel() < m:
        return None
    return pad[:m]


class Exl3RoutedMoEMethod(FusedMoEMethodBase):

    def __init__(self, moe, plan: Exl3MoEPlan, layer_idx: int, settings, prefix: str):
        super().__init__(moe)
        self.plan = plan
        self.layer_idx = layer_idx
        self.lp = plan.layer(layer_idx)
        self.settings = settings
        self.prefix = prefix
        pc = moe.moe_parallel_config
        if getattr(pc, "use_ep", False) or getattr(pc, "ep_size", 1) > 1:
            # TODO: expert parallel (whole experts per rank) needs the expert map + all2all path;
            # v0 implements the TP channel split only.
            raise NotImplementedError("vllm_exl3_moe v0 supports tensor-parallel MoE only (no --enable-expert-parallel)")
        self.tp_size = int(pc.tp_size)
        self.tp_rank = int(pc.tp_rank)
        self.top_k = int(moe.experts_per_token)
        self.act_limit = float(moe.swiglu_limit or 0.0)
        self.mcg = plan.codebook == "mcg"
        self.mul1 = plan.codebook == "mul1"
        self.bank: Exl3Bank | None = None
        self.groups: list[K.GroupTables] = []
        self.perm: torch.Tensor | None = None
        self.coop: K.CoopScratch | None = None
        self.fused: K.FusedScratch | None = None

    @property
    def method_name(self) -> str:
        return "Exl3RoutedMoEMethod"

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        """With VLLM_EXL3_MOE_SMOE_SHARED=1 apply() takes the shared expert itself (inside exl3_smoe on the decode path,
        through the runner's SharedExperts call on every other path): the runner then neither runs it before the routed
        experts nor on its aux stream (runner/shared_experts.py, the MK_INTERNAL_OVERLAPPED order)."""
        return bool(_SMOE_SHARED) and getattr(self, "smoe", None) is not None

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config):
        # The trellis encodes the checkpoint's exact shapes: never pad
        return hidden_size, intermediate_size_per_partition

    def get_fused_moe_quant_config(self, layer):
        return _unquantized_moe_quant_config()

    # ---------------------------------------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ):
        if num_experts != self.lp.num_experts:
            raise ValueError(f"{self.prefix}: layer has {num_experts} local experts, plan has {self.lp.num_experts}")
        if hidden_size != self.plan.hidden_size:
            raise ValueError(f"{self.prefix}: hidden {hidden_size} != plan {self.plan.hidden_size}")
        act = str(getattr(self.moe, "activation", "silu")).lower()
        if "silu" not in act:
            raise NotImplementedError(f"{self.prefix}: activation {act} (the EXL3 MoE kernels run SiLU-gated)")

        bank = Exl3Bank(self.lp, hidden_size, self.plan.moe_intermediate_size, self.tp_size, self.tp_rank)
        if bank.inter_local != intermediate_size_per_partition:
            raise ValueError(f"{self.prefix}: intermediate per rank {intermediate_size_per_partition} "
                             f"!= plan shard {bank.inter_local}")
        self.bank = bank
        E = num_experts
        tensors: dict[str, torch.Tensor] = dict(bank.allocate())   # default device = vLLM's target
        # codebook marker scalars (exllamav3 quantize.py:1846-1856), kept so the checkpoint stays
        # exllamav3-native; validated at load, never read by the kernels
        tensors.update({
            "w13_mul1": torch.zeros(E, 2, dtype=torch.int32),
            "w13_mcg": torch.zeros(E, 2, dtype=torch.int32),
            "w2_mul1": torch.zeros(E, dtype=torch.int32),
            "w2_mcg": torch.zeros(E, dtype=torch.int32),
        })
        for name, t in tensors.items():
            p = Parameter(t, requires_grad=False)
            layer.register_parameter(name, p)
            kind = name.split("_", 1)[1]
            set_weight_attrs(p, {"weight_loader": self._make_loader(layer, kind)})

        # vLLM's strict "every weight loaded" check is off for quantized models
        # (model_loader/default_loader.py:487-497): track pieces ourselves
        # host-side on purpose: create_weights runs under vLLM's target-device context
        layer._exl3_loaded = torch.zeros((E, 3, len(_KINDS)), dtype=torch.bool, device="cpu")  # [slot, w1/w2/w3, kind]
        layer._exl3_codebook = {}                                                    # (slot, proj) -> mul1/mcg
        bl = bank.bl
        logger.debug("%s: EXL3 banks w13 %.1f MiB, w2 %.1f MiB, groups %s", self.prefix,
                     bl.total13 * 2 / 2**20, bl.total2 * 2 / 2**20,
                     [(g.k_gu, g.k_d, g.size) for g in self.lp.groups])

    def _make_loader(self, layer: torch.nn.Module, kind: str):
        def loader(param, loaded_weight, weight_name=None, shard_id=None, expert_id=None, return_success=False):
            ok = self._load_piece(layer, param, loaded_weight, kind, shard_id, expert_id)
            return ok if return_success else None

        loader.supports_moe_loading = True  # type: ignore[attr-defined]
        return loader

    def _load_piece(self, layer, param, w: torch.Tensor, kind: str, proj: str | None, expert_id) -> bool:
        if proj not in _PROJ_INDEX or expert_id is None:
            raise ValueError(f"{self.prefix}: EXL3 expert tensor without shard/expert id ({proj}, {expert_id})")
        e = int(expert_id)
        if not 0 <= e < self.lp.num_experts:
            return False
        p = self.lp.perm[e]
        if kind in ("mul1", "mcg"):
            v = int(w.reshape(-1)[0].item()) & 0xFFFFFFFF
            want = CODEBOOK_MUL1 if kind == "mul1" else CODEBOOK_MCG
            if v != want:
                raise ValueError(f"{self.prefix} expert {e} {proj}.{kind}: marker {v:#x}, want {want:#x}")
            layer._exl3_codebook[(p, proj)] = kind
            if proj == PROJ_DOWN:
                param.data[p] = 1
            else:
                param.data[p, 0 if proj == PROJ_GATE else 1] = 1
            return True
        assert self.bank is not None
        try:
            self.bank.load_piece(kind, proj, e, w, dst=param.data)
        except ValueError as err:
            raise ValueError(f"{self.prefix}: {err}") from err
        layer._exl3_loaded[p, _PROJ_INDEX[proj], _KINDS.index(kind)] = True
        return True

    # ---------------------------------------------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        bank = self.bank
        assert bank is not None
        E = bank.num_experts
        if not _is_dummy_load():
            missing = (~layer._exl3_loaded).nonzero().tolist()
            if missing:
                raise RuntimeError(f"{self.prefix}: {len(missing)} EXL3 expert pieces never loaded, "
                                   f"first (slot, proj, kind) = {missing[:4]}")
            cbs = set(layer._exl3_codebook.values())
            if cbs != {self.plan.codebook} or len(layer._exl3_codebook) != 3 * E:
                raise RuntimeError(f"{self.prefix}: codebook markers {cbs} ({len(layer._exl3_codebook)} seen), "
                                   f"plan says {self.plan.codebook}")
        # Re-read the Parameters: loaders may have replaced .data (e.g. a device move)
        bank.bind({n: getattr(layer, n).data for n in TENSOR_NAMES})
        self.groups, full = bank.tables()
        layer._exl3_tables = full     # keep the full pointer tables alive (groups alias them)
        dev = bank.t["w13_trellis"].device
        self.perm = torch.tensor(self.lp.perm, dtype=torch.int64, device=dev)

        rows_max = max(1, min(int(self.settings.coop_max_tokens), K.COOP_MAX_SLOTS // self.top_k))
        self.coop = K.get_coop_scratch(dev, rows_max, self.top_k, bank.hidden, bank.inter_local)
        self.fused = K.get_fused_scratch(dev, int(self.settings.fused_rows), bank.hidden, bank.inter_local)
        K.prime(dev, self.groups, self.coop, self.fused, bank.hidden, self.top_k, self.act_limit, self.mcg, self.mul1)
        self.par = None
        if K.env_int("VLLM_EXL3_MOE_PARALLEL_GROUPS", 0) and len(self.groups) > 1:
            self.par = K.get_parallel_coop(dev, rows_max, self.top_k, bank.hidden, bank.inter_local, K.MAX_K_GROUPS)
        # VLLM_EXL3_MOE_SMOE (default 1): decode through our exl3_smoe kernel (smoe.py) when the plan allows it
        self.smoe = None
        if K.env_int("VLLM_EXL3_MOE_SMOE", 1) and self.mul1:
            from . import smoe as S
            if S.supported(bank):
                self.smoe = S.SmoeLayer(bank, self.top_k)
                self.smoe.layer = self.layer_idx
                xw = torch.zeros((1, bank.hidden), dtype=torch.bfloat16, device=dev)     # eager warm-up before capture
                ids = torch.arange(self.top_k, dtype=torch.int32, device=dev).view(1, -1)
                self.smoe(xw, ids, torch.zeros((1, self.top_k), dtype=torch.float32, device=dev), self.perm,
                          self.act_limit)
                logger.debug_once("EXL3 routed experts: decode through exl3_smoe")
        from .hostcopy import release_stage
        release_stage()                     # loading is over: the pinned staging buffer is not needed any more
        gc.collect()
        _malloc_trim()

    # ---------------------------------------------------------------------------------------

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        """This rank's partial routed sum in x.dtype (see forward.routed_moe_forward for the paths).
        The shared expert (passed in) is NOT run here: the runner runs it and adds it before the
        single TP all-reduce (moe_runner.py:586-648, 480-514)."""
        assert self.perm is not None and self.coop is not None and self.fused is not None
        assert self.bank is not None
        if shared_experts is not None and self.mk_can_overlap_shared_experts:
            fused = self._apply_with_shared(x, topk_weights, topk_ids, shared_experts, shared_experts_input)
            if fused is not None:
                return fused
        out = routed_moe_forward(
            x, topk_weights, topk_ids, self.perm, self.groups, self.bank, self.coop, self.fused,
            self.act_limit, self.mcg, self.mul1, decode_kernel=self.settings.decode_kernel,
            par=getattr(self, "par", None), prefill_wait=aux_stream(), smoe=getattr(self, "smoe", None),
            pad=_padding_rows(x.shape[0]))
        l2pf.point("next", self.layer_idx, x.shape[0])     # the next layer's attention weights into L2 (l2pf.py)
        l2pf_draft.point("next", self.layer_idx, x.shape[0])   # dsv41 patch 0066: drafter blocks only
        return out

    def _attach_shared(self, shared_experts) -> bool:
        """Once per layer, on the first (eager) call: the shared MLP's EXL3 gate / up / down as smoe's extra slot.
        Anything unexpected (not EXL3, another codebook, other shapes) leaves the shared expert on its own path."""
        state = getattr(self, "_shared_ok", None)
        if state is not None:
            return state
        ok = False
        try:
            mlp = shared_experts._layer
            gu, dn = mlp.gate_up_proj, mlp.down_proj
            g_lins, d_lins = getattr(gu, "_exl3", None), getattr(dn, "_exl3", None)
            markers = [int(v) & 0xFFFFFFFF for v in torch.cat([gu.mul1.reshape(-1), dn.mul1.reshape(-1)]).tolist()]
            if (g_lins and d_lins and len(g_lins) == 2 and len(d_lins) == 1 and all(v == CODEBOOK_MUL1 for v in markers)):
                self.smoe.attach_shared(g_lins[0], g_lins[1], d_lins[0])
                ok = True
            else:
                logger.warning("EXL3 smoe shared slot off for layer %s: shared MLP is not mul1 EXL3 gate_up x2 / down",
                               self.layer_idx)
        except Exception as e:  # noqa: BLE001  (a layout we do not know: keep vLLM's path)
            logger.warning("EXL3 smoe shared slot off for layer %s: %s", self.layer_idx, e)
        self._shared_ok = ok
        if ok:
            logger.info_once("EXL3 routed experts: the shared expert runs inside exl3_smoe (VLLM_EXL3_MOE_SMOE_SHARED=1)")
        return ok

    def fused_moe_forward(self, moe, hidden_states: torch.Tensor, input_ids: torch.Tensor | None):
        """The MoE layer's output for a decode-size step from one exl3_smoe launch (router + top-k + routed + shared),
        TP-reduced as vLLM's runner would; None when the step or the layer does not qualify (the runner runs as usual).
        `moe`: the DeepseekV4MoE module (gate, shared_experts, routed_scaling_factor, reduce_results). Patch 0016 calls
        it at the top of DeepseekV4MoE._forward_fused_moe."""
        if not _SMOE_FUSED or self.smoe is None or self.settings.decode_kernel != "coop":
            return None
        x = hidden_states
        gate = getattr(moe, "gate", None)
        if gate is None or x.dim() != 2 or x.dtype != torch.bfloat16 or x.shape[0] < _SMOE_FUSED_MIN_ROWS:
            return None
        w = gate.weight
        if (not self.smoe.fused_ok(x.shape[0], w.shape[0]) or w.dtype != torch.bfloat16 or w.dim() != 2
                or getattr(gate, "tid2eid", None) is not None or not getattr(gate, "is_unquantized", True)
                or getattr(moe, "use_sequence_parallel", False)):
            return None
        if getattr(self, "_fused_state", None) is None:
            self._fused_state = self._fused_setup(moe)
        if not self._fused_state:
            return None
        shared, bias = self._fused_state
        bvl, sent, tok = getattr(gate, "bias_vl", None), int(getattr(moe, "image_sentinel_lo", 0) or 0), None
        if bvl is not None and sent:
            if input_ids is None:
                return None
            tok = input_ids if input_ids.dtype == torch.int64 else input_ids.to(torch.int64)
        else:
            bvl, sent = None, 0
        m = x.shape[0]
        out, _, _ = self.smoe.fused(x, w, bias if bias is not None else gate.e_score_correction_bias.data, self.perm,
                                    self.act_limit, float(moe.routed_scaling_factor), _padding_rows(m), shared,
                                    None if bvl is None else bvl.data, tok, sent)
        l2pf.point("next", self.layer_idx, m)
        l2pf_draft.point("next", self.layer_idx, m)            # dsv41 patch 0066: drafter blocks only
        if getattr(moe, "reduce_results", True) and self.tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _fused_setup(self, moe):
        """Once per layer (first eager call): the shared MLP as smoe's extra slot, the correction bias as fp32. False
        if anything is not what the fused launch takes (the layer then keeps vLLM's runner)."""
        try:
            gate = moe.gate
            cb = getattr(gate, "e_score_correction_bias", None)
            if cb is None or cb.dtype != torch.float32 or not cb.is_contiguous() or cb.numel() != gate.weight.shape[0]:
                logger.warning("EXL3 fused MoE off for layer %s: correction bias %s", self.layer_idx,
                               None if cb is None else (cb.dtype, tuple(cb.shape)))
                return False
            shared = False
            mlp = getattr(moe, "shared_experts", None)
            if mlp is not None:
                if not getattr(self.smoe, "has_shared", False):
                    gu, dn = mlp.gate_up_proj, mlp.down_proj
                    g_lins, d_lins = getattr(gu, "_exl3", None), getattr(dn, "_exl3", None)
                    markers = [int(v) & 0xFFFFFFFF for v in torch.cat([gu.mul1.reshape(-1), dn.mul1.reshape(-1)]).tolist()]
                    if not (g_lins and d_lins and len(g_lins) == 2 and len(d_lins) == 1
                            and all(v == CODEBOOK_MUL1 for v in markers)):
                        logger.warning("EXL3 fused MoE off for layer %s: shared MLP is not mul1 EXL3 gate_up x2 / down",
                                       self.layer_idx)
                        return False
                    self.smoe.attach_shared(g_lins[0], g_lins[1], d_lins[0])
                shared = True
            logger.info_once("EXL3 fused MoE: router + top-k + routed + shared experts in one launch "
                             "(VLLM_EXL3_MOE_FUSED=1)")
            return (shared, None)
        except Exception as e:  # noqa: BLE001  (a layout we do not know: keep vLLM's path)
            logger.warning("EXL3 fused MoE off for layer %s: %s", self.layer_idx, e)
            return False

    def _apply_with_shared(self, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        """Routed + shared in one exl3_smoe call when the decode path takes smoe; the runner's shared output is set to
        zeros (it adds shared + routed). Otherwise run the shared expert through the runner's own call (it skipped its
        launch because mk_can_overlap_shared_experts is True) and return None for the routed path."""
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder

        m = x.shape[0]
        if (self.smoe is not None and self.settings.decode_kernel == "coop" and m <= self.coop.rows_max
                and x.dtype == torch.bfloat16 and self.smoe.rows_ok(m, shared=True)
                and (shared_experts_input is None or shared_experts_input.data_ptr() == x.data_ptr())
                and self._attach_shared(shared_experts)):
            out = self.smoe(x, topk_ids, topk_weights, self.perm, self.act_limit, _padding_rows(m), shared=True)
            shared_experts._output[shared_experts._output_idx] = (
                _shared_zero_output(out) if _SHARED_ZEROS else torch.zeros_like(out)  # dsv41 patch 0079c
            )
            l2pf.point("next", self.layer_idx, m)
            l2pf_draft.point("next", self.layer_idx, m)        # dsv41 patch 0066: drafter blocks only
            return out
        shared_experts(shared_experts_input if shared_experts_input is not None else x,
                       SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
        return None
