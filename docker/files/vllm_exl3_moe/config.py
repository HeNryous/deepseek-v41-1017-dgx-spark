# SPDX-License-Identifier: Apache-2.0
"""Quantization config `deepseek_v41_exl3`: DeepSeek-V4.1's own FP8/MXFP8 handling everywhere,
EXL3 trellis banks for the backbone routed experts only.

Upstream anchors (vLLM main @ 6e517b15c1833cf72a7f557ee32524d98682e617):
  * register_quantization_config      model_executor/layers/quantization/__init__.py:61-109
  * custom methods are probed first    config/model.py:1397-1431 (_verify_quantization)
  * from_config(hf quantization_config) model_executor/model_loader/weight_utils.py:247-265
  * the V4.1 base class we extend      models/deepseek_v41/quant_config.py:33-231 (what
    "deepseek_v4_fp8" resolves to on CUDA, quantization/__init__.py:120-128)
  * Fp8Config.from_config keys on "fp8" in quant_method   quantization/fp8.py:158-175

Only RoutedExperts of backbone layers listed in the plan are redirected. Everything else goes to
super(): MXFP8 [32, 32] linears (attention, shared expert, compressor, indexer) via
ModelOptLinearMethod, the DSpark drafter's MXFP4 experts (layer index >= num_hidden_layers, see
models/deepseek_v41/nvidia/dspark.py:126-128) via Mxfp4MoEMethod, and any layer the plan leaves out.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

from .plan import Exl3MoEPlan

logger = init_logger(__name__)

QUANT_METHOD = "deepseek_v41_exl3"
_ROUTED_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts(?:\.|$)")


_ENGRAM_WKV = re.compile(r"layers\.(\d+)\.engram\.wkv$")   # V4.1 Engram layers 1 and 14 (exl3_moe.dense.engram.wkv: K)

@dataclass
class Exl3Settings:
    """The `exl3_moe` block inside config.json's quantization_config."""
    plan_file: str | None = "exl3_moe_plan.json"   # relative to the model dir
    plan: dict[str, Any] | None = None              # inline plan (wins over plan_file)
    decode_kernel: str = "coop"                     # coop (exl3_moe_coop) | fused (exl3_moe)
    coop_max_tokens: int = 32                       # rows per call on the coop path (x top_k <= 256)
    fused_rows: int = 256                           # exl3_moe rows per expert before reconstruct
    tp_layout: str = "channels"                     # channels = vLLM MoE TP split (only one in v0)
    exllamav3_version: str | None = None            # informational: encoder version
    head: str = "bf16"                              # bf16 (checkpoint) | mxfp8 (Marlin, needs patch 0005)
                                                    # | exl3 (EXL3 tensors head.*, needs 0005 + 0007)
    head_k: int | None = None                       # K of the EXL3 head
    dense: dict[str, Any] | None = None             # EXL3 dense linears (dense.py, needs patch 0006):
    #   {"k": {"attn.wq_a": 5, "attn.wq_b": 5, "attn.wo_a": 5, "attn.wo_b": 5, "ffn.shared_experts.w1": 5,
    #          "ffn.shared_experts.w2": 5}, "layers": [first, last]}
    #   keys are checkpoint module names; w1 covers the gate_up pair (w1 + w3), wq_a the fused_wqa_wkv pair (wq_a + wkv)
    #   "drafter": {"attn.wq_a": 5, "attn.wq_b": 5, "attn.wo_b": 5, "ffn.shared_experts.w1": 5,
    #               "ffn.shared_experts.w2": 5}: the DSpark drafter's dense linears from mtp.* EXL3 tensors
    #   (scripts/quantize/p4_drafter.py); modules not listed stay FP8 (wo_a always: its grouped FP8 path). main_proj
    #   reads the target's raw residual stream (|x| up to ~1.6e5): above 16 rows (exllamav3's path) its input is cast
    #   to fp16 under a power-of-two scale (dense.py), which overflowed to NaN without it (LOG Oct 10 09:18)
    #   "engram": {"wkv": 4}: V4.1's Engram projections (layers.<1|14>.engram.wkv, FP8 6144 -> 25600 column-parallel) on
    #   EXL3 at that K, from the variant's layers.<i>.engram.wkv.* tensors (scripts/quantize/p4_engram.py)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Exl3Settings":
        s = cls(raw=dict(d))
        for k in ("plan_file", "plan", "decode_kernel", "coop_max_tokens", "fused_rows", "tp_layout",
                  "exllamav3_version", "head", "head_k", "dense"):
            if k in d:
                setattr(s, k, d[k])
        # runtime overrides for A/B sweeps (must be identical on every rank)
        s.decode_kernel = os.environ.get("VLLM_EXL3_MOE_DECODE_KERNEL", s.decode_kernel)
        s.coop_max_tokens = int(os.environ.get("VLLM_EXL3_MOE_COOP_MAX_TOKENS", s.coop_max_tokens))
        s.fused_rows = int(os.environ.get("VLLM_EXL3_MOE_FUSED_ROWS", s.fused_rows))
        s.head = os.environ.get("VLLM_EXL3_HEAD", s.head)
        if s.head not in ("bf16", "mxfp8", "exl3"):
            raise ValueError(f"exl3_moe.head must be bf16, mxfp8 or exl3, got {s.head!r}")
        if s.head == "exl3" and not s.head_k:
            raise ValueError("exl3_moe.head exl3 needs head_k (the K the head was encoded at)")
        if s.dense is not None:
            from .dense import CKPT_NAME
            bad = set(s.dense.get("k", {})) - set(CKPT_NAME.values())
            if bad:
                raise ValueError(f"exl3_moe.dense.k: unknown modules {sorted(bad)}; known {sorted(set(CKPT_NAME.values()))}")
            bad = set(s.dense.get("drafter", {})) - ((set(CKPT_NAME.values()) - {"attn.wo_a"}) | {"main_proj"})
            if bad:
                raise ValueError(f"exl3_moe.dense.drafter: unsupported modules {sorted(bad)}")
        if s.decode_kernel not in ("coop", "fused"):
            raise ValueError(f"exl3_moe.decode_kernel must be coop or fused, got {s.decode_kernel!r}")
        if s.tp_layout != "channels":
            raise NotImplementedError("exl3_moe.tp_layout: only 'channels' (vLLM's MoE TP split) is implemented")
        return s

    def drafter_k(self, module: str) -> int | None:
        """K for a DSpark drafter module (vLLM path, or "main_proj"), or None to keep the checkpoint's FP8."""
        from .dense import CKPT_NAME
        if not self.dense:
            return None
        k = self.dense.get("drafter", {}).get(CKPT_NAME.get(module, module))
        return int(k) if k else None

    def dense_k(self, module: str, layer: int) -> int | None:
        """K for vLLM module path `module` of backbone layer `layer`, or None to keep the checkpoint's FP8."""
        from .dense import CKPT_NAME
        if not self.dense:
            return None
        lo, hi = self.dense.get("layers", [0, 1 << 30])
        if not lo <= layer <= hi:
            return None
        k = self.dense.get("k", {}).get(CKPT_NAME[module])
        return int(k) if k else None



def _resolve_plan_file(name: str) -> str:
    """plan_file relative to the model directory, or fetched from the HF repo the model came from.
    Not tested: the HF-hub branch (local snapshot dirs are the tested case)."""
    if os.path.isabs(name):
        return name
    from vllm.config import get_current_vllm_config

    mc = get_current_vllm_config().model_config
    model = str(mc.model)
    if os.path.isdir(model):
        return os.path.join(model, name)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model, filename=name, revision=getattr(mc, "revision", None))


def _num_hidden_layers() -> int | None:
    try:
        from vllm.config import get_current_vllm_config

        return int(get_current_vllm_config().model_config.hf_text_config.num_hidden_layers)
    except Exception:  # noqa: BLE001
        return None


def routed_layer_index(prefix: str) -> int | None:
    m = _ROUTED_LAYER_RE.search(prefix)
    return int(m.group(1)) if m else None


class DeepseekV41Exl3Config(DeepseekV4FP8Config):
    """`quant_method: "deepseek_v41_exl3"`: the V4.1 FP8 config plus EXL3 routed-expert banks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exl3_settings = Exl3Settings()
        self._exl3_plan: Exl3MoEPlan | None = None

    @classmethod
    def get_name(cls):  # type: ignore[override]
        return QUANT_METHOD

    @classmethod
    def get_min_capability(cls) -> int:
        # mma.sync m16n8k16 + cp.async + 90 KB dynamic smem: sm_80 and up (exllamav3 arch.cuh:5-27)
        return max(80, super().get_min_capability())

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # Claim only our own checkpoints. Without this override the inherited V4.1 one would claim
        # every "fp8" DeepSeek-V4 checkpoint (quant_config.py:138-167) from the front of the probe
        # list where custom methods sit (config/model.py:1397-1404).
        if isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get("quant_method") == QUANT_METHOD:
            return QUANT_METHOD
        if user_quant == QUANT_METHOD:
            return QUANT_METHOD
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepseekV41Exl3Config":
        settings = Exl3Settings.from_dict(dict(config.get("exl3_moe") or {}))
        base = {k: v for k, v in config.items() if k != "exl3_moe"}
        # Fp8Config.from_config derives is_checkpoint_fp8_serialized from "fp8" in quant_method
        # (fp8.py:159-160): hand it the original FP8 description of the non-expert tensors.
        base["quant_method"] = "fp8"
        inst = super().from_config(base)
        assert isinstance(inst, DeepseekV41Exl3Config)
        inst.exl3_settings = settings
        return inst

    @property
    def exl3_plan(self) -> Exl3MoEPlan:
        if self._exl3_plan is None:
            s = self.exl3_settings
            if s.plan is not None:
                self._exl3_plan = Exl3MoEPlan.from_dict(s.plan)
            elif s.plan_file:
                path = _resolve_plan_file(s.plan_file)
                with open(path) as f:
                    self._exl3_plan = Exl3MoEPlan.from_dict(json.load(f))
                logger.info("EXL3 MoE plan %s: %d layers, %.3f bpw, groups %s", path,
                            len(self._exl3_plan.layers), self._exl3_plan.bpw(),
                            self._exl3_plan.group_histogram())
            else:
                raise ValueError("quantization_config.exl3_moe needs plan or plan_file")
        return self._exl3_plan

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and self.exl3_settings.dense:
            from .dense import Exl3DenseLinearMethod, dense_module

            dm = dense_module(prefix)
            n_backbone = _num_hidden_layers()
            if dm is not None and (n_backbone is None or dm[0] < n_backbone):
                k = self.exl3_settings.dense_k(dm[1], dm[0])
                if k:
                    return Exl3DenseLinearMethod(dm[1], k, layer_idx=dm[0])
            elif dm is not None and dm[1] != "attn.wo_a":     # DSpark drafter layers (index >= num_hidden_layers)
                k = self.exl3_settings.drafter_k(dm[1])
                if k:
                    method = Exl3DenseLinearMethod(dm[1], k, layer_idx=dm[0], prefetch=False)
                    from . import l2pf_draft                   # dsv41 patch 0066: the drafter's own L2 chain
                    method.draft_pf = l2pf_draft.enabled()
                    return method
            elif prefix.endswith("main_proj"):                 # the drafter's context combiner (replicated)
                k = self.exl3_settings.drafter_k("main_proj")
                if k:
                    return Exl3DenseLinearMethod("main_proj", k, layer_idx=-1, prefetch=False)
            else:
                m = _ENGRAM_WKV.search(prefix)
                k = int(self.exl3_settings.dense.get("engram", {}).get("wkv", 0)) if m else 0
                if k:                                         # Engram's hash-row projection (column-parallel, gathered)
                    return Exl3DenseLinearMethod("engram.wkv", k, layer_idx=int(m.group(1)), prefetch=False)
        if isinstance(layer, ParallelLMHead) and prefix.endswith("lm_head"):
            # Only reached with patch 0005 (V4.1 builds its head without a quant config otherwise). Only the lm_head:
            # another vocab projection handed this config (e.g. a DSpark Markov head, markov_w2) has no head.* pieces.
            if self.exl3_settings.head == "mxfp8":
                from .head import Mxfp8MarlinHeadMethod

                return Mxfp8MarlinHeadMethod()
            if self.exl3_settings.head == "exl3":
                from .head import Exl3HeadMethod

                return Exl3HeadMethod(self.exl3_settings.head_k)
            return None                                  # -> UnquantizedEmbeddingMethod, as without the patch
        if isinstance(layer, RoutedExperts):
            idx = routed_layer_index(prefix)
            n_backbone = _num_hidden_layers()
            plan = self.exl3_plan
            is_backbone = idx is not None and (n_backbone is None or idx < n_backbone)
            if (is_backbone and idx in plan.layers) or (idx is not None and not is_backbone and idx in plan.drafter_layers):
                from .moe_method import Exl3RoutedMoEMethod

                return Exl3RoutedMoEMethod(layer.moe_config, plan, idx, self.exl3_settings, prefix)
            # Layers the plan leaves native keep the checkpoint's own MXFP4 method: the DSpark drafter's experts
            # (idx >= num_hidden_layers) unless the plan's "drafter" section lists them (p4_drafter_experts.py).
        method = super().get_quant_method(layer, prefix)
        if isinstance(layer, LinearBase) and method is not None:
            from . import hcap
            if hcap.enabled(prefix):                     # VLLM_EXL3_HCAP: input Hessians of FP8 linears (hcap.py)
                return hcap.HCaptureMethod(method, prefix)
        return method


def register_config() -> None:
    """Idempotent: vLLM may load general plugins several times per process (plugins/__init__.py:77-91)."""
    register_quantization_config(QUANT_METHOD)(DeepseekV41Exl3Config)
