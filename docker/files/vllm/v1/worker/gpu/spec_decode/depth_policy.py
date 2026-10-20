# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-round DSpark verify depth inside a fixed-shape verify window.

After each draft, every request gets the verify depth d in 0..k with the most
expected tokens per millisecond: expected tokens = 1 + the sum of the
confidence head's prefix survivals (running products of its per-position
acceptance probabilities), cost = the verify forward with d + 1 live rows plus
the drafter pass, from a measured table. The rule is TensorFold's DeepSeek-V4.1
depth choice (github.com/bertholomus/TensorFold, branch deepseek-v41-tp2,
src/tensorfold/families/deepseek_v41/cuda/multi.py ``_choose_k``, Apache-2.0),
written here for the V2 model runner.

The window keeps its shape, so captured CUDA graphs and attention backends are
untouched. Rows past a request's depth are marked as MoE padding (the router
gives them top-k ids of -1 and routed-expert kernels skip their weights) and
as invalid drafts (the rejection sampler never accepts them). Rows up to the
depth compute what a full window computes for them (attention is causal, MoE
and dense rows are independent), so the depth decides how many drafts can be
accepted and never which tokens come out.

dsv41 patch 0041 (draft 19.10.2026, opt_runde2), all off by default, so an unchanged
environment behaves exactly as before:

* "confb[:<ms by depth 0..k>]": the depths of all requests of a step are chosen
  together. A step's cost is one table of total live rows (TensorFold's multi-stream
  ``_choose_k`` prices a stream's rows next to the others' rows); the step maximizes
  (sum of expected tokens) / cost(sum of live rows). With one request this is exactly
  "conf". The single-stream depth table is read as rows 1..k+1, longer row counts
  continue with its mean slope; "confr:<ms for 1, 2, 3 ... rows>" gives the rows
  table directly (measured with fixed depths at 1..4 streams).
  Exact solution: every request's survivals fall with the position, so the best set
  of m extra rows is the m largest survivals (prefix-closed per request), and the
  step takes the best m over 0..n*k. ~12 small device ops, no host sync.
* "cal=1" (or VLLM_DSPARK_CONF_CALIB=1): the confidence head predicts acceptance as
  trained; at temperature > 0 a greedy draft is accepted with the target's probability
  of it, which is lower. Per position and separately for greedy / sampled requests,
  the policy keeps an EMA of accepted / predicted at the positions a round reached
  and scales the confidences by it (clamped 0.3..1.2, applied after 256 reached rows
  per position). Deterministic: both TP ranks see the same confidences and the same
  accepted counts.
* "ctx=L1:ms1,L2:ms2,...": extra milliseconds per request at its context length
  (linear between points, flat outside): the indexer and the attention of the whole
  window grow with the context while the table was measured on short prompts; a
  larger fixed cost makes extra rows relatively cheaper (only "confb"/"confr").
* Window cap: with a dynamic speculation schedule (num_speculative_tokens_per_batch_size)
  the next step verifies at most cap[n] drafts with n requests; slots past it are
  never priced or chosen.

Environment:
  VLLM_DSPARK_DEPTH_POLICY  "conf" (the rule above), "confb", "confr:<rows ms>",
                            "fixed:<d>" (every request verifies d drafts) or unset / "0" (off)
  VLLM_DSPARK_ROUND_MS      verify forward ms for 1..k+1 live rows, comma separated
  VLLM_DSPARK_DRAFT_MS      the drafter pass in ms
  VLLM_DSPARK_DEPTH_FILE    a file holding a policy string, re-read every 64
                            verify steps (to sweep depths in one serve);
                            "conf:<ms,...>" / "confb:<ms,...>" also replace the cost
                            table (round ms by depth 0..k, drafter included),
                            "confr:<ms,...>" the rows table; options after a space:
                            "cal=0|1", "ctx=L:ms,L:ms"
  VLLM_DSPARK_CONF_CALIB    1: calibration on from the start (file option overrides)
  VLLM_DSPARK_LOOKUP        prompt-lookup drafts (patch 0043), "min_n,max_n,p0[,window]"
                            or "0"; depth-file token "lookup=..." overrides. File options
                            missing from the file fall back to these environment values.
  VLLM_DSPARK_DEPTH_LOG     N > 0: every N verify steps, log the chosen depths and
                            the drafts verified / accepted (and the calibration)
"""

import os

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Our two-Spark serve (DeepSeek-V4.1-Flash, EXL3 routed experts): verify forward
# ms for 1..6 live rows and the drafter pass, until a measured table is given.
_DEFAULT_ROUND_MS = (38.0, 41.0, 44.0, 46.5, 48.5, 50.5)
_DEFAULT_DRAFT_MS = 4.5
_FILE_EVERY = 64
_ROWS_MAX = 64            # rows table length (4 requests x 6 rows = 24 in our serve)
_CAL_ALPHA = 0.02         # EMA weight of one step's counts
_CAL_MIN_REACHED = 256.0  # reached rows per position before a scale is applied
_CAL_LO, _CAL_HI = 0.3, 1.2


def _head(text: str | None) -> str:
    p = (text or "").strip().lower()
    return p.split()[0] if p else ""


def _parse_policy(text: str | None) -> tuple[str, int]:
    p = _head(text)
    if p in ("", "0", "off", "none"):
        return "off", 0
    if p == "conf" or p.startswith("conf:"):
        return "conf", 0
    if p == "confb" or p.startswith("confb:"):
        return "confb", 0
    if p.startswith("confr:"):
        return "confr", 0
    if p.startswith("fixed:"):
        return "fixed", max(0, int(p.split(":", 1)[1]))
    raise ValueError(
        f"VLLM_DSPARK_DEPTH_POLICY={text!r}: expected 'conf', 'confb', 'confr:<ms>', "
        "'fixed:<d>' or '0'"
    )


def _parse_table(text: str | None) -> list[float] | None:
    """The cost table of a "conf:" / "confb:" / "confr:" policy string."""
    p = _head(text)
    for pre in ("conf:", "confb:", "confr:"):
        if p.startswith(pre):
            return [float(x) for x in p[len(pre):].split(",") if x.strip()]
    return None


def _parse_options(text: str | None) -> dict[str, str]:
    """key=value tokens after the policy (e.g. "cal=1 ctx=32768:1.5,65536:3")."""
    out: dict[str, str] = {}
    for tok in (text or "").strip().lower().split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_ctx(spec: str | None) -> list[tuple[float, float]]:
    pts = []
    for item in (spec or "").split(","):
        if ":" in item:
            a, b = item.split(":", 1)
            pts.append((float(a), float(b)))
    return sorted(pts)


def depth_policy_requested() -> bool:
    return _parse_policy(os.environ.get("VLLM_DSPARK_DEPTH_POLICY"))[0] != "off"


def round_cost_ms(num_steps: int) -> list[float]:
    """Verify forward + drafter ms for depth 0..num_steps (1..num_steps+1 rows)."""
    text = os.environ.get("VLLM_DSPARK_ROUND_MS", "")
    rows = [float(x) for x in text.split(",") if x.strip()] or list(_DEFAULT_ROUND_MS)
    while len(rows) < num_steps + 1:
        # Past the table: the last measured step per row.
        slope = rows[-1] - rows[-2] if len(rows) > 1 else 0.0
        rows.append(rows[-1] + slope)
    draft_ms = float(os.environ.get("VLLM_DSPARK_DRAFT_MS") or _DEFAULT_DRAFT_MS)
    return [rows[d] + draft_ms for d in range(num_steps + 1)]


def rows_from_depth_table(depth_ms: list[float], length: int = _ROWS_MAX) -> list[float]:
    """Step ms by total live rows 1..length from a single-stream table by depth 0..k
    (rows 1..k+1); longer row counts continue with the table's mean slope."""
    t = list(depth_ms)
    slope = (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else 0.0
    while len(t) < length:
        t.append(t[-1] + slope)
    return t[:length]


def _extend_rows(rows_ms: list[float], length: int = _ROWS_MAX) -> list[float]:
    t = list(rows_ms)
    slope = t[-1] - t[-2] if len(t) > 1 else 0.0
    while len(t) < length:
        t.append(t[-1] + slope)
    return t[:length]


def choose_depth(confidence_probs: torch.Tensor, cost_ms: torch.Tensor) -> torch.Tensor:
    """[n, k] per-position acceptance probabilities -> [n] int32 depths in 0..k."""
    conf = confidence_probs.float().clamp(0.0, 1.0)
    survival = torch.cumprod(conf, dim=1)
    expected = torch.cat(
        (torch.ones_like(conf[:, :1]), 1.0 + torch.cumsum(survival, dim=1)), dim=1
    )
    # argmax keeps the first (shallowest) depth among equal rates.
    return torch.argmax(expected / cost_ms, dim=1).to(torch.int32)


def choose_depth_joint(
    conf: torch.Tensor,
    rows_ms: torch.Tensor,
    extra_ms: torch.Tensor | float = 0.0,
    num_live: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depths of all n requests of a step at once: [n, k] acceptance probabilities,
    rows_ms[r - 1] = step ms with r live rows, extra_ms = context / stream cost added
    to every choice -> [n] int32 depths maximizing sum(1 + survivals) / cost(rows)."""
    n, k = conf.shape
    surv = torch.cumprod(conf.float().clamp(0.0, 1.0), dim=1)            # [n, k], falls with the position
    # tie-break toward lower positions so the chosen set is prefix-closed per request
    pos = torch.arange(k, device=conf.device, dtype=torch.float32)
    key = (surv - pos * 1e-6).reshape(-1)
    order = torch.sort(key, descending=True, stable=True).indices          # deterministic on every rank
    s_sorted = surv.reshape(-1)[order]
    csum = torch.cumsum(s_sorted, dim=0)
    # num_live (device scalar): requests verifying next step (rows of the others carry
    # zero confidence and are never taken); None = all n
    base = float(n) if num_live is None else num_live.to(torch.float32)
    gain = torch.cat((torch.zeros(1, device=conf.device), csum)) + base     # m = 0 .. n*k extra rows
    rows = torch.arange(n * k + 1, device=conf.device) + (n if num_live is None else num_live)
    cost = rows_ms[(rows - 1).clamp(0, rows_ms.numel() - 1)] + extra_ms
    m_best = torch.argmax(gain / cost)                                       # first max: fewer rows on ties
    take = torch.arange(n * k, device=conf.device) < m_best
    chosen = torch.zeros(n * k, dtype=torch.bool, device=conf.device).scatter_(0, order, take)
    return chosen.view(n, k).sum(dim=1).to(torch.int32)


class DSparkDepthPolicy:
    def __init__(
        self,
        num_speculative_steps: int,
        max_num_reqs: int,
        device: torch.device,
        has_confidence: bool,
        depth_cap_by_reqs: list[int] | None = None,
    ):
        self.k = num_speculative_steps
        self._has_conf = has_confidence
        text = os.environ.get("VLLM_DSPARK_DEPTH_POLICY")
        self.mode, self.fixed = _parse_policy(text)
        if self.mode in ("conf", "confb", "confr") and not has_confidence:
            raise ValueError(
                "VLLM_DSPARK_DEPTH_POLICY=conf needs the draft model's confidence "
                "head, which this checkpoint does not load."
            )
        cost = round_cost_ms(self.k)
        self.cost_ms = torch.tensor(cost, dtype=torch.float32, device=device)
        rows = _parse_table(text) if self.mode == "confr" else None
        self.rows_ms = torch.tensor(
            _extend_rows(rows) if rows else rows_from_depth_table(cost),
            dtype=torch.float32, device=device,
        )
        self.depth = torch.full(
            (max_num_reqs,), self.k, dtype=torch.int32, device=device
        )
        self.file = os.environ.get("VLLM_DSPARK_DEPTH_FILE") or None
        self.log_every = int(os.environ.get("VLLM_DSPARK_DEPTH_LOG") or 0)
        self.num_records = 0
        self._observed = 0
        # Logging counters (device): depth histogram, drafts verified, accepted.
        self._hist = torch.zeros(self.k + 1, dtype=torch.int64, device=device)
        self._live = torch.zeros((), dtype=torch.int64, device=device)
        self._kept = torch.zeros((), dtype=torch.int64, device=device)
        # 0041: window cap by request count (dynamic speculation), calibration, context cost
        self.cap = None
        if depth_cap_by_reqs:
            self.cap = torch.tensor(
                [min(self.k, int(c)) for c in depth_cap_by_reqs], dtype=torch.int64, device=device
            )
        # options: environment defaults, overridden token by token by the policy string
        # (env or depth file), so every depth-file text fully states the options
        self._env_opts = {
            "cal": os.environ.get("VLLM_DSPARK_CONF_CALIB", "0"),
            "lookup": os.environ.get("VLLM_DSPARK_LOOKUP", "0"),
        }
        self._env_opts.update(_parse_options(text))
        self.calib = False
        self.lookup_opt = "0"
        self._rec_conf = torch.ones((max_num_reqs, self.k), dtype=torch.float32, device=device)
        self._rec_sampled = torch.zeros((max_num_reqs,), dtype=torch.int64, device=device)
        self._rec_lf = torch.full((max_num_reqs,), self.k, dtype=torch.int64, device=device)
        self._cal_hit = torch.zeros((2, self.k), dtype=torch.float32, device=device)
        self._cal_pred = torch.zeros((2, self.k), dtype=torch.float32, device=device)
        self._cal_seen = torch.zeros((2, self.k), dtype=torch.float32, device=device)
        self._phi = torch.ones((2, self.k), dtype=torch.float32, device=device)
        self._pos = torch.arange(self.k, dtype=torch.int64, device=device)
        self.ctx_pts: list[tuple[float, float]] = []
        self._ctx_x = self._ctx_y = None
        self._apply_options(self._env_opts)
        logger.info(
            "DSpark depth policy %s (k=%d), round cost ms by depth %s%s%s%s",
            self.describe(),
            self.k,
            [round(c, 2) for c in cost],
            f", window cap by requests {depth_cap_by_reqs}" if self.cap is not None else "",
            ", confidence calibration on" if self.calib else "",
            f", context ms {self.ctx_pts}" if self.ctx_pts else "",
        )

    def describe(self) -> str:
        return f"fixed:{self.fixed}" if self.mode == "fixed" else self.mode

    def _apply_options(self, opts: dict[str, str]) -> None:
        self.calib = opts.get("cal", "0") in ("1", "on", "true")
        self.lookup_opt = opts.get("lookup", "0")
        self.ctx_pts = _parse_ctx(opts.get("ctx"))
        if self.ctx_pts:
            dev = self.cost_ms.device
            self._ctx_x = torch.tensor([p[0] for p in self.ctx_pts], dtype=torch.float32, device=dev)
            self._ctx_y = torch.tensor([p[1] for p in self.ctx_pts], dtype=torch.float32, device=dev)
        else:
            self._ctx_x = self._ctx_y = None

    def reset(self, req_index: int) -> None:
        self.depth[req_index] = self.k

    def _maybe_reload(self) -> None:
        """Measurement knob: take the policy in VLLM_DSPARK_DEPTH_FILE. Rank 0 reads
        the file and broadcasts it, so every rank switches at the same step (ranks
        that chose different depths would accept different tokens)."""
        if self.file is None or self.num_records % _FILE_EVERY:
            return
        from vllm.distributed.parallel_state import get_tp_group

        try:
            tp = get_tp_group()
        except AssertionError:  # no model parallel group (single-process tests)
            tp = None
        text = None
        if tp is None or tp.rank_in_group == 0:
            try:
                with open(self.file) as f:
                    text = f.read()
            except OSError:
                text = None
        if tp is not None and tp.world_size > 1:
            text = tp.broadcast_object(text, src=0)
        if text is None:
            return
        try:
            mode, fixed = _parse_policy(text)
        except ValueError as e:
            logger.warning("DSpark depth file %s ignored: %s", self.file, e)
            return
        if mode in ("conf", "confb", "confr") and not self._has_conf:
            logger.warning("DSpark depth file %s: no confidence head", self.file)
            return
        table = _parse_table(text)
        if table:
            if mode == "confr":
                new_rows = torch.tensor(_extend_rows(table), dtype=torch.float32, device=self.rows_ms.device)
                if not torch.equal(new_rows, self.rows_ms):
                    self.rows_ms.copy_(new_rows)
                    logger.info("DSpark rows cost table -> %s", [round(c, 2) for c in table])
            else:
                while len(table) < self.k + 1:
                    table.append(table[-1] + (table[-1] - table[-2] if len(table) > 1 else 0.0))
                new = torch.tensor(table[: self.k + 1], dtype=torch.float32, device=self.cost_ms.device)
                if not torch.equal(new, self.cost_ms):
                    self.cost_ms.copy_(new)
                    self.rows_ms.copy_(torch.tensor(rows_from_depth_table(table[: self.k + 1]),
                                                    dtype=torch.float32, device=self.rows_ms.device))
                    logger.info("DSpark depth cost table -> %s", [round(c, 2) for c in table[: self.k + 1]])
        opts = {**self._env_opts, **_parse_options(text)}
        before = (self.calib, tuple(self.ctx_pts), self.lookup_opt)
        self._apply_options(opts)
        if (self.calib, tuple(self.ctx_pts), self.lookup_opt) != before:
            logger.info("DSpark depth options -> calibration %s, context ms %s, lookup %s",
                        self.calib, self.ctx_pts, self.lookup_opt)
        if (mode, fixed) != (self.mode, self.fixed):
            self.mode, self.fixed = mode, fixed
            logger.info("DSpark depth policy -> %s", self.describe())

    def _ctx_extra(
        self, seq_lens: torch.Tensor | None, n: int, decoding: torch.Tensor | None = None
    ) -> torch.Tensor | float:
        """Sum over the step's decoding requests of the context cost (ms), 0 without "ctx"."""
        if self._ctx_x is None or seq_lens is None:
            return 0.0
        L = seq_lens[:n].float()
        x, y = self._ctx_x, self._ctx_y
        i = torch.searchsorted(x, L).clamp(1, max(1, x.numel() - 1))
        if x.numel() == 1:
            v = y.expand_as(L)
        else:
            x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
            v = y0 + (y1 - y0) * ((L - x0) / (x1 - x0)).clamp(0.0, 1.0)
            v = torch.where(L <= x[0], y[0], torch.where(L >= x[-1], y[-1], v))
        if decoding is not None:
            v = v * decoding
        return v.sum()

    def record(
        self,
        confidence_probs: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_reqs: int,
        temperature: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        decoding: torch.Tensor | None = None,
        lookup_from: torch.Tensor | None = None,
    ) -> None:
        """Choose each request's depth for the drafts just proposed. decoding: [n] bool,
        requests that sampled this step (partial prefills verify nothing next step);
        lookup_from: [n] first draft position copied by the prompt lookup (0043), k = none."""
        if num_reqs == 0:
            return
        self._maybe_reload()
        self.num_records += 1
        idx = idx_mapping[:num_reqs].long()
        if self.mode == "off":
            depth = torch.full(
                (num_reqs,), self.k, dtype=torch.int32, device=self.depth.device
            )
        elif self.mode == "fixed":
            depth = torch.full(
                (num_reqs,),
                min(self.fixed, self.k),
                dtype=torch.int32,
                device=self.depth.device,
            )
        else:
            conf = confidence_probs[:num_reqs].float().clamp(0.0, 1.0)
            copied = None
            if lookup_from is not None:
                copied = self._pos[None, :] >= lookup_from[:, None]
            if self.calib or self.log_every > 0:
                # raw confidences and greedy / sampled kept per request for observe()
                sampled = (
                    (temperature[idx] > 0).long()
                    if temperature is not None
                    else torch.zeros_like(idx)
                )
                self._rec_conf.index_copy_(0, idx, conf)
                self._rec_sampled.index_copy_(0, idx, sampled)
                if lookup_from is not None:
                    self._rec_lf.index_copy_(0, idx, lookup_from.long())
                else:
                    self._rec_lf.index_fill_(0, idx, self.k)
                if self.calib:
                    # copied positions carry the lookup's measured acceptance already
                    scaled = (conf * self._phi[sampled]).clamp(0.0, 1.0)
                    conf = scaled if copied is None else torch.where(copied, conf, scaled)
            if self.cap is not None:
                cap = self.cap[min(num_reqs, self.cap.numel() - 1)]
                conf = torch.where(self._pos[None, :] < cap, conf, torch.zeros_like(conf))
            if self.mode == "conf":
                depth = choose_depth(conf, self.cost_ms)
            else:
                live = None
                if decoding is not None:
                    conf = conf * decoding[:, None]
                    live = decoding.sum()
                depth = choose_depth_joint(
                    conf, self.rows_ms, self._ctx_extra(seq_lens, num_reqs, decoding), live
                )
        self.depth.index_copy_(0, idx, depth)

    def dead_rows(
        self, expanded_idx_mapping: torch.Tensor, expanded_local_pos: torch.Tensor
    ) -> torch.Tensor:
        """Logits rows past their request's depth: local position p holds draft p."""
        return expanded_local_pos > self.depth[expanded_idx_mapping.long()]

    def _calibrate(self, idx: torch.Tensor, acc: torch.Tensor, live: torch.Tensor) -> None:
        """One step of the accepted / predicted EMA per (greedy | sampled, position)."""
        p = self._pos[None, :]
        own = (p < live[:, None]) & (p < self._rec_lf[idx][:, None])        # DSpark's own drafts
        reached = (p <= acc[:, None]) & own                                  # [n, k]
        hit = ((p < acc[:, None]) & own).float()
        pred = self._rec_conf[idx] * reached.float()
        g = self._rec_sampled[idx]
        one = torch.nn.functional.one_hot(g, 2).float()                       # [n, 2]
        hit_s = one.t() @ hit                                                 # [2, k]
        pred_s = one.t() @ pred
        seen_s = one.t() @ reached.float()
        a = _CAL_ALPHA
        upd = seen_s > 0
        self._cal_hit.copy_(torch.where(upd, (1 - a) * self._cal_hit + a * hit_s, self._cal_hit))
        self._cal_pred.copy_(torch.where(upd, (1 - a) * self._cal_pred + a * pred_s, self._cal_pred))
        self._cal_seen += seen_s
        phi = (self._cal_hit / self._cal_pred.clamp_min(1e-6)).clamp(_CAL_LO, _CAL_HI)
        self._phi.copy_(torch.where(self._cal_seen >= _CAL_MIN_REACHED, phi, torch.ones_like(phi)))

    def observe(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_drafts_per_req: np.ndarray | None,
        num_reqs: int,
    ) -> None:
        """Count drafts verified and accepted; calibrate; log every VLLM_DSPARK_DEPTH_LOG steps."""
        if num_reqs == 0 or num_drafts_per_req is None:
            return
        if self.log_every <= 0 and not (self.calib and self.mode in ("conf", "confb", "confr")):
            return
        idx = idx_mapping[:num_reqs].long()
        depth = self.depth[idx].long()
        drafts = torch.from_numpy(
            np.ascontiguousarray(num_drafts_per_req[:num_reqs], dtype=np.int64)
        ).to(depth.device, non_blocking=True)
        live = torch.minimum(depth, drafts)
        acc = (num_sampled[:num_reqs].long() - 1).clamp(min=0)
        if self.calib and self.mode in ("conf", "confb", "confr"):
            self._calibrate(idx, torch.minimum(acc, live), live)
        if self.log_every <= 0:
            return
        self._hist.index_add_(0, depth.clamp(max=self.k), torch.ones_like(depth))
        self._live += live.sum()
        self._kept += acc.sum()
        self._observed += 1
        if self._observed % self.log_every == 0:
            hist = self._hist.tolist()
            live_n, kept, rounds = int(self._live), int(self._kept), sum(hist)
            logger.info(
                "DSpark depth %s: depths %s, drafts verified %d, accepted %d "
                "(%.1f%%), tokens per verify %.2f%s",
                self.describe(),
                " ".join(f"{d}:{n}" for d, n in enumerate(hist) if n),
                live_n,
                kept,
                100.0 * kept / max(live_n, 1),
                (kept + rounds) / max(rounds, 1),
                (", calibration greedy %s sampled %s" % (
                    [round(v, 2) for v in self._phi[0].tolist()],
                    [round(v, 2) for v in self._phi[1].tolist()]))
                if self.calib else "",
            )
            self._hist.zero_()
            self._live.zero_()
            self._kept.zero_()
