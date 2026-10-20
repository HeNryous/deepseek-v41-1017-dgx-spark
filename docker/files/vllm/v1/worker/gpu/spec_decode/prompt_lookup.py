# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41 patch 0043: prompt-lookup drafts on top of DSpark's greedy drafts.

After DSpark proposes its block of k drafts, each request's token history (prompt
and output, already holding this step's samples) is searched for an earlier
occurrence of its last tokens: the longest suffix match of min_n..max_n tokens
(the most recent one among equally long matches) that still has k tokens after
it. Where those copied tokens first differ from DSpark's drafts, they replace
DSpark's drafts from that position on, if the request's running lookup
acceptance is higher than the confidence head's acceptance probability for
DSpark's draft at that position. Replaced positions carry the lookup acceptance
as their confidence, so the depth policy prices them like any other draft.

Exactness: verification is unchanged. Greedy drafts cache no draft distribution,
so the rejection sampler treats every draft as a point mass; substituting another
deterministic token keeps the output tokens (T = 0) and the output distribution
(T > 0), only how many drafts get accepted changes. With probabilistic drafting
(cached draft logits) the module stays off.

Lookup acceptance: per request, decayed counts of the lookup positions that
verification reached and accepted, shrunk toward the same counts over all
requests (new requests start there, the very first ones at p0). Everything stays
on the device: no host sync per step.

Option (depth file token or VLLM_DSPARK_LOOKUP): "lookup=<min_n>,<max_n>,<p0>[,<window>]"
(window > 0: search only the last window tokens), "lookup=0" off.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_POS_BITS = 40
_POS_MASK = (1 << _POS_BITS) - 1
_BLOCK = 1024
_PROGRAMS = 16
_DECAY = 0.9  # per verify of a request with lookup positions
_DECAY_GLOBAL = 0.99  # per verify step with lookup positions
_PRIOR_REQ = 4.0  # lookup positions the global estimate weighs in a request's
_PRIOR_GLOBAL = 16.0  # lookup positions p0 weighs in the global estimate
_P_LO, _P_HI = 0.02, 0.98


def parse_lookup(text: str | None) -> tuple[int, int, float, int] | None:
    """"min_n,max_n,p0[,window]" -> tuple, None for off ("", "0", "off")."""
    p = (text or "").strip().lower()
    if p in ("", "0", "off", "none"):
        return None
    parts = [x for x in p.split(",") if x.strip()]
    if len(parts) < 3:
        raise ValueError(f"lookup={text!r}: expected min_n,max_n,p0[,window]")
    min_n, max_n = max(1, int(parts[0])), int(parts[1])
    p0 = min(max(float(parts[2]), _P_LO), _P_HI)
    window = int(parts[3]) if len(parts) > 3 else 0
    return min_n, max(min_n, max_n), p0, max(0, window)


@triton.jit
def _suffix_match_kernel(
    tokens_ptr,
    tokens_stride,
    total_len_ptr,
    idx_mapping_ptr,
    best_ptr,
    window,
    MIN_N: tl.constexpr,
    MAX_N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    POS_BITS: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + b)
    if req < 0:
        return
    length = tl.load(total_len_ptr + req)
    row = tokens_ptr + req.to(tl.int64) * tokens_stride
    # A candidate c starts the copied block: tokens[c - n : c] equal the last n
    # tokens and tokens[c : c + K] exist (c + K <= length, so c never is the end).
    first = tl.maximum(length - window, MIN_N)
    last = length - K
    step = tl.num_programs(1) * BLOCK
    offs = tl.arange(0, BLOCK)
    best = tl.full([BLOCK], -1, tl.int64)
    for start in range(first + tl.program_id(1) * BLOCK, last + 1, step):
        c = start + offs
        run = c <= last
        n = tl.zeros([BLOCK], dtype=tl.int32)
        for t in tl.static_range(MAX_N):
            s = tl.load(row + length - 1 - t, mask=length - 1 - t >= 0, other=-2)
            ok = run & (c - 1 - t >= 0)
            a = tl.load(row + c - 1 - t, mask=ok, other=-1)
            run = ok & (a == s)
            n += run.to(tl.int32)
        score = tl.where(
            n >= MIN_N, (n.to(tl.int64) << POS_BITS) | c.to(tl.int64), -1
        )
        best = tl.maximum(best, score)
    top = tl.max(best, axis=0)
    if top >= 0:
        tl.atomic_max(best_ptr + b, top)


def suffix_match_reference(
    tokens: list[int], min_n: int, max_n: int, k: int, window: int = 0
) -> int:
    """Pure-Python twin of the kernel for one request (tests): packed best or -1."""
    length = len(tokens)
    first = max(length - window, min_n) if window > 0 else min_n
    best = -1
    for c in range(first, length - k + 1):
        n = 0
        for t in range(max_n):
            if c - 1 - t < 0 or length - 1 - t < 0:
                break
            if tokens[c - 1 - t] != tokens[length - 1 - t]:
                break
            n += 1
        if n >= min_n:
            best = max(best, (n << _POS_BITS) | c)
    return best


class PromptLookup:
    def __init__(self, num_speculative_steps: int, max_num_reqs: int, device):
        self.k = num_speculative_steps
        self.device = torch.device(device)
        self.opts: tuple[int, int, float, int] | None = None
        self._text: str | None = None
        f32 = dict(dtype=torch.float32, device=self.device)
        # per-request state has one spare slot (index max_num_reqs) for masked batch
        # rows (idx_mapping -1); they never hit, their updates land there
        self._spare = max_num_reqs
        slots = max_num_reqs + 1
        self.p0 = torch.full((), 0.5, **f32)
        self.p_req = torch.full((slots,), 0.5, **f32)
        self._r = torch.zeros(slots, **f32)
        self._a = torch.zeros(slots, **f32)
        self._rg = torch.zeros((), **f32)
        self._ag = torch.zeros((), **f32)
        self.p_glob = torch.full((), 0.5, **f32)
        self.lookup_from = torch.full(
            (slots,), self.k, dtype=torch.int64, device=self.device
        )
        self._best = torch.full(
            (max_num_reqs,), -1, dtype=torch.int64, device=self.device
        )
        self._pos = torch.arange(self.k, dtype=torch.int64, device=self.device)
        # logging counters: steps with requests, hits, replaced, positions reached, accepted
        self._log = torch.zeros(5, dtype=torch.int64, device=self.device)
        self.log_every = int(os.environ.get("VLLM_DSPARK_DEPTH_LOG") or 0)
        self._observed = 0

    @property
    def enabled(self) -> bool:
        return self.opts is not None

    def configure(self, text: str | None) -> None:
        if text == self._text:
            return
        self._text = text
        try:
            opts = parse_lookup(text)
        except ValueError as e:
            logger.warning("DSpark prompt lookup option ignored: %s", e)
            return
        if opts is not None and (self.opts is None or opts[2] != self.opts[2]):
            self.p0.fill_(opts[2])
            # the global estimate restarts from the new prior
            self._rg.zero_()
            self._ag.zero_()
            self.p_glob.fill_(opts[2])
        self.opts = opts
        logger.info(
            "DSpark prompt lookup %s",
            "off" if opts is None
            else "min_n %d max_n %d p0 %.2f window %s" % (
                opts[0], opts[1], opts[2], opts[3] or "all"),
        )

    def reset(self, req_index: int) -> None:
        self._r[req_index] = 0.0
        self._a[req_index] = 0.0
        self.p_req[req_index] = self.p_glob
        self.lookup_from[req_index] = self.k

    def _slots(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.long()
        return torch.where(idx < 0, torch.full_like(idx, self._spare), idx)

    def _search(self, all_token_ids, total_len, idx, n: int) -> torch.Tensor:
        min_n, max_n, _, window = self.opts
        best = self._best[:n]
        best.fill_(-1)
        if self.device.type != "cuda":
            for b, r in enumerate(idx.tolist()):
                if r < 0:
                    continue
                toks = all_token_ids[r, : int(total_len[r])].tolist()
                best[b] = suffix_match_reference(toks, min_n, max_n, self.k, window)
            return best
        _suffix_match_kernel[(n, _PROGRAMS)](
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            idx,
            best,
            window if window > 0 else all_token_ids.shape[1],
            MIN_N=min_n,
            MAX_N=max_n,
            K=self.k,
            BLOCK=_BLOCK,
            POS_BITS=_POS_BITS,
            num_warps=4,
        )
        return best

    def apply(
        self,
        draft_tokens: torch.Tensor,
        confidence: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_reqs: int,
        all_token_ids: torch.Tensor,
        total_len: torch.Tensor,
    ) -> torch.Tensor:
        """[n, k] DSpark drafts -> drafts with copied tokens where the lookup wins;
        rewrites confidence[:n] (the speculator's buffer) for replaced positions."""
        if self.opts is None or num_reqs == 0:
            return draft_tokens
        n = num_reqs
        idx = idx_mapping[:n]
        best = self._search(all_token_ids, total_len, idx, n)
        hit = best >= 0
        start = torch.where(hit, best & _POS_MASK, torch.zeros_like(best))
        pos = start[:, None] + self._pos[None, :]
        slots = self._slots(idx)
        rows = idx.long().clamp(min=0)
        copied = all_token_ids[rows[:, None], pos].to(draft_tokens.dtype)
        drafts = draft_tokens[:n]
        conf = confidence[:n]
        diff = (copied != drafts) & hit[:, None]
        has_diff = diff.any(dim=1)
        first = torch.where(
            has_diff, diff.to(torch.int32).argmax(dim=1), torch.full_like(best, self.k)
        )
        p = self.p_req[slots]
        conf_at = conf.gather(1, first.clamp(max=self.k - 1)[:, None]).squeeze(1)
        use = has_diff & (p > conf_at)
        replace = use[:, None] & (self._pos[None, :] >= first[:, None])
        conf.copy_(torch.where(replace, p[:, None].to(conf.dtype), conf))
        self.lookup_from.index_copy_(
            0, slots, torch.where(use, first, torch.full_like(first, self.k))
        )
        self._log[0] += 1
        self._log[1] += hit.sum()
        self._log[2] += use.sum()
        return torch.where(replace, copied, drafts)

    def lookup_from_batch(self, idx_mapping: torch.Tensor, num_reqs: int) -> torch.Tensor:
        return self.lookup_from[self._slots(idx_mapping[:num_reqs])]

    def observe(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        depth: torch.Tensor | None,
        cu_num_logits: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Count the lookup positions this verify reached / accepted."""
        if self.opts is None or num_reqs == 0:
            return
        n = num_reqs
        idx = self._slots(idx_mapping[:n])
        lf = self.lookup_from[idx]
        drafts = (cu_num_logits[1 : n + 1] - cu_num_logits[:n] - 1).long().clamp(min=0)
        live = (
            drafts if depth is None
            else torch.minimum(depth[idx_mapping[:n].long().clamp(min=0)].long(), drafts)
        )
        acc = torch.minimum((num_sampled[:n].long() - 1).clamp(min=0), live)
        reached = (torch.minimum(live, acc + 1) - lf).clamp(min=0).float()
        accepted = (torch.minimum(live, acc) - lf).clamp(min=0).float()
        seen = reached > 0
        r = torch.where(seen, _DECAY * self._r[idx] + reached, self._r[idx])
        a = torch.where(seen, _DECAY * self._a[idx] + accepted, self._a[idx])
        self._r.index_copy_(0, idx, r)
        self._a.index_copy_(0, idx, a)
        tot_r, tot_a = reached.sum(), accepted.sum()
        any_seen = tot_r > 0
        self._rg.copy_(torch.where(any_seen, _DECAY_GLOBAL * self._rg + tot_r, self._rg))
        self._ag.copy_(torch.where(any_seen, _DECAY_GLOBAL * self._ag + tot_a, self._ag))
        self.p_glob.copy_(
            (self._ag + _PRIOR_GLOBAL * self.p0) / (self._rg + _PRIOR_GLOBAL)
        )
        p = (a + _PRIOR_REQ * self.p_glob) / (r + _PRIOR_REQ)
        self.p_req.index_copy_(0, idx, p.clamp(_P_LO, _P_HI))
        self.lookup_from.index_fill_(0, idx, self.k)
        self._log[3] += reached.sum().long()
        self._log[4] += accepted.sum().long()
        self._observed += 1
        if self.log_every > 0 and self._observed % self.log_every == 0:
            logger.info(self.log_line())

    def log_line(self) -> str:
        """Counters since the last call (host sync: logging only)."""
        steps, hits, used, reached, accepted = self._log.tolist()
        self._log.zero_()
        return (
            "DSpark lookup: steps %d, hits %d, replaced %d, lookup drafts verified %d, "
            "accepted %d (%.1f%%), p global %.2f"
            % (steps, hits, used, reached, accepted,
               100.0 * accepted / max(reached, 1), float(self.p_glob))
        )
