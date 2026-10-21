# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41 patch 0066: the DSpark Markov loop's greedy steps in three launches each (VLLM_DSV41_MARKOV_FUSED=1).

With VLLM_DSPARK_LOCAL_ARGMAX=1 a draft step (dspark.py draft_argmax_local) ran ~13 kernels: the Markov embedding
lookup, the Markov bias (markov_cache.py), the bf16 add of base logits and bias, torch.max over the rank's 64,640
vocabulary columns (15 us of a ~46 us step in the F15 trace), four casts and a stack, the (value, index) all-gather,
argmax over the ranks, gather, cast and the draft column write. Here a step is:
  1. markov_bias_rows: the unchanged bias kernel on the same inputs (the same bias bits; cached rows are copies),
  2. _partial_kernel: base + bias rounded to the add's dtype as torch rounds it (fp32 sum, round to nearest even),
     then per CHUNK columns the maximum and its first column, kept as an order key: torch.max's order (GreaterOrNan:
     NaN above everything, -0.0 equal to +0.0, ties to the lower index),
  3. one all-gather of every rank's chunk maxima ([rows, 2 * chunks] int32 a rank) instead of the (value, index) pair,
  4. _pick_kernel: the first maximum over (rank, chunk), rank-major. That is torch.max per rank followed by the argmax
     over the ranks (lower rank on ties, a NaN wins), so the same token. It writes the draft column, the previous token
     of the next step and that token's Markov embedding (a copy of its markov_w1 row) for the next step's bias and for
     the confidence head.
Step 0 starts with _first_kernel (anchor token and its embedding, one launch instead of an index and a lookup).
The drafts are the unfused path's bit for bit. Drafts only propose: a different draft could only change the tokens a
round, never a reply (the target's verification decides every emitted token).
"""

import torch

from vllm.triton_utils import tl, triton

from .markov_cache import MarkovBiasCache, markov_bias_rows

CHUNK = 2048            # columns a partial program reduces (64,640-column shard: 32 programs a row)


@triton.jit
def _first_kernel(ids_ptr, anchor_ptr, w1_ptr, w1_stride, prev_ptr, emb_ptr, emb_stride, R, RP: tl.constexpr):
    """Step 0: the anchor token (input id at the request's first query slot) and its Markov embedding."""
    b = tl.program_id(0)
    tok = tl.load(ids_ptr + tl.load(anchor_ptr + b)).to(tl.int64)
    tl.store(prev_ptr + b, tok)
    r = tl.arange(0, RP)
    rm = r < R
    tl.store(emb_ptr + b * emb_stride + r, tl.load(w1_ptr + tok * w1_stride + r, mask=rm), mask=rm)


@triton.jit
def _partial_kernel(
    base_ptr,
    base_stride,
    bias_ptr,
    bias_stride,
    part_ptr,
    part_stride,
    V,
    lo,
    ROUND: tl.constexpr,      # 1: round the sum to bf16, 2: to fp16, 0: keep fp32 (torch's result dtype of the add)
    CHUNK_N: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    cols = c * CHUNK_N + tl.arange(0, CHUNK_N)
    m = cols < V
    row = b.to(tl.int64)
    x = tl.load(base_ptr + row * base_stride + cols, mask=m, other=0.0).to(tl.float32)
    y = tl.load(bias_ptr + row * bias_stride + cols, mask=m, other=0.0).to(tl.float32)
    s = x + y
    if ROUND == 1:
        s = s.to(tl.bfloat16).to(tl.float32)
    elif ROUND == 2:
        s = s.to(tl.float16).to(tl.float32)
    s = tl.where(s == 0.0, 0.0, s)                       # -0.0 -> +0.0: torch.max sees them equal
    bits = tl.cast(s, tl.int32, bitcast=True)
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)             # monotone in the float value (signed int order)
    key = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 2147483647, key)    # NaN: above everything
    key = tl.where(m, key, -2147483648)
    kmax = tl.max(key, axis=0)
    first = tl.min(tl.where(key == kmax, cols, 2147483647), axis=0)
    tl.store(part_ptr + b * part_stride + 2 * c, kmax)
    tl.store(part_ptr + b * part_stride + 2 * c + 1, first + lo)


@triton.jit
def _pick_kernel(
    g_ptr,
    g_stride,
    N,                          # tp ranks x chunks
    draft_ptr,
    draft_stride,
    col,
    prev_ptr,
    w1_ptr,
    w1_stride,
    emb_ptr,
    emb_stride,
    R,
    WRITE_EMB: tl.constexpr,
    NP: tl.constexpr,
    RP: tl.constexpr,
):
    b = tl.program_id(0)
    p = tl.arange(0, NP)
    pm = p < N
    key = tl.load(g_ptr + b * g_stride + 2 * p, mask=pm, other=-2147483648)
    kmax = tl.max(key, axis=0)
    pos = tl.min(tl.where((key == kmax) & pm, p, NP), axis=0)
    tok = tl.load(g_ptr + b * g_stride + 2 * pos + 1).to(tl.int64)
    tl.store(draft_ptr + b * draft_stride + col, tok)
    tl.store(prev_ptr + b, tok)
    if WRITE_EMB:
        r = tl.arange(0, RP)
        rm = r < R
        tl.store(emb_ptr + b * emb_stride + r, tl.load(w1_ptr + tok * w1_stride + r, mask=rm), mask=rm)


def round_mode(base_dtype: torch.dtype, bias_dtype: torch.dtype) -> int | None:
    """The _partial_kernel ROUND for torch's add of these dtypes, None when unsupported."""
    out = torch.promote_types(base_dtype, bias_dtype)
    return {torch.bfloat16: 1, torch.float16: 2, torch.float32: 0}.get(out)


class FusedMarkovLoop:
    """Buffers and launches of the fused greedy Markov loop for one rank's vocabulary shard [lo, lo + vocab)."""

    def __init__(
        self,
        cache: MarkovBiasCache,
        w1: torch.Tensor,
        lo: int,
        vocab: int,
        tp: int,
        max_rows: int,
        num_steps: int,
        base_dtype: torch.dtype,
    ):
        mode = round_mode(base_dtype, cache.out_dtype)
        if mode is None:
            raise ValueError(f"fused Markov loop: base {base_dtype} + bias {cache.out_dtype} not supported")
        if w1.dim() != 2 or w1.stride(1) != 1:
            raise ValueError("fused Markov loop: markov_w1 must be a row-major [vocab, rank] matrix")
        self.cache = cache
        self.w1 = w1
        self.lo = int(lo)
        self.vocab = int(vocab)
        self.tp = int(tp)
        self.round = mode
        self.num_steps = int(num_steps)
        self.chunks = triton.cdiv(self.vocab, CHUNK)
        self.rank = int(w1.shape[1])
        dev = w1.device
        self.prev = torch.zeros(max_rows, dtype=torch.int64, device=dev)
        # [req, position, rank]: position i holds the embedding of the token before draft i (anchor, d0, d1, ...),
        # the layout torch.stack(confidence_markov_embeds, dim=1) gives the confidence head
        self.emb = torch.zeros(max_rows, self.num_steps, self.rank, dtype=w1.dtype, device=dev)
        self.bias = torch.empty(max_rows, cache.w2.shape[0], dtype=cache.out_dtype, device=dev)
        self.part = torch.zeros(max_rows, 2 * self.chunks, dtype=torch.int32, device=dev)
        self._np = triton.next_power_of_2(self.tp * self.chunks)
        self._rp = triton.next_power_of_2(self.rank)

    def run(
        self,
        base_logits: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_idx: torch.Tensor,
        draft_tokens: torch.Tensor,
        steps: int,
        gather,
    ) -> torch.Tensor:
        """Greedy drafts 0 .. steps-1 into draft_tokens[:n, :steps] (base_logits: [n, num_steps, >= vocab], this
        rank's shard); returns the Markov embeddings [n, num_steps, rank] (positions >= steps hold stale rows)."""
        n = base_logits.shape[0]
        assert base_logits.shape[1] == self.num_steps and base_logits.stride(2) == 1
        assert 1 <= steps <= self.num_steps and draft_tokens.stride(1) == 1
        if n == 0:
            return self.emb[:0]
        c = self.cache
        _first_kernel[(n,)](
            input_ids, anchor_idx, self.w1, self.w1.stride(0), self.prev, self.emb, self.emb.stride(0),
            self.rank, RP=self._rp,
        )
        for i in range(steps):
            markov_bias_rows(self.prev[:n], c.slot_of_token, c.cache, self.emb[:n, i], c.w2, self.bias[:n])
            base_i = base_logits[:, i]
            _partial_kernel[(n, self.chunks)](
                base_i, base_i.stride(0), self.bias, self.bias.stride(0), self.part, self.part.stride(0),
                self.vocab, self.lo, ROUND=self.round, CHUNK_N=CHUNK, num_warps=4,
            )
            g = gather(self.part[:n])                           # [n, tp * 2 * chunks], rank-major
            nxt = i + 1 < steps
            emb_next = self.emb[:, i + 1] if nxt else self.emb[:, 0]
            _pick_kernel[(n,)](
                g, g.stride(0), self.tp * self.chunks, draft_tokens, draft_tokens.stride(0), i, self.prev,
                self.w1, self.w1.stride(0), emb_next, self.emb.stride(0), self.rank,
                WRITE_EMB=nxt, NP=self._np, RP=self._rp, num_warps=1,
            )
        return self.emb[:n]
