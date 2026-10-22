# SPDX-License-Identifier: Apache-2.0
"""dsv41 patch 0078 (Loop C, 22.10.2026): Engram reads of a prompt chunk off the prefill's critical path.

Background (engram.py, engram_disk.cpp): each prefill step hashes its tokens at the start of the forward and queues the
Engram lookups of layers 1 and 14 on one prefetch stream; with disk offload (spin mode) a lookup of more than 512 rows
goes to the reader's 256-thread buffered pread pool, 2 preads a row (weight 256 B + scale 8 B), no deduplication. A
4,096-token chunk is 49,152 rows a rank and layer. Layer 1 waits for its rows after layer 0, so only embedding, hash and
layer 0 hide the read; cold rows (new text) come from the NVMe, warm ones from the page cache (the Engram cache here).
The pool threads inherit EXL3_CPU_PIN (the X925 cores) and compete with the launch thread while they read.

Two switches, both default off (= v23 bit for bit), both leave every lookup and every output unchanged:

VLLM_DSV41_ENGRAM_POOL_CPUS=0-4,10-13
    The reader's pread pool threads, created in EngramDiskTable's constructor, start on these CPUs (the creating thread
    takes the affinity for the duration of the constructor and gets its own back afterwards). The spin proxy thread,
    created later by enable_spin(), and every other thread keep EXL3_CPU_PIN. GB10: 5-9 and 15-19 are Cortex-X925,
    0-4 and 10-14 Cortex-A725; 14 carries the RDMA ring thread (EXL3_RDMA_CPU=14), so 0-4,10-13.

VLLM_DSV41_ENGRAM_LOOKAHEAD=<tokens>  (0 = off; 4096 = one full prefill chunk)
    After a prefill step's own Engram lookups have read their rows (a CUDA event on the Engram prefetch stream), a
    background thread hashes the next <tokens> prompt positions of every request that is still prefilling (numpy, the
    same arithmetic as engram.py's _hash_ids_kernel; the token history is the runner's host-visible all_token_ids) and
    reads those rows once through a second reader of the same table (own EngramDiskTable instance: own pread pool, no
    hot-row cache, no spin mode) into a throwaway buffer. The rows then sit in the page cache when the next step's real
    lookup asks for them. The real lookup path is untouched; a wrong guess (another chunk size, a preemption) costs only
    reads. Ranges the real path has read meanwhile are skipped; duplicate rows of a range are read once.
    VLLM_DSV41_ENGRAM_LOOKAHEAD_THREADS=64     pread threads of each lookahead reader (two tables a rank)
    VLLM_DSV41_ENGRAM_LOOKAHEAD_CPUS=...       CPUs of those threads and of the background thread
                                               (default VLLM_DSV41_ENGRAM_POOL_CPUS, else 0-4,10-13)
    VLLM_DSV41_ENGRAM_LOOKAHEAD_BATCH=8192     rows a read_rows call (scratch 8192 x 264 B = 2.1 MB a rank)
    VLLM_DSV41_ENGRAM_LOOKAHEAD_VERIFY=1       compare the host hash of each processed chunk with the row ids the real
                                               lookup used (the pinned staging ids); counts land in the log line
    VLLM_DSV41_ENGRAM_LOOKAHEAD_LOG_S=60       period of the INFO statistics line

Page cache: one 4,096-token window is ~98K rows a rank (2 tables), up to ~0.8 GB of cached pages - the pages the next
step reads anyway, one step earlier. MemAvailable counts page cache as available.
"""

from __future__ import annotations

import collections
import contextlib
import os
import threading
import time

import numpy as np

try:  # vLLM's logger in the serve; plain logging where vLLM is absent (CPU tests)
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


TOKENS = max(0, _int_env("VLLM_DSV41_ENGRAM_LOOKAHEAD", 0))
ENABLED = TOKENS > 0
THREADS = max(1, _int_env("VLLM_DSV41_ENGRAM_LOOKAHEAD_THREADS", 64))
BATCH_ROWS = max(256, _int_env("VLLM_DSV41_ENGRAM_LOOKAHEAD_BATCH", 8192))
VERIFY = os.environ.get("VLLM_DSV41_ENGRAM_LOOKAHEAD_VERIFY", "0") == "1"
LOG_S = float(os.environ.get("VLLM_DSV41_ENGRAM_LOOKAHEAD_LOG_S", "60") or 60)
POOL_CPUS = os.environ.get("VLLM_DSV41_ENGRAM_POOL_CPUS", "").strip()
READER_CPUS = os.environ.get("VLLM_DSV41_ENGRAM_LOOKAHEAD_CPUS", "").strip() or POOL_CPUS or "0-4,10-13"
# mm_preprocess.IMAGE_SENTINEL_BASE_ID: image-span tokens are dead in the n-gram hash (engram.py, model.py)
IMAGE_SENTINEL_ID = 129264
_SUB_POSITIONS = 1024  # positions per hash / skip decision inside a job
_MAX_JOBS = 64


def parse_cpus(spec: str) -> set[int]:
    """'0-4,10-13' -> {0..4, 10..13}, limited to the CPUs this machine has; malformed parts are ignored."""
    cpus: set[int] = set()
    n = os.cpu_count() or 0
    for part in (spec or "").split(","):
        a, _, b = part.strip().partition("-")
        try:
            lo, hi = int(a), int(b or a)
        except ValueError:
            continue
        cpus.update(c for c in range(lo, hi + 1) if 0 <= c < n)
    return cpus


@contextlib.contextmanager
def thread_affinity(spec: str):
    """The calling thread runs on `spec` inside the block (threads it starts there inherit it), then gets its old
    affinity back. Empty spec, no sched_setaffinity or no usable CPU: nothing is changed."""
    cpus = parse_cpus(spec) if spec else set()
    if not cpus or not hasattr(os, "sched_setaffinity"):
        yield
        return
    old = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, cpus)  # 0 = the calling thread on Linux
    except OSError as exc:
        logger.warning("dsv41 0078: sched_setaffinity(%s) failed (%s); affinity unchanged", spec, exc)
        yield
        return
    try:
        yield
    finally:
        os.sched_setaffinity(0, old)


class HashTables:
    """Host copies of NgramHashState's hash parameters (built once at model init when the lookahead is on)."""

    def __init__(self, token_map, multipliers, primes, offsets, pad_id: int, dead_token_id: int = IMAGE_SENTINEL_ID):
        # own copies: never a view on the model's buffers
        self.token_map = np.array(token_map, dtype=np.int64)  # [vocab]
        self.multipliers = np.array(multipliers, dtype=np.int64)  # [layers, max_ngram]
        self.primes = np.array(primes, dtype=np.int64)  # [layers, max_ngram - 1, heads]
        self.offsets = np.array(offsets, dtype=np.int64)  # [layers, (max_ngram - 1) * heads]
        self.pad_id = int(pad_id)
        self.dead_token_id = int(dead_token_id)
        self.num_layers, self.max_ngram = self.multipliers.shape
        self.num_heads = self.primes.shape[-1]

    @classmethod
    def build(cls, token_map, layout, multipliers, pad_id: int) -> "HashTables":
        """From NgramHashState's inputs: the token map (list), the EngramLayout and the multiplier tensor."""
        primes = np.asarray(layout.primes, dtype=np.int64)
        flat = primes.reshape(primes.shape[0], -1)
        # EngramLayout.offsets: per layer np.cumsum([0, *sizes[:-1]]) over the flattened (n-gram, head) primes
        offsets = np.concatenate(
            [np.zeros((flat.shape[0], 1), np.int64), np.cumsum(flat, axis=1)[:, :-1]], axis=1
        )
        mult = multipliers.detach().to("cpu").numpy() if hasattr(multipliers, "detach") else multipliers
        return cls(token_map, mult, primes, offsets, pad_id)


def segment_start(tables: HashTables, start: int) -> int:
    """First history position the hashes of positions >= start read."""
    return max(0, start - (tables.max_ngram - 1))


def host_hashes(tables: HashTables, seg: np.ndarray, start: int, end: int) -> np.ndarray:
    """Hash ids [end - start, layers, (max_ngram - 1) * heads] (int64) of positions start..end-1 of one request.

    `seg` holds the request's token ids of positions segment_start(start) .. end-1. Same arithmetic as engram.py's
    _hash_ids_kernel with the V2 runner's inputs (no slot cache; the chunk and the lookback window both come from the
    token history): position p hashes ids p, p-1, .., p-max_ngram+1; a negative position or an image-span token blocks
    that n-gram and every longer one (pad id); n-gram n of head h is
    (xor over s < n of compressed_id(p - s) * multiplier[layer, s]) mod prime[layer, n-2, h] + offset[layer, n-2, h].
    """
    t = tables
    m, heads = t.max_ngram, t.num_heads
    n = end - start
    lo = segment_start(t, start)
    seg = np.asarray(seg, dtype=np.int64)
    assert seg.shape == (end - lo,), "token segment does not cover the hashed positions"
    comp = t.token_map[seg]
    dead = seg == t.dead_token_id
    pos = np.arange(start, end, dtype=np.int64)
    out = np.empty((n, t.num_layers, (m - 1) * heads), dtype=np.int64)
    for layer in range(t.num_layers):
        rolling = np.zeros(n, dtype=np.int64)
        blocked = np.zeros(n, dtype=bool)
        for shift in range(m):
            q = pos - shift
            inside = q >= 0
            idx = np.where(inside, q - lo, 0)
            blocked |= ~inside | dead[idx]
            value = np.where(blocked, t.pad_id, comp[idx])
            rolling ^= value * t.multipliers[layer, shift]
            if shift > 0:
                cols = slice((shift - 1) * heads, shift * heads)
                out[:, layer, cols] = rolling[:, None] % t.primes[layer, shift - 1][None, :] + t.offsets[layer, cols][None, :]
    return out


class _Table:
    """One Engram table of this rank: where its rows sit in the hash ids and the readers that serve them."""

    def __init__(self, engram, embed):
        self.layer_hash_index = int(engram.layer_hash_index)
        self.h0 = int(embed.head_start)
        self.width = int(embed.part_n_hash_cols)
        self.h1 = min(self.h0 + self.width, int(embed.n_hash_cols))
        self.vocab_start = int(embed.vocab_start_idx)
        self.num_rows = int(embed.part_num_embeddings)
        self.row_bytes = int(embed.dim)
        self.scale_bytes = int(embed.dim) // int(embed.block_size)
        self.reader = embed._disk_la
        ids = getattr(embed, "_disk_ids", None)
        self.staged_ids = ids.numpy() if ids is not None else None  # pinned staging of the real lookup

    def local_rows(self, hashes: np.ndarray) -> np.ndarray:
        """[tokens, width] local row ids as the real lookup stages them (padded heads -1)."""
        local = hashes[:, self.layer_hash_index, self.h0:self.h1] - self.vocab_start
        if local.shape[1] < self.width:
            local = np.pad(local, ((0, 0), (0, self.width - local.shape[1])), constant_values=-1)
        return local


class _Job:
    __slots__ = ("event", "tokens", "work", "chunks", "ends", "marked")

    def __init__(self, event, tokens, work, chunks, ends):
        self.event = event
        self.tokens = tokens  # [max_num_reqs, max_model_len] int32 host view of the token history
        self.work = work  # [(req_id, req_index, lo, hi)] positions to prefetch
        self.chunks = chunks  # [(req_index, start, end, batch_row)] chunks of this step (verification)
        self.ends = ends  # {req_id: chunk end} read by the real path once `event` completes
        self.marked = False


class Lookahead:
    def __init__(self, model):
        import torch

        self._torch = torch
        self.tables: list[_Table] = []
        self.hash_tables: HashTables | None = None
        self.stream = None
        for mod in model.modules():
            host = getattr(mod, "_dsv41_host", None)
            if isinstance(host, HashTables):
                self.hash_tables = host
            embed = getattr(mod, "embed_tokens", None)
            if hasattr(mod, "layer_hash_index") and getattr(embed, "_disk_la", None) is not None:
                if int(getattr(embed, "dp_size", 1)) != 1:
                    raise RuntimeError("Engram tables shared over DP ranks are not supported by the lookahead")
                self.tables.append(_Table(mod, embed))
                self.stream = self.stream or getattr(mod, "_prefetch_stream", None)
        if not self.tables or self.hash_tables is None or self.stream is None:
            raise RuntimeError(
                f"no disk-offloaded Engram table with a lookahead reader ({len(self.tables)} tables, hash tables "
                f"{self.hash_tables is not None}, prefetch stream {self.stream is not None})"
            )
        self._ids = np.empty(BATCH_ROWS, dtype=np.int64)
        row_bytes = max(t.row_bytes for t in self.tables)
        scale_bytes = max(t.scale_bytes for t in self.tables)
        self._w = np.empty((BATCH_ROWS, row_bytes), dtype=np.uint8)
        self._s = np.empty((BATCH_ROWS, scale_bytes), dtype=np.uint8)
        self._done: dict[str, int] = {}  # req_id -> positions below this are prefetched (or being prefetched)
        self._ends: dict[str, int] = {}  # req_id -> end of its chunk in the last prefill step
        self._read_upto: dict[str, int] = {}  # req_id -> the real path has read the chunks below this
        self._jobs: collections.deque[_Job] = collections.deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.stats = collections.Counter()
        self._max_job_s = 0.0
        self._last_log = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="dsv41-engram-lookahead", daemon=True)
        self._thread.start()
        logger.info(
            "dsv41 0078: Engram lookahead %d tokens, %d tables, %d reader threads each on CPUs %s, batch %d rows%s",
            TOKENS, len(self.tables), THREADS, READER_CPUS, BATCH_ROWS, ", verify on" if VERIFY else "",
        )

    # ---- main thread: after the forward of an eager prefill step is enqueued ----------------------------------------
    def after_forward(self, input_batch, req_states) -> None:
        torch = self._torch
        if torch.cuda.is_current_stream_capturing():
            return
        buf = getattr(getattr(req_states, "all_token_ids", None), "_uva_buf", None)
        if buf is None:
            raise RuntimeError("the token history is not host-visible (no UVA buffer)")
        work, chunks, ends, new_done = [], [], {}, {}
        prefilling = input_batch.is_prefilling_np
        for i in range(input_batch.num_reqs):
            if not prefilling[i]:
                continue
            req_id = input_batch.req_ids[i]
            req_index = int(input_batch.idx_mapping_np[i])
            start = int(input_batch.num_computed_prefill_tokens_np[i])
            prefill_len = int(input_batch.prefill_len_np[i])
            end = min(start + int(input_batch.num_scheduled_tokens[i]), prefill_len)
            ends[req_id] = end
            if VERIFY and end > start:
                chunks.append((req_index, start, end, int(input_batch.query_start_loc_np[i])))
            if end >= prefill_len:
                continue
            done = self._done.get(req_id, 0)
            if start < self._ends.get(req_id, 0):  # restarted below an earlier chunk (preemption): forget the window
                done = 0
            lo, hi = max(end, done), min(prefill_len, end + TOKENS)
            new_done[req_id] = max(hi, done)
            if hi > lo:
                work.append((req_id, req_index, lo, hi))
        self._done = new_done
        self._ends = ends
        if not work and not chunks:
            return
        event = torch.cuda.Event()
        event.record(self.stream)  # after this step's Engram lookups (they run on this stream)
        with self._cv:
            if len(self._jobs) >= _MAX_JOBS:
                self._jobs.popleft()
                self.stats["dropped_jobs"] += 1
            self._jobs.append(_Job(event, buf.np, work, chunks, ends))
            self._cv.notify()
        self.stats["steps"] += 1

    # ---- background thread ------------------------------------------------------------------------------------------
    def _run(self) -> None:
        cpus = parse_cpus(READER_CPUS)
        if cpus and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, cpus)
            except OSError:
                pass
        while True:
            with self._cv:
                while not self._jobs:
                    self._cv.wait(timeout=LOG_S)
                    self._maybe_log()
                job = self._jobs.popleft()
            try:
                if not self._wait(job.event):
                    self.stats["timeouts"] += 1
                    continue
                self._mark(job)
                if job.chunks:
                    self._verify(job)
                t0 = time.monotonic()
                self._prefetch(job)
                dt = time.monotonic() - t0
                self.stats["jobs"] += 1
                self.stats["read_us"] += int(dt * 1e6)
                self._max_job_s = max(self._max_job_s, dt)
            except Exception as exc:  # noqa: BLE001  (a prefetch must never take the serve down)
                self.stats["errors"] += 1
                if self.stats["errors"] <= 3:
                    logger.warning("dsv41 0078: lookahead job failed: %r", exc)
            self._maybe_log()

    def _wait(self, event, timeout_s: float = 120.0) -> bool:
        t_end = time.monotonic() + timeout_s
        while not event.query():
            if time.monotonic() > t_end:
                return False
            time.sleep(0.0005)
        return True

    def _mark(self, job: _Job) -> None:
        if job.marked:
            return
        job.marked = True
        for req_id, end in job.ends.items():
            if end > self._read_upto.get(req_id, 0):
                self._read_upto[req_id] = end
        if len(self._read_upto) > 4 * _MAX_JOBS:  # forget requests that are long gone
            live = set(self._done) | set(job.ends)
            self._read_upto = {k: v for k, v in self._read_upto.items() if k in live}

    def _mark_finished_newer(self) -> None:
        """Steps queued after the current job whose lookups are done: their chunks need no prefetch any more."""
        with self._lock:
            newer = list(self._jobs)
        for job in newer:
            if job.marked:
                continue
            if not job.event.query():
                break
            self._mark(job)

    def _segment(self, job: _Job, req_index: int, start: int, end: int) -> np.ndarray | None:
        """A copy of the token ids the hashes of start..end-1 read, or None if any id is outside the token map."""
        seg = np.array(job.tokens[req_index, segment_start(self.hash_tables, start):end], dtype=np.int64)
        if seg.size and (seg.min() < 0 or seg.max() >= self.hash_tables.token_map.shape[0]):
            self.stats["bad_tokens"] += 1
            return None
        return seg

    def _prefetch(self, job: _Job) -> None:
        ht = self.hash_tables
        for req_id, req_index, lo, hi in job.work:
            seen = [np.zeros(0, dtype=np.int64) for _ in self.tables]  # rows already read for this range
            for a in range(lo, hi, _SUB_POSITIONS):
                b = min(hi, a + _SUB_POSITIONS)
                self._mark_finished_newer()
                if b <= self._read_upto.get(req_id, 0):
                    self.stats["skipped_ranges"] += 1
                    continue
                seg = self._segment(job, req_index, a, b)
                if seg is None:
                    continue
                hashes = host_hashes(ht, seg, a, b)
                for k, table in enumerate(self.tables):
                    rows = table.local_rows(hashes).reshape(-1)
                    self.stats["rows_asked"] += rows.size
                    rows = np.unique(rows[(rows >= 0) & (rows < table.num_rows)])
                    rows = np.setdiff1d(rows, seen[k], assume_unique=True)
                    seen[k] = np.union1d(seen[k], rows)
                    self._read(table, rows)

    def _read(self, table: _Table, rows: np.ndarray) -> None:
        for i in range(0, rows.size, BATCH_ROWS):
            n = min(BATCH_ROWS, rows.size - i)
            self._ids[:n] = rows[i:i + n]
            table.reader.read_rows(self._ids.ctypes.data, self._w.ctypes.data, self._s.ctypes.data, n)
            self.stats["rows_read"] += n
            self.stats["batches"] += 1

    def _verify(self, job: _Job) -> None:
        ht = self.hash_tables
        for req_index, start, end, row in job.chunks:
            seg = self._segment(job, req_index, start, end)
            if seg is None:
                continue
            hashes = host_hashes(ht, seg, start, end)
            for table in self.tables:
                if table.staged_ids is None:
                    continue
                want = table.local_rows(hashes).reshape(-1)
                got = table.staged_ids[row * table.width:(row + end - start) * table.width]
                self.stats["verify_rows"] += want.size
                self.stats["verify_bad"] += int(np.count_nonzero(want != got))

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < LOG_S or not self.stats:
            return
        s, self.stats = self.stats, collections.Counter()
        mx, self._max_job_s = self._max_job_s, 0.0
        self._last_log = now
        logger.info(
            "dsv41 0078 lookahead: %d steps, %d jobs, %d rows read (%d asked), %.2f s reading (max %.0f ms a job), "
            "%d ranges skipped (already read), %d timeouts, %d dropped, %d errors%s",
            s["steps"], s["jobs"], s["rows_read"], s["rows_asked"], s["read_us"] / 1e6, mx * 1e3,
            s["skipped_ranges"], s["timeouts"], s["dropped_jobs"], s["errors"],
            f", verify {s['verify_bad']} of {s['verify_rows']} row ids differ" if s["verify_rows"] else "",
        )


_DRIVER: Lookahead | None = None
_FAILED = False


def after_forward(model, input_batch, req_states) -> None:
    """Runner hook (eager prefill steps, after the forward is enqueued). Never raises."""
    global _DRIVER, _FAILED
    if _FAILED:
        return
    try:
        if _DRIVER is None:
            _DRIVER = Lookahead(model)
        _DRIVER.after_forward(input_batch, req_states)
    except Exception as exc:  # noqa: BLE001
        _FAILED = True
        logger.warning("dsv41 0078: Engram lookahead switched off: %s", exc)
