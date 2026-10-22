# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    derive_canonical_mappings,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)

logger = init_logger(__name__)

# dsv41 patch 0077: KV offload with TP across nodes. With the one-copy layout
# (config.py), only rank 0 stores, and the scheduler's tiers fill rank 0's region.
# Every other rank has its own region on its own node, which nothing fills, so
# before a load each chunk is copied from rank 0's region into the others' over a
# gloo group (fabric NIC via GLOO_SOCKET_IFNAME); then every rank loads from its
# own region as before. _VERIFY: the other ranks store too (old per-node copies),
# for comparing the regions across nodes before any load.
_DSV41_OFFLOAD_MN = os.environ.get("VLLM_DSV41_OFFLOAD_MULTINODE", "0") == "1"
_DSV41_OFFLOAD_MN_VERIFY = os.environ.get("VLLM_DSV41_OFFLOAD_MN_VERIFY", "0") == "1"
_DSV41_OFFLOAD_MN_STAGE_MB = int(os.environ.get("VLLM_DSV41_OFFLOAD_MN_STAGE_MB", "32"))
_DSV41_OFFLOAD_MN_RUN_MIN = 16  # runs of >= 16 consecutive chunks go without staging


class OffloadingConnectorWorker:
    """Implementation of Worker side methods."""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig,
    ):
        self.spec = spec
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.worker: OffloadingWorker | None = None
        # Non-writers still ack: pending_count waits for world_size per job.
        self._is_store_writer = (
            not self.spec.replicated_layout
            or self.spec.config.canonical_layout
            or self.spec.config.parallel.rank == 0
            or (_DSV41_OFFLOAD_MN and _DSV41_OFFLOAD_MN_VERIFY)  # patch 0077
        )
        # patch 0077: rank-0 -> other ranks chunk copy before loads (see _mn_init)
        self._mn_group = None
        self._mn_rows: torch.Tensor | None = None
        self._mn_stage: torch.Tensor | None = None
        self._mn_is_src = False
        self._mn_stat = [0, 0, 0.0, time.monotonic()]  # chunks, bytes, seconds, last log

        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)
        self._mn_init()

    def _mn_init(self) -> None:
        """patch 0077: set up the chunk copy from rank 0 for TP across nodes.

        Called on every rank from register_kv_caches, so new_group() runs
        collectively, in the same order on all ranks."""
        if not (_DSV41_OFFLOAD_MN and self.spec.replicated_layout):
            return
        if self.spec.config.canonical_layout:
            return
        dist = torch.distributed
        if not dist.is_initialized() or dist.get_world_size() < 2:
            return
        region = getattr(self.worker, "_mmap_region", None)
        if region is None:
            raise RuntimeError(
                "dsv41 patch 0077: KV offload across nodes needs the shared CPU region"
            )
        row_stride = region.total_size_bytes // region.num_chunks
        self._mn_rows = region.base_tensor.view(region.num_chunks, row_stride)
        self._mn_group = dist.new_group(backend="gloo")
        self._mn_is_src = dist.get_rank() == 0
        stage_rows = max(1, (_DSV41_OFFLOAD_MN_STAGE_MB << 20) // row_stride)
        self._mn_stage = torch.empty((stage_rows, row_stride), dtype=torch.int8)
        logger.info(
            "dsv41 patch 0077: rank %d %s CPU chunks %s rank 0 before loads "
            "(%d chunks x %d B, staging %d rows, rank-%d stores %s)",
            dist.get_rank(),
            "sends" if self._mn_is_src else "receives",
            "to the other ranks from" if self._mn_is_src else "from",
            region.num_chunks,
            row_stride,
            stage_rows,
            dist.get_rank(),
            "on" if self._is_store_writer else "off",
        )

    def _mn_sync(self, src_spec: LoadStoreSpec) -> None:
        """patch 0077: copy the load's CPU chunks from rank 0 into every rank's
        region. Same chunk list and order on every rank (same metadata), so the
        broadcasts pair up; rank 0's chunks are complete here (stores acked by
        all ranks, tier reads finished before the scheduler issues the load)."""
        rows = self._mn_rows
        stage = self._mn_stage
        assert rows is not None and stage is not None
        ids = np.asarray(getattr(src_spec, "block_ids"), dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return
        t0 = time.perf_counter()
        dist = torch.distributed
        cap = stage.shape[0]
        pending: list[np.ndarray] = []
        npending = 0

        def flush() -> None:
            nonlocal pending, npending
            if not npending:
                return
            idx = torch.from_numpy(np.concatenate(pending))
            buf = stage[:npending]
            if self._mn_is_src:
                torch.index_select(rows, 0, idx, out=buf)
            dist.broadcast(buf, src=0, group=self._mn_group)
            if not self._mn_is_src:
                rows.index_copy_(0, idx, buf)
            pending, npending = [], 0

        for run in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1):
            if run.size >= _DSV41_OFFLOAD_MN_RUN_MIN:
                flush()
                dist.broadcast(
                    rows[int(run[0]) : int(run[-1]) + 1], src=0, group=self._mn_group
                )
                continue
            off = 0
            while off < run.size:
                take = min(cap - npending, run.size - off)
                pending.append(run[off : off + take])
                npending += take
                off += take
                if npending == cap:
                    flush()
        flush()
        st = self._mn_stat
        st[0] += int(ids.size)
        st[1] += int(ids.size) * rows.shape[1]
        st[2] += time.perf_counter() - t0
        now = time.monotonic()
        if now - st[3] >= 60.0:
            logger.info(
                "dsv41 patch 0077: copied %d chunks (%.1f MiB) from rank 0 in "
                "%.1f ms since the last report",
                st[0],
                st[1] / 2**20,
                st[2] * 1e3,
            )
            self._mn_stat = [0, 0, 0.0, now]

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        kv_cache_config = self.kv_cache_config
        selected_group_ids = get_offloading_group_ids(kv_cache_config)
        selected_groups = tuple(
            kv_cache_config.kv_cache_groups[group_id] for group_id in selected_group_ids
        )
        mappings = derive_canonical_mappings(
            self.vllm_config, kv_cache_config, kv_caches
        )

        # layer_name -> (num_blocks, page_size_bytes) tensor
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        for kv_cache_group in selected_groups:
            assert not kv_cache_group.host_resident
            num_blocks = kv_cache_config.num_blocks
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group_layer_names:
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                ref = group_kernel_blocks(kv_caches[layer_name], num_blocks)
                page = layer_kv_cache_spec.page_size_bytes
                elem_size = ref.element_size()
                byte_offset = ref.storage_offset() * elem_size
                # Packed layouts (e.g. DSv4) interleave layers per block, so the
                # attention tensor's stride(0) (the manager-block stride) exceeds
                # page_size_bytes; other layouts have stride(0) == page_size_bytes.
                block_stride_bytes = (
                    ref.stride(0) * elem_size
                    if isinstance(layer_kv_cache_spec, AttentionSpec)
                    else page
                )
                tensors_per_block[layer_name] = (
                    torch.tensor([], dtype=torch.int8, device=ref.device).set_(
                        ref.untyped_storage(),
                        byte_offset,
                        (num_blocks, page),
                        (block_stride_bytes, 1),
                    ),
                )
                page_size_bytes[layer_name] = page

                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    unpadded_page_size_bytes[layer_name] = (
                        layer_kv_cache_spec.unpadded_page_size_bytes
                    )
                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes
                else:
                    raise NotImplementedError

        # Packed layouts (e.g. DSv4) interleave all layers within each manager
        # block: a layer view's block stride exceeds its page size. Offload the
        # whole packed block as a single transfer region.
        packed_layer_name = next(
            (
                layer_name
                for layer_name, (tensor,) in tensors_per_block.items()
                if tensor.stride(0) != tensor.shape[1]
            ),
            None,
        )
        if (
            packed_layer_name is not None
            and kv_cache_config.hisparse_host_num_blocks is None
        ):
            (tensor,) = tensors_per_block[packed_layer_name]
            num_blocks = tensor.shape[0]
            block_stride = tensor.stride(0)
            packed_tensor = tensor.as_strided(
                (num_blocks, block_stride),
                (block_stride, 1),
                storage_offset=0,
            )
            self._init_worker(
                CanonicalKVCaches(
                    [CanonicalKVCacheTensor(packed_tensor, block_stride)],
                    [[CanonicalKVCacheRef(0, block_stride)] for _ in selected_groups],
                )
            )
            return

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        # Layers that alias the same bytes (cache groups overlay each other) share
        # one transfer region: exactly the views with equal address and strides.
        aliased_layers: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
        for layer_name, layer_tensors in tensors_per_block.items():
            view = layer_tensors[0]
            aliased_layers[(view.data_ptr(), view.stride())].append(layer_name)

        for tensor_layer_names in aliased_layers.values():
            assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1

            first_layer_name = tensor_layer_names[0]
            for tensor in tensors_per_block[first_layer_name]:
                block_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=page_size_bytes[first_layer_name],
                    )
                )

                curr_tensor_idx = len(block_tensors) - 1
                for layer_name in tensor_layer_names:
                    mapping = (
                        mappings.get(layer_name)
                        if len(tensors_per_block[first_layer_name]) == 1
                        else None
                    )
                    assert (
                        mapping is None
                        or mapping.local_page_size_bytes
                        == unpadded_page_size_bytes[layer_name]
                    )
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=curr_tensor_idx,
                            page_size_bytes=(unpadded_page_size_bytes[layer_name]),
                            mapping=mapping,
                        )
                    )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in selected_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._init_worker(canonical_kv_caches)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        assert self.worker is not None

        # Pop jobs_to_flush from store_jobs into _unsubmitted_store_jobs
        # so the existing submission loop below submits them before wait().
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )

        # Submit deferred stores from previous step (and jobs_to_flush above).
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert isinstance(src_spec, GPULoadStoreSpec)
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            if self._mn_group is not None:  # patch 0077
                self._mn_sync(entry.src_spec)
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:
                # Gate before queueing: no _unsubmitted_store_jobs entry.
                self._connector_worker_meta.mark_completed(job_id)
                continue
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Returns:
        tuple of (finished_sending, finished_recving). Stores never
        emit finished_sending — the scheduler tracks store completion
        via kv_connector_worker_meta.completed_jobs and fences any
        block reuse via jobs_to_flush. Loads still emit
        finished_recving so the base scheduler can resume requests
        blocked on remote KV (and free aborted-during-load reqs).

        """
        assert self.worker is not None
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            is_load = job_id in self._load_jobs
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                if is_load:
                    stats = self._connector_worker_meta.transfer_stats.load
                else:
                    stats = self._connector_worker_meta.transfer_stats.store
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
                )

            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        return set(), finished_recving

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()
