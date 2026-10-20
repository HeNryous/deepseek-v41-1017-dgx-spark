// One-hop all-reduce of small bf16 tensors over RoCE RDMA writes, for vLLM's tensor-parallel decode on GB10 (rdma.py).
//
// A staging kernel copies this rank's tensor into a slot of registered host memory and rings a doorbell there. A CPU
// thread splits the slot over the node's RDMA devices and RDMA-writes the parts, with immediate, into every peer's
// receive slot for this rank. A peer's thread sees each write's receive completion (generated only after the data is
// placed) and, once every part of a slot is in, publishes the sequence number in a flag; the summing kernel waits for
// every peer's flag and writes sum over ranks in rank order (fp32, rounded once to bf16), so every rank gets the same
// bits. No NCCL proxy is on the path, and a CUDA graph captures the kernel pair as any other.
//
// Flag-write mode (create(..., flagw=1), EXL3_RDMA_FLAGW=1, the same on every rank): the sending thread chains each
// part's data write with an inline 8-byte write of the sequence number into the peer's flag for (slot, rank, device),
// on the same queue pair, so the peer's GPU polls the flags the NIC writes and no receiving thread is on the path. An
// RC queue pair places writes in order and the rings are registered without relaxed ordering, so a flag lands after its
// data; tests/rdma_ar_2node.py checks every sum of a long run (STRESS=N, both flag-write modes).
//
// All-gather (dsv41 patch 0045, `all_gather`, rdma.py EXL3_RDMA_AG=1): the same ring, sequence and flags, with the
// summing kernel replaced by a copying one: the slots of every rank go out in rank order as the concatenation along
// one dim (x seen as [outer, inner bytes], out as [outer, world, inner bytes]), the bytes NCCL's all-gather plus vLLM's
// movedim would give, for any dtype. All-reduces and all-gathers share the one sequence: every rank issues them in the
// same order (they are the same model code), so a slot means the same call on every rank. 16-byte copies when the
// inner bytes and both pointers allow it, else 4-byte ones (the decision is local: the wire carries bytes).
//
// Adapted from TensorFold's one-hop gather (github.com/bertholomus/TensorFold, branch deepseek-v41-tp2,
// src/tensorfold/cuda/rdma_gather.cu, v5; Copyright 2026 TensorFold contributors; Apache License 2.0, MIT for its
// parts written before TensorFold 0.6.0; see NOTICE.tensorfold): its rings, flags, doorbell, proxy, queue-pair setup and
// abort word as written there, with the collecting kernel turned into a rank-order bf16 sum, the probes and the
// non-registered-memory modes left out.

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <c10/cuda/CUDAStream.h>         // c10, not ATen/cuda: the image lacks cusparse.h
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int STAGE_THREADS = 512;
constexpr int SUM_THREADS = 256;
constexpr int RECV_DEPTH = 256;           // receive WRs kept posted on every QP (a write with immediate consumes one)

__device__ __forceinline__ uint64_t load_sys(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void store_relaxed_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

struct Nic {
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    ibv_mr* mr = nullptr;                 // the send ring's
    ibv_mr* mr_recv = nullptr;            // the receive ring's
    ibv_gid gid{};
    ibv_mtu mtu = IBV_MTU_1024;           // the port's active MTU (1024 on a 1500-byte Ethernet link, 4096 with jumbo)
    std::vector<ibv_qp*> qps;             // per peer
    std::vector<uint32_t> psns;
};

struct Peer {
    std::vector<uint32_t> qpn, psn, rkey;  // per device
    std::vector<std::string> gid;
    uint64_t recv = 0;
};

struct Ring {
    int rank = 0, world = 1, slots = 4, port = 1, gid_index = 3, cpu = -1;
    int flagw = 0;                        // flag-write mode (see the top)
    size_t max_bytes = 0, send_bytes = 0, recv_bytes = 0, meta_bytes = 0;
    size_t rflag_off = 0;                 // flag-write mode: the remote flags' offset in the receive region
    char* send_ring = nullptr;            // [slots][max_bytes]
    char* recv_ring = nullptr;            // [slots][world][max_bytes]
    char* meta = nullptr;
    uint64_t* flags = nullptr;            // [slots][world]: the sequence whose data from a peer is in (slot, peer)
    uint64_t* rflags = nullptr;           // flag-write mode: [slots][world][devices], written by the peers' NICs
    uint64_t* doorbell = nullptr;         // the last sequence staged
    uint64_t* sizes = nullptr;            // [slots]: bytes staged in a slot
    uint64_t* sent = nullptr;             // [slots]: the last sequence whose writes from a slot completed
    uint64_t* abort_word = nullptr;       // nonzero: every wait gives up (a peer is gone)
    uint64_t* seq = nullptr;              // device: all-reduces so far
    unsigned* staged = nullptr;           // device: blocks of the current staging launch done copying
    std::vector<Nic> nics;
    std::vector<Peer> peers;
    std::thread proxy;
    std::atomic<bool> running{false};
    std::atomic<int> failed{0};
    char why[256] = {0};

    ~Ring() {
        running.store(false);
        if (proxy.joinable()) proxy.join();
    }
};

std::vector<std::unique_ptr<Ring>> rings;

Ring& get(int64_t h) {
    TORCH_CHECK(h >= 0 && h < (int64_t)rings.size() && rings[h], "rdma all-reduce: bad handle");
    return *rings[h];
}

template <typename T>
T* dev(T* host_ptr) {
    T* d = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), host_ptr, 0));
    return d;
}

// Each block: wait until the slot's last writes are done, copy its share of the tensor in; the last block to finish
// (a counter) publishes the size and the doorbell. One system fence a block after a barrier (as NCCL publishes its
// proxy FIFO), the size written before that release. T: the copy unit (uint4 for the all-reduce, as before; the
// all-gather also uses uint32_t when its rows are not 16-byte multiples); n16 counts units of T.
template <typename T>
__global__ void __launch_bounds__(STAGE_THREADS) stage_kernel(const T* __restrict__ src, int n16, char* send_ring,
                                                               uint64_t* sizes, uint64_t* doorbell, const uint64_t* sent,
                                                               uint64_t* seq, int slots, uint64_t max_bytes,
                                                               const uint64_t* abort_word, unsigned* staged) {
    __shared__ uint64_t s_seq;
    __shared__ int s_last;
    if (threadIdx.x == 0) {
        const uint64_t q = *seq + 1;
        const int slot = (int)(q % slots);
        if (q > (uint64_t)slots)
            for (uint32_t i = 1; load_sys(sent + slot) + slots < q; ++i)
                if ((i & 1023) == 0 && load_sys(abort_word)) break;
        s_seq = q;
        if (blockIdx.x == 0) store_relaxed_sys(sizes + slot, (uint64_t)n16 * sizeof(T));
    }
    __syncthreads();
    const uint64_t q = s_seq;
    const int slot = (int)(q % slots);
    T* dst = reinterpret_cast<T*>(send_ring + (size_t)slot * max_bytes);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) dst[i] = src[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_last) {
        __threadfence_system();
        *staged = 0;
        *seq = q;
        store_relaxed_sys(doorbell, q);
    }
}

// Wait for every peer's flag at this sequence, then out = sum over ranks in rank order (this rank's part from `own`).
__global__ void __launch_bounds__(SUM_THREADS) sum_kernel(const uint4* __restrict__ own, uint4* __restrict__ out,
                                                           int n16, const char* recv_ring, const uint64_t* flags,
                                                           const uint64_t* seq, int rank, int world, int slots,
                                                           uint64_t max_bytes, const uint64_t* abort_word,
                                                           const uint64_t* rflags, int nd) {
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    if (rflags == nullptr) {
        if (threadIdx.x < world && threadIdx.x != rank)
            for (uint32_t i = 1; load_sys(flags + (size_t)slot * world + threadIdx.x) < q; ++i)
                if ((i & 1023) == 0 && load_sys(abort_word)) break;
    } else if (threadIdx.x < world * nd && (int)threadIdx.x / nd != rank) {    // a thread a (peer, device) flag
        const uint64_t* f = rflags + ((size_t)slot * world + threadIdx.x / nd) * nd + threadIdx.x % nd;
        for (uint32_t i = 1; load_sys(f) < q; ++i)
            if ((i & 1023) == 0 && load_sys(abort_word)) break;
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) {
        float acc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.f;
        for (int p = 0; p < world; ++p) {
            uint4 v;
            if (p == rank) {
                v = own[i];
            } else {
                const uint4* in = reinterpret_cast<const uint4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
                v = __ldcv(in + i);
            }
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(h[e]);
                acc[2 * e] += f.x;
                acc[2 * e + 1] += f.y;
            }
        }
        uint4 r;
        __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = __floats2bfloat162_rn(acc[2 * e], acc[2 * e + 1]);
        out[i] = r;
    }
}

// dsv41 patch 0045: wait for every peer's flag at this sequence (the sum's wait, both flag modes), then copy every
// rank's n units (this rank's from `own`) to out as [outer][world][inner]: rank p's unit k = (i, j) with i = k / inner
// lands at (i * world + p) * inner + j. A copy, so out holds the same bits on every rank and the same bytes as NCCL's
// all-gather followed by movedim(0, dim).
template <typename T>
__global__ void __launch_bounds__(SUM_THREADS) gather_kernel(const T* __restrict__ own, T* __restrict__ out, int n,
                                                              int inner, const char* recv_ring, const uint64_t* flags,
                                                              const uint64_t* seq, int rank, int world, int slots,
                                                              uint64_t max_bytes, const uint64_t* abort_word,
                                                              const uint64_t* rflags, int nd) {
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    if (rflags == nullptr) {
        if (threadIdx.x < world && threadIdx.x != rank)
            for (uint32_t i = 1; load_sys(flags + (size_t)slot * world + threadIdx.x) < q; ++i)
                if ((i & 1023) == 0 && load_sys(abort_word)) break;
    } else if (threadIdx.x < world * nd && (int)threadIdx.x / nd != rank) {
        const uint64_t* f = rflags + ((size_t)slot * world + threadIdx.x / nd) * nd + threadIdx.x % nd;
        for (uint32_t i = 1; load_sys(f) < q; ++i)
            if ((i & 1023) == 0 && load_sys(abort_word)) break;
    }
    __syncthreads();
    const int total = n * world;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
        const int p = e / n, k = e - p * n;
        const int i = k / inner, j = k - i * inner;
        T v;
        if (p == rank) {
            v = own[k];
        } else {
            const T* in = reinterpret_cast<const T*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
            v = __ldcv(in + k);
        }
        out[((size_t)i * world + p) * inner + j] = v;
    }
}

void fail(Ring& g, const std::string& what) {
    if (!g.failed.exchange(1)) snprintf(g.why, sizeof(g.why), "%s", what.c_str());
    __atomic_store_n(g.abort_word, (uint64_t)1, __ATOMIC_RELEASE);      // the GPU's waits give up
}

bool post_recvs(ibv_qp* qp, int count) {
    for (int i = 0; i < count; ++i) {
        ibv_recv_wr wr{};
        ibv_recv_wr* bad = nullptr;
        if (ibv_post_recv(qp, &wr, &bad)) return false;
    }
    return true;
}

void proxy_loop(Ring* gp) {
    Ring& g = *gp;
    if (g.cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(g.cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    const int nd = (int)g.nics.size();
    uint64_t posted = 0;
    std::vector<uint64_t> done_seq(g.slots, 0);
    std::vector<int> done(g.slots, 0);
    std::vector<std::vector<uint64_t>> got(g.world, std::vector<uint64_t>(nd, 0));   // per peer: last part per device
    std::vector<uint64_t> published(g.world, 0);
    std::vector<std::pair<uint32_t, int>> owner;                                       // qp_num -> peer
    for (int d = 0; d < nd; ++d)
        for (int p = 0; p < g.world; ++p)
            if (g.nics[d].qps[p]) owner.push_back({g.nics[d].qps[p]->qp_num, p});
    ibv_wc wc[64];
    const volatile uint64_t* doorbell = g.doorbell;
    while (g.running.load(std::memory_order_relaxed)) {
        const uint64_t ring = *doorbell;
        while (posted < ring) {
            const uint64_t q = posted + 1;
            const int slot = (int)(q % g.slots);
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            const uint64_t n = ((volatile uint64_t*)g.sizes)[slot];
            const uint64_t unit = ((n / 16) / nd) * 16;   // whole 16-byte units a device, the last takes the rest
            for (int d = 0; d < nd; ++d) {
                const uint64_t off = unit * d, len = d == nd - 1 ? n - off : unit;
                for (int p = 0; p < g.world; ++p) {
                    if (p == g.rank) continue;
                    const Peer& peer = g.peers[p];
                    ibv_sge sge{(uint64_t)(uintptr_t)(g.send_ring + (size_t)slot * g.max_bytes + off), (uint32_t)len,
                                g.nics[d].mr->lkey};
                    if (g.flagw) {
                        // the part's data, then (same QP, one post) the sequence into the peer's (slot, rank, d) flag
                        uint64_t qv = q;
                        ibv_sge fsge{(uint64_t)(uintptr_t)&qv, 8, 0};
                        ibv_send_wr data{}, flag{};
                        data.wr_id = q;
                        data.sg_list = &sge;
                        data.num_sge = len ? 1 : 0;
                        data.opcode = IBV_WR_RDMA_WRITE;
                        data.wr.rdma.remote_addr = peer.recv + ((uint64_t)slot * g.world + g.rank) * g.max_bytes + off;
                        data.wr.rdma.rkey = peer.rkey[d];
                        data.next = &flag;
                        flag.wr_id = q;
                        flag.sg_list = &fsge;
                        flag.num_sge = 1;
                        flag.opcode = IBV_WR_RDMA_WRITE;
                        flag.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
                        flag.wr.rdma.remote_addr =
                            peer.recv + g.rflag_off + (((uint64_t)slot * g.world + g.rank) * nd + d) * 8;
                        flag.wr.rdma.rkey = peer.rkey[d];
                        ibv_send_wr* bad = nullptr;
                        if (ibv_post_send(g.nics[d].qps[p], &data, &bad)) {
                            fail(g, "ibv_post_send failed");
                            return;
                        }
                        continue;
                    }
                    ibv_send_wr wr{};
                    wr.wr_id = q;
                    wr.sg_list = &sge;
                    wr.num_sge = len ? 1 : 0;
                    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
                    wr.send_flags = IBV_SEND_SIGNALED;
                    wr.imm_data = htonl((uint32_t)(q & 0xffffffffu));
                    wr.wr.rdma.remote_addr = peer.recv + ((uint64_t)slot * g.world + g.rank) * g.max_bytes + off;
                    wr.wr.rdma.rkey = peer.rkey[d];
                    ibv_send_wr* bad = nullptr;
                    if (ibv_post_send(g.nics[d].qps[p], &wr, &bad)) {
                        fail(g, "ibv_post_send failed");
                        return;
                    }
                }
            }
            posted = q;
        }
        for (int d = 0; d < nd; ++d) {
            const int k = ibv_poll_cq(g.nics[d].cq, 64, wc);
            if (k < 0) {
                fail(g, "ibv_poll_cq failed");
                return;
            }
            for (int i = 0; i < k; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fail(g, std::string("RDMA completion: ") + ibv_wc_status_str(wc[i].status));
                    return;
                }
                if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                    int p = -1;
                    for (auto& o : owner)
                        if (o.first == wc[i].qp_num) p = o.second;
                    if (p < 0) {
                        fail(g, "receive completion on an unknown QP");
                        return;
                    }
                    // a device delivers a peer's parts in order: the 32-bit immediate extends its last sequence
                    uint64_t q = (got[p][d] & ~0xffffffffull) | ntohl(wc[i].imm_data);
                    if (q <= got[p][d]) q += 1ull << 32;
                    got[p][d] = q;
                    // a sequence is in once every device has delivered its part (a peer may already be one ahead)
                    uint64_t in = got[p][0];
                    for (int x = 1; x < nd; ++x) in = std::min(in, got[p][x]);
                    while (published[p] < in) {
                        const uint64_t r = ++published[p];
                        __atomic_store_n(g.flags + (size_t)(r % g.slots) * g.world + p, r, __ATOMIC_RELEASE);
                    }
                    if (!post_recvs(g.nics[d].qps[p], 1)) {
                        fail(g, "ibv_post_recv failed");
                        return;
                    }
                } else {
                    const uint64_t q = wc[i].wr_id;
                    const int slot = (int)(q % g.slots);
                    if (done_seq[slot] != q) {
                        done_seq[slot] = q;
                        done[slot] = 0;
                    }
                    if (++done[slot] == (g.world - 1) * nd) __atomic_store_n(g.sent + slot, q, __ATOMIC_RELEASE);
                }
            }
        }
    }
}

char* pinned(size_t n) {
    // malloc'd pages registered with CUDA (mapped: on GB10 the device pointer is the host pointer, and the GPU reaches
    // them through its coherent L2 path, unlike cudaHostAlloc memory) and locked (best effort); the NIC's registration
    // pins the rings
    void* p = nullptr;
    TORCH_CHECK(posix_memalign(&p, 1 << 16, n) == 0 && p, "rdma all-reduce: no memory for ", n, " pinned bytes");
    std::memset(p, 0, n);
    (void)mlock(p, n);
    C10_CUDA_CHECK(cudaHostRegister(p, n, cudaHostRegisterMapped | cudaHostRegisterPortable));
    return reinterpret_cast<char*>(p);
}

Nic open_nic(const std::string& name, Ring& g) {
    int count = 0;
    ibv_device** list = ibv_get_device_list(&count);
    TORCH_CHECK(list && count > 0, "rdma all-reduce: no RDMA devices");
    ibv_device* chosen = nullptr;
    for (int i = 0; i < count; ++i)
        if (name == ibv_get_device_name(list[i])) chosen = list[i];
    TORCH_CHECK(chosen, "rdma all-reduce: no RDMA device named ", name);
    Nic nic;
    nic.ctx = ibv_open_device(chosen);
    ibv_free_device_list(list);
    TORCH_CHECK(nic.ctx, "rdma all-reduce: ibv_open_device ", name, " failed");
    nic.pd = ibv_alloc_pd(nic.ctx);
    TORCH_CHECK(nic.pd, "rdma all-reduce: ibv_alloc_pd failed");
    nic.cq = ibv_create_cq(nic.ctx, 4096, nullptr, nullptr, 0);
    TORCH_CHECK(nic.cq, "rdma all-reduce: ibv_create_cq failed");
    nic.mr = ibv_reg_mr(nic.pd, g.send_ring, g.send_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    TORCH_CHECK(nic.mr, "rdma all-reduce: ibv_reg_mr of the send ring on ", name, " failed");
    nic.mr_recv = ibv_reg_mr(nic.pd, g.recv_ring, g.recv_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    TORCH_CHECK(nic.mr_recv, "rdma all-reduce: ibv_reg_mr of the receive ring on ", name, " failed");
    TORCH_CHECK(ibv_query_gid(nic.ctx, g.port, g.gid_index, &nic.gid) == 0, "rdma all-reduce: ibv_query_gid failed");
    ibv_port_attr pa{};
    TORCH_CHECK(ibv_query_port(nic.ctx, g.port, &pa) == 0, "rdma all-reduce: ibv_query_port failed");
    nic.mtu = pa.active_mtu;   // a path MTU above the link's drops every packet (TensorFold's 4096 needs jumbo frames)
    std::random_device rd;
    nic.qps.assign(g.world, nullptr);
    nic.psns.assign(g.world, 0);
    for (int p = 0; p < g.world; ++p) {
        if (p == g.rank) continue;
        ibv_qp_init_attr attr{};
        attr.send_cq = nic.cq;
        attr.recv_cq = nic.cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = 512;
        attr.cap.max_recv_wr = RECV_DEPTH;
        attr.cap.max_send_sge = 1;
        attr.cap.max_inline_data = g.flagw ? 64 : 0;    // flag-write mode's 8-byte flag writes go inline
        attr.cap.max_recv_sge = 1;
        attr.sq_sig_all = 0;
        nic.qps[p] = ibv_create_qp(nic.pd, &attr);
        TORCH_CHECK(nic.qps[p], "rdma all-reduce: ibv_create_qp failed");
        ibv_qp_attr init{};
        init.qp_state = IBV_QPS_INIT;
        init.pkey_index = 0;
        init.port_num = g.port;
        init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        TORCH_CHECK(ibv_modify_qp(nic.qps[p], &init,
                                  IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                    "rdma all-reduce: QP to INIT failed");
        TORCH_CHECK(post_recvs(nic.qps[p], RECV_DEPTH), "rdma all-reduce: ibv_post_recv failed");
        nic.psns[p] = rd() & 0xffffff;
    }
    return nic;
}

int64_t create(int64_t rank, int64_t world, int64_t max_bytes, int64_t slots, const std::vector<std::string>& devices,
               int64_t gid_index, int64_t cpu, int64_t flagw) {
    TORCH_CHECK(world >= 2 && rank >= 0 && rank < world, "rdma all-reduce: rank ", rank, " of ", world);
    TORCH_CHECK(max_bytes > 0 && max_bytes % 64 == 0, "rdma all-reduce: max_bytes must be a positive multiple of 64");
    TORCH_CHECK(slots >= 2, "rdma all-reduce: at least 2 slots");
    TORCH_CHECK(!devices.empty(), "rdma all-reduce: name at least one RDMA device");
    auto g = std::make_unique<Ring>();
    g->rank = (int)rank;
    g->world = (int)world;
    g->slots = (int)slots;
    g->max_bytes = (size_t)max_bytes;
    g->gid_index = (int)gid_index;
    g->cpu = (int)cpu;
    g->send_bytes = (size_t)slots * max_bytes;
    g->flagw = flagw ? 1 : 0;
    g->rflag_off = (size_t)slots * world * max_bytes;
    g->recv_bytes = g->rflag_off + (((size_t)slots * world * devices.size() * 8 + 4095) / 4096) * 4096;
    const size_t meta_b = ((size_t)slots * world + 1 + 2 * (size_t)slots + 1) * 8;
    g->meta_bytes = ((meta_b + 4095) / 4096) * 4096;
    g->send_ring = pinned(g->send_bytes);
    g->recv_ring = pinned(g->recv_bytes);
    g->rflags = reinterpret_cast<uint64_t*>(g->recv_ring + g->rflag_off);
    g->meta = pinned(g->meta_bytes);
    uint64_t* meta = reinterpret_cast<uint64_t*>(g->meta);
    g->flags = meta;
    g->doorbell = g->flags + (size_t)slots * world;
    g->sizes = g->doorbell + 1;
    g->sent = g->sizes + slots;
    g->abort_word = g->sent + slots;
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->seq), 8));
    C10_CUDA_CHECK(cudaMemset(g->seq, 0, 8));
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->staged), 8));
    C10_CUDA_CHECK(cudaMemset(g->staged, 0, 8));
    for (const auto& name : devices) g->nics.push_back(open_nic(name, *g));
    g->peers.assign(world, Peer{});
    rings.push_back(std::move(g));
    return (int64_t)rings.size() - 1;
}

// Per peer: what the other side's QPs toward this rank need: ([qpn], [psn], [gid], receive ring, [rkey]), a device each.
std::vector<py::tuple> local_info(int64_t h) {
    Ring& g = get(h);
    std::vector<py::tuple> out;
    for (int p = 0; p < g.world; ++p) {
        std::vector<uint32_t> qpn, psn, rkey;
        std::vector<py::bytes> gid;
        for (auto& nic : g.nics) {
            qpn.push_back(nic.qps[p] ? nic.qps[p]->qp_num : 0);
            psn.push_back(nic.psns[p]);
            rkey.push_back(nic.mr_recv->rkey);
            gid.push_back(py::bytes(reinterpret_cast<const char*>(nic.gid.raw), 16));
        }
        out.push_back(py::make_tuple(qpn, psn, gid, (uint64_t)(uintptr_t)g.recv_ring, rkey, g.flagw));
    }
    return out;
}

void link_peers(int64_t h, const std::vector<py::tuple>& remote) {
    Ring& g = get(h);
    TORCH_CHECK((int)remote.size() == g.world, "rdma all-reduce: one entry a rank");
    const int nd = (int)g.nics.size();
    for (int p = 0; p < g.world; ++p) {
        if (p == g.rank) continue;
        Peer& peer = g.peers[p];
        peer.qpn = remote[p][0].cast<std::vector<uint32_t>>();
        peer.psn = remote[p][1].cast<std::vector<uint32_t>>();
        peer.gid = remote[p][2].cast<std::vector<std::string>>();
        peer.recv = remote[p][3].cast<uint64_t>();
        peer.rkey = remote[p][4].cast<std::vector<uint32_t>>();
        TORCH_CHECK(remote[p].size() > 5 && remote[p][5].cast<int>() == g.flagw,
                    "rdma all-reduce: every rank must use the same flag-write mode (EXL3_RDMA_FLAGW)");
        TORCH_CHECK((int)peer.qpn.size() == nd, "rdma all-reduce: every rank must open the same number of devices");
        for (int d = 0; d < nd; ++d) {
            TORCH_CHECK(peer.gid[d].size() == 16, "rdma all-reduce: a GID is 16 bytes");
            ibv_qp_attr rtr{};
            rtr.qp_state = IBV_QPS_RTR;
            rtr.path_mtu = g.nics[d].mtu;
            rtr.dest_qp_num = peer.qpn[d];
            rtr.rq_psn = peer.psn[d];
            rtr.max_dest_rd_atomic = 1;
            rtr.min_rnr_timer = 12;
            rtr.ah_attr.is_global = 1;
            std::memcpy(rtr.ah_attr.grh.dgid.raw, peer.gid[d].data(), 16);
            rtr.ah_attr.grh.sgid_index = g.gid_index;
            rtr.ah_attr.grh.hop_limit = 64;
            rtr.ah_attr.port_num = g.port;
            TORCH_CHECK(ibv_modify_qp(g.nics[d].qps[p], &rtr,
                                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
                        "rdma all-reduce: QP to RTR failed (peer ", p, ", device ", d, ")");
            ibv_qp_attr rts{};
            rts.qp_state = IBV_QPS_RTS;
            rts.timeout = 14;
            rts.retry_cnt = 7;
            rts.rnr_retry = 7;
            rts.sq_psn = g.nics[d].psns[p];
            rts.max_rd_atomic = 1;
            TORCH_CHECK(ibv_modify_qp(g.nics[d].qps[p], &rts,
                                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0,
                        "rdma all-reduce: QP to RTS failed (peer ", p, ", device ", d, ")");
        }
    }
}

void start(int64_t h) {
    Ring& g = get(h);
    TORCH_CHECK(!g.running.load(), "rdma all-reduce: already started");
    g.running.store(true);
    g.proxy = std::thread(proxy_loop, &g);
}

void stop(int64_t h) {
    Ring& g = get(h);
    if (g.running.exchange(false) && g.proxy.joinable()) g.proxy.join();
}

std::string failure(int64_t h) {
    Ring& g = get(h);
    return g.failed.load() ? std::string(g.why) : std::string();
}

void abort_all(int64_t h, const std::string& why) { fail(get(h), why); }

int64_t max_bytes(int64_t h) { return (int64_t)get(h).max_bytes; }

// out = sum over ranks of x (bf16, contiguous, 16-byte aligned, a multiple of 8 elements), on the current stream.
void all_reduce(int64_t h, const at::Tensor& x, at::Tensor out) {
    Ring& g = get(h);
    TORCH_CHECK(x.is_cuda() && out.is_cuda() && x.scalar_type() == at::kBFloat16 &&
                    out.scalar_type() == at::kBFloat16 && x.is_contiguous() && out.is_contiguous(),
                "rdma all-reduce: contiguous bf16 CUDA tensors");
    const int64_t n = x.numel();
    TORCH_CHECK(out.numel() == n, "rdma all-reduce: out must match x");
    TORCH_CHECK(n % 8 == 0 && (uint64_t)n * 2 <= g.max_bytes, "rdma all-reduce: ", n, " elements (a multiple of 8, at most ",
                g.max_bytes / 2, ")");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "rdma all-reduce: 16-byte aligned tensors");
    if (g.failed.load()) TORCH_CHECK(false, "rdma all-reduce: ", g.why);
    const int n16 = (int)(n / 8);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const int sblocks = std::max(1, std::min(16, (n16 + 1023) / 1024));
    stage_kernel<uint4><<<sblocks, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const uint4*>(x.data_ptr()), n16,
                                                        dev(g.send_ring), dev(g.sizes), dev(g.doorbell), dev(g.sent),
                                                        g.seq, g.slots, g.max_bytes, dev(g.abort_word), g.staged);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int blocks = std::max(1, std::min(48, (n16 + SUM_THREADS - 1) / SUM_THREADS));
    sum_kernel<<<blocks, SUM_THREADS, 0, stream>>>(reinterpret_cast<const uint4*>(x.data_ptr()),
                                                   reinterpret_cast<uint4*>(out.data_ptr()), n16, dev(g.recv_ring),
                                                   dev(g.flags), g.seq, g.rank, g.world, g.slots, g.max_bytes,
                                                   dev(g.abort_word), g.flagw ? dev(g.rflags) : nullptr,
                                                   (int)g.nics.size());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dsv41 patch 0045
template <typename T>
void launch_gather(Ring& g, const at::Tensor& x, at::Tensor& out, int64_t bytes, int64_t inner_bytes) {
    const int n = (int)(bytes / (int64_t)sizeof(T)), inner = (int)(inner_bytes / (int64_t)sizeof(T));
    auto stream = c10::cuda::getCurrentCUDAStream();
    const int sblocks = std::max(1, std::min(16, (n + 1023) / 1024));
    stage_kernel<T><<<sblocks, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const T*>(x.data_ptr()), n,
                                                           dev(g.send_ring), dev(g.sizes), dev(g.doorbell),
                                                           dev(g.sent), g.seq, g.slots, g.max_bytes,
                                                           dev(g.abort_word), g.staged);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int blocks = std::max(1, std::min(48, (n * g.world + SUM_THREADS - 1) / SUM_THREADS));
    gather_kernel<T><<<blocks, SUM_THREADS, 0, stream>>>(reinterpret_cast<const T*>(x.data_ptr()),
                                                         reinterpret_cast<T*>(out.data_ptr()), n, inner,
                                                         dev(g.recv_ring), dev(g.flags), g.seq, g.rank, g.world,
                                                         g.slots, g.max_bytes, dev(g.abort_word),
                                                         g.flagw ? dev(g.rflags) : nullptr, (int)g.nics.size());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out = every rank's x concatenated in rank order along a dim, on the current stream: x is seen as [outer, inner]
// (inner = the bytes of x from that dim on), out as [outer, world, inner]. Any dtype (a byte copy); contiguous
// tensors of one dtype, the inner bytes a multiple of 4, 4-byte aligned (16-byte units when inner and both pointers
// allow), at most max_bytes a rank.
void all_gather(int64_t h, const at::Tensor& x, at::Tensor out, int64_t outer) {
    Ring& g = get(h);
    TORCH_CHECK(x.is_cuda() && out.is_cuda() && x.is_contiguous() && out.is_contiguous() &&
                    x.scalar_type() == out.scalar_type(), "rdma all-gather: contiguous CUDA tensors of one dtype");
    const int64_t bytes = x.numel() * (int64_t)x.element_size();
    TORCH_CHECK(out.numel() == x.numel() * g.world, "rdma all-gather: out must hold world x the elements of x");
    TORCH_CHECK(bytes > 0 && (uint64_t)bytes <= g.max_bytes, "rdma all-gather: ", bytes, " bytes (at most ",
                g.max_bytes, ")");
    TORCH_CHECK(outer > 0 && bytes % outer == 0 && (bytes / outer) % 4 == 0,
                "rdma all-gather: the inner bytes (", bytes, " / ", outer, ") must be a multiple of 4");
    const int64_t inner_bytes = bytes / outer;
    const uintptr_t px = reinterpret_cast<uintptr_t>(x.data_ptr()), po = reinterpret_cast<uintptr_t>(out.data_ptr());
    TORCH_CHECK(px % 4 == 0 && po % 4 == 0, "rdma all-gather: 4-byte aligned tensors");
    if (g.failed.load()) TORCH_CHECK(false, "rdma all-gather: ", g.why);
    if (inner_bytes % 16 == 0 && px % 16 == 0 && po % 16 == 0)
        launch_gather<uint4>(g, x, out, bytes, inner_bytes);
    else
        launch_gather<uint32_t>(g, x, out, bytes, inner_bytes);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create", &create);
    m.def("local_info", &local_info);
    m.def("connect", &link_peers);
    m.def("start", &start);
    m.def("stop", &stop);
    m.def("failure", &failure);
    m.def("abort", &abort_all);
    m.def("max_bytes", &max_bytes);
    m.def("all_reduce", &all_reduce);
    m.def("all_gather", &all_gather);
}
