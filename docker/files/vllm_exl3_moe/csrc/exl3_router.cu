// SPDX-License-Identifier: Apache-2.0
// MoE router at decode sizes: logits = x @ W^T (bf16 x [M, K] with M <= 8, bf16 W [N, K], fp32 logits [M, N]) streamed
// at DRAM rate, optionally followed in the same launch by DeepSeek-V4's top-k, run by the last block to finish (a
// grid-wide counter that block resets, so CUDA-graph replays work).
//
// Why: vLLM's GateLinear tiers skip family-120 Blackwell (GB10), which leaves cuBLAS bf16 x bf16 -> fp32 with a split-K
// reduce, ~30 us + a reduce kernel per layer for the 3.75 MiB [384, 5120] gate (half of DRAM rate), then a top-k kernel.
//
// Layout: one warp per expert row (8 a block, so 48 blocks for 384 experts). The weights do not depend on the previous
// kernel, so each lane issues its first row loads before the PDL wait; then x is staged in shared memory (all M rows,
// M * K * 2 bytes) and the row streams in 16-byte loads, PF of them in flight a lane. Each lane sums its columns in a
// fixed order and the warp reduces by a fixed butterfly: the logits are deterministic (not bit-equal to cuBLAS, whose
// split-K order differs; both accumulate in fp32).
//
// Top-k (fused): DeepSeek-V4's routing as vLLM's dsv4_topk (fused_moe/router/dsv4_topk.py): weight = sqrt(softplus(l))
// (softplus = l above 20), score = weight + correction bias (bias_vl for image-sentinel tokens), NaN scores -> -1e30,
// k rounds of argmax with the lowest expert index on equal scores, the selected weights (no bias) renormalized to sum 1
// and multiplied by routed_scaling_factor. Each expert's warp computes its scores and weights (one lane a row) and
// writes them to a workspace, so the last block only selects: one warp a row, each lane keeps its experts' top k sorted
// (score descending, index ascending), then k rounds of a warp argmax over the lanes' list heads.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>         // c10, not ATen/cuda: the image lacks cusparse.h
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cfloat>
#include <climits>

#include "exl3_common.cuh"
#include "exl3_launch.h"

namespace {

constexpr int WARPS = 8;                 // expert rows a block, one warp each
constexpr int MAXM = 8;                  // rows (decode sizes; x is staged whole in shared memory)
// dsv41 patch 0049: the prefetch depth and the x staging batch are build flags (EXL3_ROUTER_FLAGS="-DROUTER_PF=20
// -DROUTER_XS=8"; the JIT hash includes the flags). Neither changes the arithmetic: every lane still adds its row's
// 16-byte chunks in chunk order (j = it * 32 + lane, it ascending) and the warp reduces by the same butterfly, so the
// logits keep their bits; only how many loads are in flight changes. Defaults = the kernel as before.
#ifndef ROUTER_PF
#define ROUTER_PF 4
#endif
#ifndef ROUTER_XS
#define ROUTER_XS 1
#endif
constexpr int PF = ROUTER_PF;            // 16-byte row loads in flight a lane
constexpr int XS = ROUTER_XS;            // x staging: 16-byte loads a thread issues before storing them
constexpr int MAXN = 384;                // experts the fused top-k handles (a multiple of 32)
constexpr int MAXK = 8;                  // top-k

__device__ __forceinline__ uint4 ld_stream(const uint4* p)
{
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float dot8(const uint4& w, const uint4& x)
{
    const __nv_bfloat162* a = reinterpret_cast<const __nv_bfloat162*>(&w);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&x);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fa = __bfloat1622float2(a[i]);
        const float2 fb = __bfloat1622float2(b[i]);
        s = fmaf(fa.x, fb.x, s);
        s = fmaf(fa.y, fb.y, s);
    }
    return s;
}

struct TopK {
    const float* bias;                   // [N] correction bias
    const float* bias_vl;                // [N] or null: bias for image-sentinel tokens
    const int64_t* ids;                  // [M] token ids (with bias_vl)
    long long sentinel_lo;               // image sentinels: ids sentinel_lo .. sentinel_lo + 4
    float scale;                         // routed_scaling_factor
    int k;                               // top-k (<= MAXK)
    float* w;                            // [M, k] fp32
    int* idx;                            // [M, k] int32
    unsigned* counter;                   // zero before the first launch; the last block resets it
    float* ws;                           // [2, M, N]: scores, then weights
};

template <int M, bool FUSED>
__global__ void __launch_bounds__(WARPS * 32) router_kernel(const __nv_bfloat16* __restrict__ x, int ldx,
                                                            const __nv_bfloat16* __restrict__ w,
                                                            float* __restrict__ logits, int K, int N,
                                                            const __grid_constant__ TopK t)
{
    extern __shared__ uint4 xs[];        // [M][K / 8]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * WARPS + warp;
    const int K8 = K >> 3;
    const int nit = (K8 + 31) >> 5;      // 16-byte loads a lane over the row
    const uint4* wr = reinterpret_cast<const uint4*>(w) + (size_t) (e < N ? e : N - 1) * K8;

    // Weights first: they do not depend on the previous kernel (PDL lets these loads overlap its tail).
    uint4 pre[PF];
#pragma unroll
    for (int i = 0; i < PF; ++i) {
        const int j = i * 32 + lane;
        pre[i] = j < K8 ? ld_stream(wr + j) : make_uint4(0, 0, 0, 0);
    }
    exl3c::griddep_wait();
    if constexpr (XS <= 1) {
        for (int i = threadIdx.x; i < M * K8; i += blockDim.x) {
            const int m = i / K8, j = i - m * K8;
            xs[i] = reinterpret_cast<const uint4*>(x + (size_t) m * ldx)[j];
        }
    } else {
        for (int i0 = threadIdx.x; i0 < M * K8; i0 += XS * blockDim.x) {
            uint4 v[XS];
#pragma unroll
            for (int u = 0; u < XS; ++u) {
                const int i = i0 + u * blockDim.x;
                if (i < M * K8) {
                    const int m = i / K8, j = i - m * K8;
                    v[u] = reinterpret_cast<const uint4*>(x + (size_t) m * ldx)[j];
                }
            }
#pragma unroll
            for (int u = 0; u < XS; ++u) {
                const int i = i0 + u * blockDim.x;
                if (i < M * K8) xs[i] = v[u];
            }
        }
    }
    __syncthreads();

    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int base = 0; base < nit; base += PF) {
        uint4 cur[PF];
#pragma unroll
        for (int i = 0; i < PF; ++i) cur[i] = pre[i];
#pragma unroll
        for (int i = 0; i < PF; ++i) {
            const int j = (base + PF + i) * 32 + lane;
            pre[i] = j < K8 ? ld_stream(wr + j) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < PF; ++i) {
            const int j = (base + i) * 32 + lane;
            if (j < K8) {
#pragma unroll
                for (int m = 0; m < M; ++m) acc[m] += dot8(cur[i], xs[m * K8 + j]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
    }
    exl3c::griddep_launch();
    if (e < N) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            if (lane != m) continue;               // lane m writes row m (every lane holds every row's sum)
            const float l = acc[m];
            logits[(size_t) m * N + e] = l;
            if constexpr (FUSED) {
                bool img = false;
                if (t.bias_vl != nullptr) {
                    const long long id = t.ids[m];
                    img = id >= t.sentinel_lo && id < t.sentinel_lo + 5;
                }
                const float wt = sqrtf(l > 20.f ? l : logf(1.f + expf(l)));
                const float sc = wt + (img ? t.bias_vl[e] : t.bias[e]);
                t.ws[(size_t) m * N + e] = sc == sc ? sc : -1e30f;
                t.ws[(size_t) (M + m) * N + e] = wt;
                __threadfence();
            }
        }
    }
    if constexpr (!FUSED) return;

    // Last block to finish: select the top k of each row.
    __shared__ bool last;
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(t.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int npl = N >> 5;                        // experts a lane: lane, lane + 32, ...
    for (int m = warp; m < M; m += WARPS) {
        float ls[MAXK];                            // this lane's best k: score descending, index ascending
        int li[MAXK];
#pragma unroll
        for (int s = 0; s < MAXK; ++s) { ls[s] = -INFINITY; li[s] = INT_MAX; }
#pragma unroll
        for (int q = 0; q < MAXN / 32; ++q) {
            if (q >= npl) break;
            float v = __ldcg(t.ws + (size_t) m * N + q * 32 + lane);
            int vi = q * 32 + lane;                // ascending within the lane: equal scores keep the lower index first
#pragma unroll
            for (int s = 0; s < MAXK; ++s) {       // insertion: carry the smaller element down the list
                const bool better = v > ls[s];
                const float ts = ls[s];
                const int ti = li[s];
                ls[s] = better ? v : ls[s];
                li[s] = better ? vi : li[s];
                v = better ? ts : v;
                vi = better ? ti : vi;
            }
        }
        float selw[MAXK];
        int seli[MAXK];
#pragma unroll
        for (int s = 0; s < MAXK; ++s) {
            float bv = ls[0];
            int bi = li[0];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            seli[s] = bi;
            selw[s] = __ldcg(t.ws + (size_t) (M + m) * N + bi);
            if (li[0] == bi) {                     // the owner pops its head
#pragma unroll
                for (int r = 0; r < MAXK - 1; ++r) { ls[r] = ls[r + 1]; li[r] = li[r + 1]; }
                ls[MAXK - 1] = -INFINITY;
                li[MAXK - 1] = INT_MAX;
            }
        }
        if (lane == 0) {
            float sum = 0.f;
#pragma unroll
            for (int s = 0; s < MAXK; ++s) sum += s < t.k ? selw[s] : 0.f;
            const float f = t.scale / (sum > 0.f ? sum : 1.f);
#pragma unroll
            for (int s = 0; s < MAXK; ++s) {
                if (s < t.k) {
                    t.w[m * t.k + s] = selw[s] * f;
                    t.idx[m * t.k + s] = seli[s];
                }
            }
        }
    }
    if (threadIdx.x == 0) *t.counter = 0u;
}

template <int M, bool FUSED>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& logits, const TopK& t)
{
    const int K = (int) x.size(1), N = (int) w.size(0);
    const size_t smem = (size_t) M * K * 2;
    auto k = router_kernel<M, FUSED>;
    static size_t set = 0;               // per instantiation: raise the dynamic smem cap once (to the largest asked)
    if (smem > 48 * 1024 && smem > set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        set = smem;
    }
    exl3c::launch_k(k, dim3((N + WARPS - 1) / WARPS), dim3(WARPS * 32), smem, c10::cuda::getCurrentCUDAStream(),
                    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int) x.stride(0),
                    reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), logits.data_ptr<float>(), K, N, t);
}

template <bool FUSED>
void dispatch(const at::Tensor& x, const at::Tensor& w, at::Tensor& logits, const TopK& t)
{
    switch (x.size(0)) {
        case 1: launch<1, FUSED>(x, w, logits, t); break;
        case 2: launch<2, FUSED>(x, w, logits, t); break;
        case 3: launch<3, FUSED>(x, w, logits, t); break;
        case 4: launch<4, FUSED>(x, w, logits, t); break;
        case 5: launch<5, FUSED>(x, w, logits, t); break;
        case 6: launch<6, FUSED>(x, w, logits, t); break;
        case 7: launch<7, FUSED>(x, w, logits, t); break;
        case 8: launch<8, FUSED>(x, w, logits, t); break;
        default: TORCH_CHECK(false, "router: 1..", MAXM, " rows");
    }
}

void check(const at::Tensor& x, const at::Tensor& w, const at::Tensor& logits)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && logits.is_cuda(), "router: CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "router: bf16 x and weight");
    TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.is_contiguous(), "router: contiguous fp32 logits");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "router: x [M, K], weight [N, K]");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= MAXM, "router: 1..", MAXM, " rows");
    TORCH_CHECK(logits.size(0) == x.size(0) && logits.size(1) == w.size(0), "router: logits [M, N]");
    TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && w.is_contiguous() && x.size(1) % 8 == 0,
                "router: K a multiple of 8, rows contiguous, row stride a multiple of 8");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "router: 16-byte aligned x and weight");
    TORCH_CHECK((size_t) x.size(0) * x.size(1) * 2 <= 96 * 1024, "router: M * K * 2 bytes must fit shared memory");
}

}  // namespace

int64_t router_max_rows() { return MAXM; }

// logits[M, N] = x[M, K] @ weight[N, K]^T in fp32, on the current stream.
void router_logits(const at::Tensor& x, const at::Tensor& weight, at::Tensor& logits)
{
    check(x, weight, logits);
    const c10::cuda::CUDAGuard g(x.device());
    TopK t{};
    dispatch<false>(x, weight, logits, t);
}

// router_logits, then DeepSeek-V4's top-k in the same launch: topk_w [M, k] fp32, topk_ids [M, k] int32. bias_vl and
// ids empty (numel 0) when the model has no image routing; counter: a zeroed int32 [1] owned by the caller (one per
// concurrent caller), left zero; ws: fp32 scratch of at least 2 * M * N.
void router_topk(const at::Tensor& x, const at::Tensor& weight, at::Tensor& logits, const at::Tensor& bias,
                 const at::Tensor& bias_vl, const at::Tensor& ids, int64_t sentinel_lo, double scale,
                 at::Tensor& topk_w, at::Tensor& topk_ids, at::Tensor& counter, at::Tensor& ws)
{
    check(x, weight, logits);
    const int M = (int) x.size(0), N = (int) weight.size(0), k = (int) topk_w.size(1);
    TORCH_CHECK(N % 32 == 0 && N <= MAXN, "router_topk: experts a multiple of 32, at most ", MAXN);
    TORCH_CHECK(k >= 1 && k <= MAXK && topk_ids.size(1) == k && topk_w.size(0) == M && topk_ids.size(0) == M,
                "router_topk: topk_w / topk_ids [M, k], k <= ", MAXK);
    TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_ids.scalar_type() == at::kInt && topk_w.is_contiguous() &&
                topk_ids.is_contiguous(), "router_topk: contiguous fp32 weights, int32 ids");
    TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.is_contiguous() && bias.numel() == N, "router_topk: fp32 bias [N]");
    TORCH_CHECK(counter.scalar_type() == at::kInt && counter.numel() >= 1 && counter.is_cuda(), "router_topk: int32 counter");
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.is_contiguous() && ws.numel() >= 2 * M * N, "router_topk: fp32 ws [2, M, N]");
    const bool vl = bias_vl.numel() > 0;
    if (vl) {
        TORCH_CHECK(bias_vl.scalar_type() == at::kFloat && bias_vl.is_contiguous() && bias_vl.numel() == N,
                    "router_topk: fp32 bias_vl [N]");
        TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous() && ids.numel() >= M, "router_topk: int64 ids [M]");
    }
    const c10::cuda::CUDAGuard g(x.device());
    TopK t{bias.data_ptr<float>(), vl ? bias_vl.data_ptr<float>() : nullptr, vl ? ids.data_ptr<int64_t>() : nullptr,
           (long long) sentinel_lo, (float) scale, k, topk_w.data_ptr<float>(), topk_ids.data_ptr<int>(),
           reinterpret_cast<unsigned*>(counter.data_ptr<int>()), ws.data_ptr<float>()};
    dispatch<true>(x, weight, logits, t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("router_logits", &router_logits, "MoE router logits at decode sizes (bf16 -> fp32)");
    m.def("router_topk", &router_topk, "router logits + DeepSeek-V4 top-k in one launch");
    m.def("router_max_rows", &router_max_rows, "rows a launch");
}
