// SPDX-License-Identifier: Apache-2.0
// exl3_mhc: DeepSeek-V4.1's mHC post + the next sublayer's input collapse + RMSNorm in one launch, for decode sizes.
//
// vLLM runs these as two TileLang kernels with one block per token (6 blocks at a k=5 verify step), so both are
// latency-bound: ~12 us post + ~9 us collapse at every sublayer boundary, 80 a round, on the critical path. Here the
// grid is token x 256-column chunk (120 blocks at 6 tokens):
//   residual_out[t, o, c] = bf16( post[t, o] * x[t, c] + sum_i comb[t, i, o] * residual[t, i, c] )      (fp32, i in order)
//   rounded[t, c]         = bf16( sum_o pre[t, o] * residual_out[t, o, c] )                              (o in order)
//   layer_input[t, c]     = bf16( rounded[t, c] * rsqrt(sum_c rounded^2 / H + eps) * norm_weight[c] )
// which is vLLM's mhc_post_tilelang followed by mhc_pre_big_fuse_with_norm in split mode "input" with a carried
// pre-mix (save_pre_mix keeps the bf16 boundary before the RMSNorm). Each chunk block writes its rounded values and its
// partial sum of squares; the last block of a token (one counter a token, reset by that block, so graph replays start
// clean) adds the partials in chunk order and normalizes the token's row. The sum of squares is added in a different
// order than TileLang's (per position, then a tree), so rsqrt can differ in its last bits: rounding class.
//
// post_collapse_exact (dsv41 patch 0048, VLLM_DSV41_MHC_FUSED_POST=2): the same fusion with TileLang's bits. One block
// a token, 128 threads; thread (q, h) with q in [0, 64), h in {0, 1} owns the columns b * 1024 + h * 512 + q * 8 + [0, 8)
// of every 1024-column block b: exactly TileLang thread q's slots 8h..8h+7 in mhc_pre_big_fuse_with_norm (split mode
// "input", hidden_block 1024, 64 collapse threads; generated code read from the serve image's TileLang cache). So the
// sum of squares is TileLang's: per slot an fma chain over the blocks in order, per q the slots added in the order
// 0, 8, 1, 9, ..., 7, 15 (h = 0 and h = 1 of a q are adjacent lanes), then the butterfly over q with xor 32, 16, 8, 4,
// 2, 1; rsqrt(sum / H + eps) and (v * rsqrt) * w as TileLang writes them. The post and the collapse are written in the
// same C form as TileLang's generated code (v = c * d; v = v + a * b; ...), so nvcc contracts them to the same FMAs.
// The post's output stays in registers for the collapse and the RMSNorm (no second pass over memory, no atomics).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {

constexpr int HC = 4;           // mHC streams (V4.1: hc_mult 4)
constexpr int CH = 256;         // columns a block
constexpr int THREADS = 64;     // 4 columns a thread
constexpr int MAX_CHUNKS = 64;  // hidden up to 16384

__global__ void __launch_bounds__(THREADS) post_collapse_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ residual, const float* __restrict__ comb,
    const float* __restrict__ post, const float* __restrict__ pre, const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ layer_input, float* __restrict__ part,
    int* __restrict__ counter, int H, float eps)
{
    const int t = blockIdx.y, chunk = blockIdx.x, nchunks = gridDim.x;
    const int c0 = chunk * CH + threadIdx.x * 4;
    __shared__ float red[THREADS / 32];
    __shared__ bool last;
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    float cm[HC][HC], pm[HC], qm[HC];
    #pragma unroll
    for (int i = 0; i < HC; ++i)
    {
        pm[i] = post[t * HC + i];
        qm[i] = pre[t * HC + i];
        #pragma unroll
        for (int o = 0; o < HC; ++o) cm[i][o] = comb[(t * HC + i) * HC + o];
    }
    float sq = 0.f;
    if (c0 < H)
    {
        float xv[4], rv[HC][4];
        const uint2 xa = *reinterpret_cast<const uint2*>(x + (size_t) t * H + c0);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xa);
        xv[0] = __low2float(xp[0]); xv[1] = __high2float(xp[0]); xv[2] = __low2float(xp[1]); xv[3] = __high2float(xp[1]);
        #pragma unroll
        for (int i = 0; i < HC; ++i)
        {
            const uint2 ra = *reinterpret_cast<const uint2*>(residual + ((size_t) t * HC + i) * H + c0);
            const __nv_bfloat162* rp = reinterpret_cast<const __nv_bfloat162*>(&ra);
            rv[i][0] = __low2float(rp[0]); rv[i][1] = __high2float(rp[0]);
            rv[i][2] = __low2float(rp[1]); rv[i][3] = __high2float(rp[1]);
        }
        float col[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int o = 0; o < HC; ++o)
        {
            __nv_bfloat16 ob[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float v = pm[o] * xv[j];
                #pragma unroll
                for (int i = 0; i < HC; ++i) v += cm[i][o] * rv[i][j];
                ob[j] = __float2bfloat16_rn(v);
                col[j] += qm[o] * __bfloat162float(ob[j]);       // the collapse reads the rounded residual
            }
            uint2 w;
            __nv_bfloat162* wp = reinterpret_cast<__nv_bfloat162*>(&w);
            wp[0] = __halves2bfloat162(ob[0], ob[1]);
            wp[1] = __halves2bfloat162(ob[2], ob[3]);
            *reinterpret_cast<uint2*>(residual_out + ((size_t) t * HC + o) * H + c0) = w;
        }
        __nv_bfloat16 rb[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            rb[j] = __float2bfloat16_rn(col[j]);
            const float r = __bfloat162float(rb[j]);
            sq += r * r;
        }
        uint2 w;
        __nv_bfloat162* wp = reinterpret_cast<__nv_bfloat162*>(&w);
        wp[0] = __halves2bfloat162(rb[0], rb[1]);
        wp[1] = __halves2bfloat162(rb[2], rb[3]);
        *reinterpret_cast<uint2*>(layer_input + (size_t) t * H + c0) = w;   // unnormalized for now
    }
    // the block's partial sum of squares, in a fixed order (warp shuffles, then the two warps in order)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        part[t * MAX_CHUNKS + chunk] = red[0] + red[1];
        __threadfence();
        const int prev = atomicAdd(counter + t, 1);
        last = prev == nchunks - 1;
        if (last) counter[t] = 0;                    // self-reset for the next launch / graph replay
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    float tot = 0.f;
    if (threadIdx.x == 0)
        for (int k = 0; k < nchunks; ++k) tot += __ldcg(part + t * MAX_CHUNKS + k);
    if (threadIdx.x == 0) red[0] = tot;
    __syncthreads();
    const float scale = rsqrtf(red[0] / (float) H + eps);
    for (int c = threadIdx.x * 4; c < H; c += THREADS * 4)
    {
        uint2 a = __ldcg(reinterpret_cast<const uint2*>(layer_input + (size_t) t * H + c));
        const uint2 wa = *reinterpret_cast<const uint2*>(norm_w + c);
        __nv_bfloat162* ap = reinterpret_cast<__nv_bfloat162*>(&a);
        const __nv_bfloat162* wq = reinterpret_cast<const __nv_bfloat162*>(&wa);
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const float lo = __low2float(ap[j]) * scale * __low2float(wq[j]);
            const float hi = __high2float(ap[j]) * scale * __high2float(wq[j]);
            ap[j] = __floats2bfloat162_rn(lo, hi);
        }
        *reinterpret_cast<uint2*>(layer_input + (size_t) t * H + c) = a;
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

// ---- post_collapse_exact (patch 0048) ---------------------------------------------------------------------------------
constexpr int XT = 128;         // threads a block (one block a token)
constexpr int XB = 1024;        // TileLang's hidden_block: gcd(1024, H)

__device__ __forceinline__ void unpack8(const uint4& a, float (&f)[8])
{
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&a);
    #pragma unroll
    for (int i = 0; i < 4; ++i) { f[2 * i] = __low2float(p[i]); f[2 * i + 1] = __high2float(p[i]); }
}

template <int NB>
__global__ void __launch_bounds__(XT) post_collapse_exact_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ residual, const float* __restrict__ comb,
    const float* __restrict__ post, const float* __restrict__ pre, const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ layer_input, float eps)
{
    constexpr int H = NB * XB;
    const int t = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q = warp * 16 + (lane >> 1), h = lane & 1;   // TileLang collapse thread q, slot half h
    __shared__ float red[2][64];

    // the RMSNorm weights do not depend on earlier kernels: read them before the PDL wait
    uint4 wv[NB];
    #pragma unroll
    for (int b = 0; b < NB; ++b) wv[b] = *reinterpret_cast<const uint4*>(norm_w + b * XB + h * 512 + q * 8);
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    float cm[HC][HC], pm[HC], qm[HC];
    #pragma unroll
    for (int i = 0; i < HC; ++i)
    {
        pm[i] = post[t * HC + i];
        qm[i] = pre[t * HC + i];
        #pragma unroll
        for (int o = 0; o < HC; ++o) cm[i][o] = comb[(t * HC + i) * HC + o];
    }
    // every load of the token's columns in flight at once
    uint4 xa[NB], ra[NB][HC];
    #pragma unroll
    for (int b = 0; b < NB; ++b)
    {
        const int c = b * XB + h * 512 + q * 8;
        xa[b] = *reinterpret_cast<const uint4*>(x + (size_t) t * H + c);
        #pragma unroll
        for (int i = 0; i < HC; ++i) ra[b][i] = *reinterpret_cast<const uint4*>(residual + ((size_t) t * HC + i) * H + c);
    }
    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
    uint4 rnd[NB];                              // the collapse rounded to bf16, kept for the RMSNorm
    #pragma unroll
    for (int b = 0; b < NB; ++b)
    {
        const int c = b * XB + h * 512 + q * 8;
        float xv[8], rv[HC][8];
        unpack8(xa[b], xv);
        #pragma unroll
        for (int i = 0; i < HC; ++i) unpack8(ra[b][i], rv[i]);
        float col[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) col[j] = 0.f;
        #pragma unroll
        for (int o = 0; o < HC; ++o)
        {
            __nv_bfloat16 ob[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                // mhc_post_tilelang: x_local = c_local[o] * d_local[j]; x_local = x_local + a_local[i * 4 + o] * b_local[i][j]
                float v = pm[o] * xv[j];
                #pragma unroll
                for (int i = 0; i < HC; ++i) v = v + cm[i][o] * rv[i][j];
                ob[j] = __float2bfloat16_rn(v);
            }
            uint4 w;
            __nv_bfloat162* wp = reinterpret_cast<__nv_bfloat162*>(&w);
            #pragma unroll
            for (int k = 0; k < 4; ++k) wp[k] = __halves2bfloat162(ob[2 * k], ob[2 * k + 1]);
            *reinterpret_cast<uint4*>(residual_out + ((size_t) t * HC + o) * H + c) = w;
            // pre_big_fuse "input": ol = ol + pre * xl, streams in order, from the stored (rounded) residual
            #pragma unroll
            for (int j = 0; j < 8; ++j) col[j] = col[j] + qm[o] * __bfloat162float(ob[j]);
        }
        __nv_bfloat16 rb[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            rb[j] = __float2bfloat16_rn(col[j]);
            const float r = __bfloat162float(rb[j]);
            acc[j] = __fmaf_rn(r, r, acc[j]);   // sumsq_per_pos: tl::fma2(value, value, acc), blocks in order
        }
        __nv_bfloat162* rp = reinterpret_cast<__nv_bfloat162*>(&rnd[b]);
        #pragma unroll
        for (int k = 0; k < 4; ++k) rp[k] = __halves2bfloat162(rb[2 * k], rb[2 * k + 1]);
    }
    // TileLang thread q's sum: slots 0, 8, 1, 9, ..., 7, 15 (h = 0 then h = 1), from 0
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        const float other = __shfl_xor_sync(0xffffffffu, acc[j], 1);
        const float lo = h ? other : acc[j], hi = h ? acc[j] : other;
        s = s + lo;
        s = s + hi;
    }
    // tl::AllReduce<SumOp, 64> (TileLang 0.1.12 reduce.h): the largest offset first, q ^ 32 through shared memory, then
    // q ^ 16, 8, 4, 2, 1 (here q ^ 16 crosses our warps, so shared memory too; lane ^ 2k = q ^ k). -DMHCX_TREE=1: the
    // other plausible tree (warp offsets first, then q ^ 32), kept as a build flag for the bit test should the
    // installed reduce.h differ.
#if defined(MHCX_TREE) && MHCX_TREE == 1
    if (h == 0) red[1][q] = s;                 // TileLang's warp first: q ^ 16, 8, 4, 2, 1, then q ^ 32
    __syncthreads();
    s = s + red[1][q ^ 16];
    #pragma unroll
    for (int off = 16; off >= 2; off >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, off);
    if (h == 0) red[0][q] = s;
    __syncthreads();
    s = s + red[0][q ^ 32];
#else
    if (h == 0) red[0][q] = s;
    __syncthreads();
    s = s + red[0][q ^ 32];
    if (h == 0) red[1][q] = s;
    __syncthreads();
    s = s + red[1][q ^ 16];
    #pragma unroll
    for (int off = 16; off >= 2; off >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, off);
#endif
    const float scale = rsqrtf(s / (float) H + eps);
    #pragma unroll
    for (int b = 0; b < NB; ++b)
    {
        const int c = b * XB + h * 512 + q * 8;
        float v[8], w[8];
        unpack8(rnd[b], v);
        unpack8(wv[b], w);
        uint4 o;
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            op[k] = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(v[2 * k], scale), w[2 * k]),
                                          __fmul_rn(__fmul_rn(v[2 * k + 1], scale), w[2 * k + 1]));
        *reinterpret_cast<uint4*>(layer_input + (size_t) t * H + c) = o;
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

}  // namespace

// x [n, H] bf16 (the reduced sublayer output), residual [n, 4, H] bf16, comb [n, 4, 4] f32, post [n, 4] f32,
// pre [n, 4] f32 (the carried pre-mix), norm_w [H] bf16 -> residual_out [n, 4, H], layer_input [n, H] (bf16);
// scratch: part [n, 64] f32, counter [n] int32 (zeroed once; the kernel leaves it zeroed)
void post_collapse(torch::Tensor x, torch::Tensor residual, torch::Tensor comb, torch::Tensor post, torch::Tensor pre,
                   torch::Tensor norm_w, torch::Tensor residual_out, torch::Tensor layer_input, torch::Tensor part,
                   torch::Tensor counter, double eps)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const int n = x.size(0), H = x.size(1);
    TORCH_CHECK(residual.size(1) == HC && residual.size(2) == H, "residual must be [n, 4, H]");
    TORCH_CHECK(H % CH == 0 && H / CH <= MAX_CHUNKS, "H must be a multiple of 256, at most 16384");
    TORCH_CHECK(counter.size(0) >= n && part.size(0) >= n && part.size(1) >= MAX_CHUNKS, "scratch too small");
    for (auto* tp : {&x, &residual, &norm_w, &residual_out, &layer_input})
        TORCH_CHECK(tp->is_contiguous() && tp->scalar_type() == at::kBFloat16, "bf16 contiguous tensors");
    for (auto* tp : {&comb, &post, &pre})
        TORCH_CHECK(tp->is_contiguous() && tp->scalar_type() == at::kFloat, "fp32 contiguous mixes");
    if (n == 0) return;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(H / CH, n);
    cfg.blockDim = dim3(THREADS);
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, post_collapse_kernel,
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        comb.data_ptr<float>(), post.data_ptr<float>(), pre.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()), reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr()), part.data_ptr<float>(), counter.data_ptr<int>(), H,
        (float) eps));
}

// post_collapse with TileLang's bits (patch 0048): same tensors, no scratch; H = 1024 * NB with NB 5 (V4.1-Flash) or 7.
void post_collapse_exact(torch::Tensor x, torch::Tensor residual, torch::Tensor comb, torch::Tensor post,
                         torch::Tensor pre, torch::Tensor norm_w, torch::Tensor residual_out, torch::Tensor layer_input,
                         double eps)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const int n = x.size(0), H = x.size(1);
    TORCH_CHECK(residual.size(0) == n && residual.size(1) == HC && residual.size(2) == H, "residual must be [n, 4, H]");
    TORCH_CHECK(H == 5 * XB || H == 7 * XB, "post_collapse_exact: H must be 5120 or 7168");
    for (auto* tp : {&x, &residual, &norm_w, &residual_out, &layer_input})
        TORCH_CHECK(tp->is_contiguous() && tp->scalar_type() == at::kBFloat16, "bf16 contiguous tensors");
    for (auto* tp : {&comb, &post, &pre})
        TORCH_CHECK(tp->is_contiguous() && tp->scalar_type() == at::kFloat, "fp32 contiguous mixes");
    TORCH_CHECK(comb.numel() == n * HC * HC && post.numel() == n * HC && pre.numel() == n * HC, "mix shapes");
    if (n == 0) return;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n);
    cfg.blockDim = dim3(XT);
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    auto* k = H == 5 * XB ? post_collapse_exact_kernel<5> : post_collapse_exact_kernel<7>;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k,
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        comb.data_ptr<float>(), post.data_ptr<float>(), pre.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()), reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr()), (float) eps));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("post_collapse", &post_collapse, "mHC post + carried-pre collapse + RMSNorm, one launch (decode sizes)");
    m.def("post_collapse_exact", &post_collapse_exact,
          "post_collapse with TileLang's bits: one block a token, TileLang's sum-of-squares order (patch 0048)");
}
