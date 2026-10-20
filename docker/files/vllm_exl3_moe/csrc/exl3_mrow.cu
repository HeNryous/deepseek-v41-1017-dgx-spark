// SPDX-License-Identifier: Apache-2.0
// exl3_mrow: EXL3 linears (mul1 codebook, K = 2..6) at decode sizes (1..32 rows), several linears per launch.
// dsv41 patch 0042b (19.10.2026, mrow32_tuning; replaces the 0042 draft): 17..32 rows run in mrow32_kernel as two m16
// tiles on the same decoded trellis fragments (weights read and decoded once for all rows), on a second plan g32 with
// its own workspace and counters. 1..16 rows: the original kernel, plan, workspace and counters, unchanged (same bits).
// Variants of the 32-row kernel, EXL3_MROW32_VARIANT (read at registration; default ring):
//   draft  0042 as measured in fenster14c: the item's whole k-slice of 32 input rows in shared memory (slices capped at
//          1280 at K5 so it fits: ~99 KB, one block per SM; wo_b 160 items instead of 80), rows m..31 zeroed one half
//          at a time, 4 transform tasks per warp in flight, 8 trellis stages
//   lean   draft's plan and layout without the zero fill, 8 transform tasks in flight, the trellis ring as deep as the
//          epilogue buffer it shares (12 stages at K5, no extra shared memory), k-split sum with its loads ahead
//   ring   the input transformed per 128-wide k block inside the main loop (loads in flight under the mma, one consumer
//          barrier per block) into two stages of 32 x 128 halves instead of the whole slice: ~27 KB of shared memory at
//          K5 for any slice, so the 16-row plan's k-slices (no cap, no extra items) and two blocks per SM (MROW32_MINB)
//   ring1  ring with one block per SM budgeted (no register cap) and 16 trellis stages
// Why: with 0042 18 / 24 rows cost 1.23 / 1.31 x 16 rows (fenster14c) although 5 of 6 dense plans were unchanged: the
// ~99 KB items ran one per SM, transformed the whole input slice (and zeroed the unused rows) before their weight stream
// could pass the first 8 stages, and summed k-slices with one dependent load at a time.
// Tuning knobs (registration): EXL3_MROW32_KS caps the k-slice, EXL3_MROW32_ITEMS sets the item target (0: as for 16
// rows), EXL3_MROW32_MIN_ROWS (default 17) sends fewer rows to the 32-row kernel too (measurement only).
//
// One work item = one problem (linear) x one 128-column output block x one k-slice. Eight consumer warps (one 16-column
// tile each) transform the item's slice of the input (bf16 -> * suh -> 128-wide Hadamard -> fp16 in shared memory),
// decode the trellis tiles straight into tensor-core B fragments (the decode's register layout is the m16n8k16 B
// layout), accumulate in fp32 with mma.sync, apply the output Hadamard and svh to their partial (both linear, so per
// k-slice is exact), and the last block of an output block (one atomic counter, no waiting) sums the k-slices in slice
// order and writes bf16. Deterministic, graph-safe (no spin across blocks, no co-residency assumption), counters
// self-reset.
// Weights stream through a ring of MROW_PIPE shared-memory stages. A stage is one k-row of the item's 8 tiles, which is
// contiguous in the trellis (8 x 32K bytes), fetched with one TMA bulk copy that a producer warp issues as soon as the
// stage is free (mbarriers; no block-wide barrier in the loop). Why: GB10 reads 1280-byte chunks per k-row at 230-237
// GB/s but 160-byte per-warp tiles at 202-213 GB/s (read-pattern microbench, Oct 9), and per-warp direct loads left the
// warps waiting on DRAM latency (ncu: long-scoreboard stalls, 12 of 16.6 cycles per issued instruction at wo_b).
//
// The trellis decode it uses (exl3_common.cuh) is adapted from exllamav3 v1.6.0, MIT License, Copyright (c) 2025
// Turboderp; the notice is in LICENSE.exllamav3 next to this file.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_common.cuh"
#include "exl3_launch.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using namespace exl3c;

constexpr int NB = 128;          // output columns per work item (one output Hadamard block)
constexpr int WARPS = 8;         // consumer warps, one 16-column tile each
constexpr int CONSUMER_THREADS = WARPS * 32;
constexpr int THREADS = CONSUMER_THREADS + 32;   // + the producer warp
constexpr int MAX_M = 16;
constexpr int MAX_P = 8;
#ifndef MROW_MAX_KS
#define MROW_MAX_KS 2048
#endif
#ifndef MROW_PIPE
#define MROW_PIPE 8
#endif
#ifndef MROW_MINB
#define MROW_MINB 1
#endif
#ifndef MROW_L2PF
#define MROW_L2PF 0      // k-rows of the item's weights past the ring prefetched into L2 before griddep_wait (0 = off)
#endif
#ifndef MROW_CPASYNC
#define MROW_CPASYNC 0   // 1: the producer's 32 lanes copy each stage with 16-byte cp.async (smoe's path) instead of one TMA bulk copy
#endif
#ifndef MROW_L2HINT
#define MROW_L2HINT 0    // with MROW_CPASYNC: 128 / 256 = the copies ask L2 for 128- / 256-byte DRAM fetches
#endif
constexpr int MAX_KS = MROW_MAX_KS;   // k-slice cap: the transformed input slice lives in shared memory
constexpr int PIPE = MROW_PIPE;       // trellis stages (k-rows of the item) in flight
static_assert(PIPE >= 2 && PIPE % 2 == 0, "MROW_PIPE: even, >= 2 (two k-rows per loop step)");
constexpr int SX_PAD = 8;        // halves of row padding: rows land on different banks for the A-fragment loads

struct Problem
{
    const uint16_t* trellis;     // [k/16][n/16][16K]
    const half* suh;             // [k]
    const half* svh;             // [n]
    int k, n;
    int x_col, out_col;          // column offsets of this problem's input / output in x / out
    int ks, n_slices, n_blocks;
    int item_start;              // first work item of this problem
    long long ws_off;            // float offset of this problem's k-slice partials ([n_slices][MAX_M][n])
    int ctr_off;                 // offset of this problem's per-output-block counters
};

struct Group
{
    Problem p[MAX_P];
    int n_problems;
    int n_items;
    int ks_max;                  // widest k-slice in the group: the shared-memory input rows are this wide
};

// shared memory per block: the transformed input [rows][ks_max + SX_PAD] halves, then the trellis ring (PIPE stages of
// 8 tiles), which the epilogue's [rows][128] fp32 partial reuses
__host__ __device__ constexpr int smem_sx_bytes(int rows, int ks_max) { return ((rows * (ks_max + SX_PAD) * 2 + 15) / 16) * 16; }
__host__ __device__ constexpr int smem_ring_bytes(int K) { return WARPS * PIPE * 32 * K; }
__host__ __device__ constexpr int smem_bytes(int K, int rows, int ks_max)
{
    return smem_sx_bytes(rows, ks_max) + (smem_ring_bytes(K) > rows * NB * 4 ? smem_ring_bytes(K) : rows * NB * 4);
}

// mbarrier / TMA helpers, tile decode, mma and Hadamard: exl3_common.cuh
__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;\n" :: "n"(CONSUMER_THREADS) : "memory"); }

// one stage (SB bytes, contiguous) by the producer warp's 32 lanes in 16-byte cp.async pieces; the stage's mbarrier
// counts one arrival per lane, each fired when that lane's copies have landed (MROW_CPASYNC)
template <int SB>
__device__ __forceinline__ void stage_cp_async(unsigned char* dst, const unsigned char* src, uint64_t* bar, int lane)
{
    const uint32_t d = smem_u32(dst);
    for (int i = lane; i < SB / 16; i += 32)
    {
#if MROW_L2HINT == 256
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(d + i * 16), "l"(src + i * 16) : "memory");
#elif MROW_L2HINT == 128
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(d + i * 16), "l"(src + i * 16) : "memory");
#else
        cp_async16(d + i * 16, src + i * 16);
#endif
    }
    cp_async_arrive_noinc(smem_u32(bar));
}

// M16: 9..16 rows (two 8-row halves of the m16 fragment); otherwise 1..8 rows, the upper half of A is zero registers
template <int K, bool M16>
__global__ void __launch_bounds__(THREADS, MROW_MINB) mrow_kernel
(
    const __nv_bfloat16* __restrict__ x, int x_stride,
    __nv_bfloat16* __restrict__ out, int out_stride,
    int m, float* __restrict__ ws, int* __restrict__ ctr, const __grid_constant__ Group g
)
{
    constexpr int RX = M16 ? 16 : 8;     // input rows held in shared memory
    constexpr int TB = 32 * K;           // bytes per 16x16 trellis tile
    constexpr int SB = WARPS * TB;       // bytes per stage: the item's 8 tiles of one k-row, contiguous in the trellis
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[PIPE];
    __shared__ __align__(8) uint64_t empty[PIPE];
    __shared__ int s_last;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    int pi = 0;
    #pragma unroll 1
    while (pi + 1 < g.n_problems && (int) blockIdx.x >= g.p[pi + 1].item_start) ++pi;
    const Problem& P = g.p[pi];
    const int local = blockIdx.x - P.item_start;
    const int nb = local % P.n_blocks;
    const int sl = local / P.n_blocks;
    const int k0 = sl * P.ks;
    const int n0 = nb * NB;
    const int kts = P.ks / 16;
    half* sx = reinterpret_cast<half*>(smem);
    unsigned char* ring = smem + smem_sx_bytes(RX, g.ks_max);
    float* sy = reinterpret_cast<float*>(ring);                              // [RX][NB], once every stage is consumed

    if (threadIdx.x == 0)
    {
        for (int s = 0; s < PIPE; ++s) { mbar_init(&full[s], MROW_CPASYNC ? 32 : 1); mbar_init(&empty[s], WARPS); }
        mbar_fence_init();
    }
    __syncthreads();

    // producer warp: one bulk copy per k-row of the slice, issued as soon as its stage is free. The first PIPE stages
    // are weights only (static), so they go out before griddep_wait: under PDL they stream while the previous kernel
    // finishes.
    if (warp == WARPS)
    {
        const int tiles_n = P.n / 16;
        const size_t row_bytes = (size_t) tiles_n * TB;
        const unsigned char* src = reinterpret_cast<const unsigned char*>(P.trellis)
                                   + ((size_t) (k0 / 16) * tiles_n + n0 / 16) * TB;
        const int pre = kts < PIPE ? kts : PIPE;
#if MROW_CPASYNC
        for (int kk = 0; kk < pre; ++kk) stage_cp_async<SB>(ring + kk * SB, src + kk * row_bytes, &full[kk], lane);
        griddep_wait();
        griddep_launch();
        #pragma unroll 1
        for (int kk = pre; kk < kts; ++kk)
        {
            const int s = kk % PIPE;
            mbar_wait(&empty[s], ((kk / PIPE) - 1) & 1);
            stage_cp_async<SB>(ring + s * SB, src + kk * row_bytes, &full[s], lane);
        }
        return;
#endif
        if (lane == 0)
        {
            for (int kk = 0; kk < pre; ++kk)
            {
                mbar_expect_tx(&full[kk], SB);
                bulk_load(ring + kk * SB, src + kk * row_bytes, SB, &full[kk]);
            }
#if MROW_L2PF > 0
            // the next k-rows into L2 as well, so the stream after the wait starts from L2 (TensorFold's kernels
            // prefetch their weights to L2 before griddepcontrol.wait)
            const int pf_end = kts < pre + MROW_L2PF ? kts : pre + MROW_L2PF;
            for (int kk = pre; kk < pf_end; ++kk) l2_prefetch(src + kk * row_bytes, SB);
#endif
        }
        griddep_wait();
        griddep_launch();
        if (lane == 0)
        {
            #pragma unroll 1
            for (int kk = pre; kk < kts; ++kk)
            {
                const int s = kk % PIPE;
                mbar_wait(&empty[s], ((kk / PIPE) - 1) & 1);
                mbar_expect_tx(&full[s], SB);
                bulk_load(ring + s * SB, src + kk * row_bytes, SB, &full[s]);
            }
        }
        return;
    }
    griddep_wait();                      // x comes from the previous kernel
    griddep_launch();

    // 1. input slice: x[:, x_col + k0 : +ks] * suh, Hadamard per 128 block, fp16 into shared memory; rows >= m are zero
    const int kblocks = P.ks / 128;
    const int sxw = g.ks_max + SX_PAD;                                       // halves per shared input row
    const int ntask = m * kblocks;                                           // (row, 128 block) pairs
    constexpr int XB = 4;                                                    // tasks per warp with loads in flight
    for (int u0 = warp; u0 < ntask; u0 += XB * WARPS)
    {
        uint2 xv[XB], hv[XB];
        #pragma unroll
        for (int i = 0; i < XB; ++i)
        {
            const int u = u0 + i * WARPS;
            if (u >= ntask) break;
            const int r = u / kblocks;
            const int c = k0 + (u % kblocks) * 128 + lane * 4;
            xv[i] = *reinterpret_cast<const uint2*>(x + (size_t) r * x_stride + P.x_col + c);
            hv[i] = *reinterpret_cast<const uint2*>(P.suh + c);
        }
        #pragma unroll
        for (int i = 0; i < XB; ++i)
        {
            const int u = u0 + i * WARPS;
            if (u >= ntask) break;
            const int r = u / kblocks;
            const int bb = u % kblocks;
            const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv[i].x));
            const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv[i].y));
            const float2 h01 = __half22float2(*reinterpret_cast<const half2*>(&hv[i].x));
            const float2 h23 = __half22float2(*reinterpret_cast<const half2*>(&hv[i].y));
            float v[4] = {x01.x * h01.x, x01.y * h01.y, x23.x * h23.x, x23.y * h23.y};
            had128(v, lane);
            const half2 o01 = __floats2half2_rn(v[0] * RSQRT128, v[1] * RSQRT128);
            const half2 o23 = __floats2half2_rn(v[2] * RSQRT128, v[3] * RSQRT128);
            uint2 ov;
            ov.x = *reinterpret_cast<const uint32_t*>(&o01);
            ov.y = *reinterpret_cast<const uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(sx + r * sxw + bb * 128 + lane * 4) = ov;
        }
    }
    for (int i = threadIdx.x; i < (RX - m) * P.ks; i += CONSUMER_THREADS)
        sx[(m + i / P.ks) * sxw + i % P.ks] = __float2half_rn(0.f);
    consumer_sync();

    // 2. this warp's 16 output columns over the k-slice: decode + mma, two k-rows per step
    const int gq = lane / 4;
    const int tq = lane % 4;
    const LaneDq<K> cfg(lane);
    float c0[4] = {0.f, 0.f, 0.f, 0.f};
    float c1[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t* xr0 = reinterpret_cast<const uint32_t*>(sx + gq * sxw);
    const uint32_t* xr1 = reinterpret_cast<const uint32_t*>(sx + (gq + 8) * sxw);
    const unsigned char* wtile = ring + warp * TB;
    #pragma unroll 1
    for (int kk = 0; kk < kts; kk += 2)
    {
        const int s0 = kk % PIPE, s1 = s0 + 1;
        const uint32_t par = (kk / PIPE) & 1;
        mbar_wait(&full[s0], par);
        mbar_wait(&full[s1], par);
        half2 f0[2], f1[2], e0[2], e1[2];
        decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s0 * SB), cfg, f0, f1);
        decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s1 * SB), cfg, e0, e1);
        uint32_t a[4], b[4];
        a[0] = xr0[kk * 8 + tq];     a[2] = xr0[kk * 8 + tq + 4];
        b[0] = xr0[kk * 8 + 8 + tq]; b[2] = xr0[kk * 8 + 8 + tq + 4];
        if constexpr (M16)
        {
            a[1] = xr1[kk * 8 + tq];     a[3] = xr1[kk * 8 + tq + 4];
            b[1] = xr1[kk * 8 + 8 + tq]; b[3] = xr1[kk * 8 + 8 + tq + 4];
        }
        else { a[1] = 0u; a[3] = 0u; b[1] = 0u; b[3] = 0u; }
        mma16816(c0, a, f0);
        mma16816(c1, a, f1);
        mma16816(c0, b, e0);
        mma16816(c1, b, e1);
        __syncwarp();                    // every lane has read both stages before they are handed back
        if (lane == 0) { mbar_arrive(&empty[s0]); mbar_arrive(&empty[s1]); }
    }

    // 3. output Hadamard + svh on the partial, then write (one slice) or keep for the k-split sum
    consumer_sync();                     // sy reuses the ring: every warp is past its last stage
    const int col = warp * 16 + 2 * tq;
    sy[gq * NB + col] = c0[0];     sy[gq * NB + col + 1] = c0[1];
    sy[gq * NB + col + 8] = c1[0]; sy[gq * NB + col + 9] = c1[1];
    if constexpr (M16)
    {
        sy[(gq + 8) * NB + col] = c0[2];     sy[(gq + 8) * NB + col + 1] = c0[3];
        sy[(gq + 8) * NB + col + 8] = c1[2]; sy[(gq + 8) * NB + col + 9] = c1[3];
    }
    consumer_sync();

    const int oc = n0 + lane * 4;
    for (int r = warp; r < m; r += WARPS)
    {
        float v[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = sy[r * NB + lane * 4 + j];
        had128(v, lane);
        const uint2 sv = *reinterpret_cast<const uint2*>(P.svh + oc);
        const float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sv.x));
        const float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sv.y));
        const float y0 = v[0] * RSQRT128 * s01.x, y1 = v[1] * RSQRT128 * s01.y;
        const float y2 = v[2] * RSQRT128 * s23.x, y3 = v[3] * RSQRT128 * s23.y;
        if (P.n_slices == 1)
        {
            const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0, y1), o23 = __floats2bfloat162_rn(y2, y3);
            uint2 ov;
            ov.x = *reinterpret_cast<const uint32_t*>(&o01);
            ov.y = *reinterpret_cast<const uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(out + (size_t) r * out_stride + P.out_col + oc) = ov;
        }
        else
            *reinterpret_cast<float4*>(ws + P.ws_off + ((size_t) sl * m + r) * P.n + oc) = make_float4(y0, y1, y2, y3);
    }
    if (P.n_slices == 1) return;

    // 4. k-split: the last block of this output block sums the slices in order (no waiting on anyone)
    __threadfence();
    consumer_sync();
    if (threadIdx.x == 0)
    {
        const int prev = atomicAdd(&ctr[P.ctr_off + nb], 1);
        s_last = prev == P.n_slices - 1;
    }
    consumer_sync();
    if (!s_last) return;
    __threadfence();
    for (int r = warp; r < m; r += WARPS)
    {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int q = 0; q < P.n_slices; ++q)
        {
            const float4 t = __ldcg(reinterpret_cast<const float4*>(ws + P.ws_off + ((size_t) q * m + r) * P.n + oc));
            acc.x += t.x; acc.y += t.y; acc.z += t.z; acc.w += t.w;
        }
        const __nv_bfloat162 o01 = __floats2bfloat162_rn(acc.x, acc.y), o23 = __floats2bfloat162_rn(acc.z, acc.w);
        uint2 ov;
        ov.x = *reinterpret_cast<const uint32_t*>(&o01);
        ov.y = *reinterpret_cast<const uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(out + (size_t) r * out_stride + P.out_col + oc) = ov;
    }
    if (threadIdx.x == 0) ctr[P.ctr_off + nb] = 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// dsv41 0042b: 17..32 rows, mrow32_kernel on plan Handle::g32 (variants: head of the file). The 1..16-row kernel above
// is the original, untouched. LEAN and RING leave rows m..31 of the shared input unwritten: row r of an m16n8k16 result
// reads row r of A only, and the epilogue stores rows < m only, so whatever those rows hold cannot reach the output.

constexpr int MAX_M32 = 32;
constexpr int V_DRAFT = 0, V_LEAN = 1, V_RING = 2, V_RING1 = 3, N_V32 = 4;
#ifndef MROW32_VARIANTS
#define MROW32_VARIANTS 0xF  // bit v: variant v is compiled (e.g. -DMROW32_VARIANTS=4: RING only, a shorter JIT build)
#endif
#ifndef MROW32_MINB
#define MROW32_MINB 2        // RING: blocks per SM the registers are budgeted for (2: <= 112 per thread)
#endif
#ifndef MROW32_PIPE
#define MROW32_PIPE 0        // trellis stages, 0 = per variant: DRAFT MROW_PIPE, LEAN what its epilogue buffer holds
#endif                       // (12 at K5), RING 8, RING1 16
constexpr int XRW = NB + SX_PAD;              // RING: halves per input row of a stage (conflict-free A loads)
constexpr int XR_STAGE = MAX_M32 * XRW;       // RING: halves per stage, 32 rows x one 128-wide k block
constexpr int XR_BYTES = 2 * XR_STAGE * 2;    // RING: two stages
constexpr int SYW = NB + 8;                   // LEAN / RING: floats per epilogue row (conflict-free float2 stores)

__host__ __device__ constexpr int max_i(int a, int b) { return a > b ? a : b; }
__host__ __device__ constexpr int sb_bytes(int K) { return WARPS * 32 * K; }                        // one trellis stage
__host__ __device__ constexpr int sy32_bytes(int V) { return MAX_M32 * (V == V_DRAFT ? NB : SYW) * 4; }  // [32][..] fp32
__host__ __device__ constexpr int pipe_fit(int K, int V)
{
    return ((sy32_bytes(V) / sb_bytes(K)) & ~1) > 16 ? 16 : ((sy32_bytes(V) / sb_bytes(K)) & ~1);
}
__host__ __device__ constexpr int pipe32(int K, int V)
{
    return MROW32_PIPE > 0 ? MROW32_PIPE
         : V == V_LEAN ? max_i(pipe_fit(K, V), PIPE)
         : V == V_RING1 ? 16 : (V == V_RING ? 8 : PIPE);
}
__host__ __device__ constexpr int minb32(int V) { return V == V_RING ? MROW32_MINB : (V == V_RING1 ? 1 : MROW_MINB); }
template <int V> struct Minb32 { static constexpr int value = minb32(V); };   // __launch_bounds__ operand
// dynamic shared memory: DRAFT / LEAN the item's [32][ks_max + pad] input halves, then the trellis ring, which the
// epilogue's fp32 partial reuses; RING the two input stages and the trellis ring, all of which the partial reuses
__host__ __device__ constexpr int smem32_bytes(int K, int V, int ks_max)
{
    return V >= V_RING ? max_i(XR_BYTES + pipe32(K, V) * sb_bytes(K), sy32_bytes(V))
                       : smem_sx_bytes(MAX_M32, ks_max) + max_i(pipe32(K, V) * sb_bytes(K), sy32_bytes(V));
}

// one 128-wide row block of the input: x * suh, Hadamard, / sqrt(128), fp16 into shared memory, 4 per lane (the
// arithmetic of mrow_kernel's step 1)
__device__ __forceinline__ void xform_store(half* dst, const uint2& xv, const uint2& hv, int lane)
{
    const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
    const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
    const float2 h01 = __half22float2(*reinterpret_cast<const half2*>(&hv.x));
    const float2 h23 = __half22float2(*reinterpret_cast<const half2*>(&hv.y));
    float v[4] = {x01.x * h01.x, x01.y * h01.y, x23.x * h23.x, x23.y * h23.y};
    had128(v, lane);
    const half2 o01 = __floats2half2_rn(v[0] * RSQRT128, v[1] * RSQRT128);
    const half2 o23 = __floats2half2_rn(v[2] * RSQRT128, v[3] * RSQRT128);
    uint2 ov;
    ov.x = *reinterpret_cast<const uint32_t*>(&o01);
    ov.y = *reinterpret_cast<const uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(dst + lane * 4) = ov;
}

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* dst, float y0, float y1, float y2, float y3)
{
    const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0, y1), o23 = __floats2bfloat162_rn(y2, y3);
    uint2 ov;
    ov.x = *reinterpret_cast<const uint32_t*>(&o01);
    ov.y = *reinterpret_cast<const uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(dst) = ov;
}

// two k-rows into rows 0..15 (c0, c1: columns 0..7 / 8..15 of the warp's tile) and rows 16..31 (c2, c3): A at word w
// (first k-row) and w + 8 (second) of the four row pointers, B the decoded fragments f (first k-row) and e (second).
// Every accumulator adds its k-rows in order, so a row's sum does not depend on the other rows.
__device__ __forceinline__ void mma32(float (&c0)[4], float (&c1)[4], float (&c2)[4], float (&c3)[4],
                                      const uint32_t* xr0, const uint32_t* xr1, const uint32_t* xr2,
                                      const uint32_t* xr3, int w, int tq, const half2 (&f0)[2], const half2 (&f1)[2],
                                      const half2 (&e0)[2], const half2 (&e1)[2])
{
    uint32_t a[4], b[4], a2[4], b2[4];
    a[0] = xr0[w + tq];      a[2] = xr0[w + tq + 4];      a[1] = xr1[w + tq];      a[3] = xr1[w + tq + 4];
    b[0] = xr0[w + 8 + tq];  b[2] = xr0[w + 8 + tq + 4];  b[1] = xr1[w + 8 + tq];  b[3] = xr1[w + 8 + tq + 4];
    a2[0] = xr2[w + tq];     a2[2] = xr2[w + tq + 4];     a2[1] = xr3[w + tq];     a2[3] = xr3[w + tq + 4];
    b2[0] = xr2[w + 8 + tq]; b2[2] = xr2[w + 8 + tq + 4]; b2[1] = xr3[w + 8 + tq]; b2[3] = xr3[w + 8 + tq + 4];
    mma16816(c0, a, f0);  mma16816(c2, a2, f0);
    mma16816(c1, a, f1);  mma16816(c3, a2, f1);
    mma16816(c0, b, e0);  mma16816(c2, b2, e0);
    mma16816(c1, b, e1);  mma16816(c3, b2, e1);
}

template <int K, int V>
__global__ void __launch_bounds__(THREADS, Minb32<V>::value) mrow32_kernel
(
    const __nv_bfloat16* __restrict__ x, int x_stride,
    __nv_bfloat16* __restrict__ out, int out_stride,
    int m, float* __restrict__ ws, int* __restrict__ ctr, const __grid_constant__ Group g
)
{
    constexpr bool RING = V >= V_RING;
    constexpr int RX = MAX_M32;          // input rows held in shared memory (DRAFT / LEAN; RING per stage)
    constexpr int TB = 32 * K;           // bytes per 16x16 trellis tile
    constexpr int SB = WARPS * TB;       // bytes per stage: the item's 8 tiles of one k-row, contiguous in the trellis
    constexpr int PP = pipe32(K, V);     // trellis stages
    constexpr int YW = V == V_DRAFT ? NB : SYW;    // floats per epilogue row
    static_assert(PP >= 2 && PP % 2 == 0, "MROW32_PIPE: even, >= 2 (two k-rows per loop step)");
    static_assert(!RING || PP % 8 == 0, "MROW32_PIPE: RING needs a multiple of 8 (a 128-wide k block = 8 stages)");
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[PP];
    __shared__ __align__(8) uint64_t empty[PP];
    __shared__ int s_last;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    int pi = 0;
    #pragma unroll 1
    while (pi + 1 < g.n_problems && (int) blockIdx.x >= g.p[pi + 1].item_start) ++pi;
    const Problem& P = g.p[pi];
    const int local = blockIdx.x - P.item_start;
    const int nb = local % P.n_blocks;
    const int sl = local / P.n_blocks;
    const int k0 = sl * P.ks;
    const int n0 = nb * NB;
    const int kts = P.ks / 16;
    const int kblocks = P.ks / 128;
    half* sx = reinterpret_cast<half*>(smem);
    unsigned char* ring = smem + (RING ? XR_BYTES : smem_sx_bytes(RX, g.ks_max));
    float* sy = reinterpret_cast<float*>(RING ? smem : ring);      // [RX][YW], once every stage (and input) is consumed

    if (threadIdx.x == 0)
    {
        for (int s = 0; s < PP; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], WARPS); }
        mbar_fence_init();
    }
    __syncthreads();

    // producer warp, as in mrow_kernel (TMA path): one bulk copy per k-row of the slice, the first PP before griddep_wait
    if (warp == WARPS)
    {
        const int tiles_n = P.n / 16;
        const size_t row_bytes = (size_t) tiles_n * TB;
        const unsigned char* src = reinterpret_cast<const unsigned char*>(P.trellis)
                                   + ((size_t) (k0 / 16) * tiles_n + n0 / 16) * TB;
        const int pre = kts < PP ? kts : PP;
        if (lane == 0)
        {
            for (int kk = 0; kk < pre; ++kk)
            {
                mbar_expect_tx(&full[kk], SB);
                bulk_load(ring + kk * SB, src + kk * row_bytes, SB, &full[kk]);
            }
#if MROW_L2PF > 0
            const int pf_end = kts < pre + MROW_L2PF ? kts : pre + MROW_L2PF;
            for (int kk = pre; kk < pf_end; ++kk) l2_prefetch(src + kk * row_bytes, SB);
#endif
        }
        griddep_wait();
        griddep_launch();
        if (lane == 0)
        {
            #pragma unroll 1
            for (int kk = pre; kk < kts; ++kk)
            {
                const int s = kk % PP;
                mbar_wait(&empty[s], ((kk / PP) - 1) & 1);
                mbar_expect_tx(&full[s], SB);
                bulk_load(ring + s * SB, src + kk * row_bytes, SB, &full[s]);
            }
        }
        return;
    }
    griddep_wait();                      // x comes from the previous kernel
    griddep_launch();

    const int gq = lane / 4;
    const int tq = lane % 4;
    const LaneDq<K> cfg(lane);
    float c0[4] = {0.f, 0.f, 0.f, 0.f};
    float c1[4] = {0.f, 0.f, 0.f, 0.f};
    float c2[4] = {0.f, 0.f, 0.f, 0.f};
    float c3[4] = {0.f, 0.f, 0.f, 0.f};
    const unsigned char* wtile = ring + warp * TB;

    if constexpr (!RING)
    {
        // 1. the item's input slice x[:, x_col + k0 : +ks] * suh, Hadamard per 128 block, fp16 into shared memory
        const int sxw = g.ks_max + SX_PAD;                                   // halves per shared input row
        const int ntask = m * kblocks;                                       // (row, 128 block) pairs
        constexpr int XB = V == V_DRAFT ? 4 : 8;                             // tasks per warp with loads in flight
        for (int u0 = warp; u0 < ntask; u0 += XB * WARPS)
        {
            uint2 xv[XB], hv[XB];
            #pragma unroll
            for (int i = 0; i < XB; ++i)
            {
                const int u = u0 + i * WARPS;
                if (u >= ntask) break;
                const int r = u / kblocks;
                const int c = k0 + (u % kblocks) * 128 + lane * 4;
                xv[i] = *reinterpret_cast<const uint2*>(x + (size_t) r * x_stride + P.x_col + c);
                hv[i] = *reinterpret_cast<const uint2*>(P.suh + c);
            }
            #pragma unroll
            for (int i = 0; i < XB; ++i)
            {
                const int u = u0 + i * WARPS;
                if (u >= ntask) break;
                xform_store(sx + (u / kblocks) * sxw + (u % kblocks) * 128, xv[i], hv[i], lane);
            }
        }
        if constexpr (V == V_DRAFT)          // 0042: rows m..31 zeroed, one half per thread and step (LEAN: dropped)
            for (int i = threadIdx.x; i < (RX - m) * P.ks; i += CONSUMER_THREADS)
                sx[(m + i / P.ks) * sxw + i % P.ks] = __float2half_rn(0.f);
        consumer_sync();

        // 2. this warp's 16 output columns over the k-slice: decode + mma, two k-rows per step, both m16 tiles
        const uint32_t* xr0 = reinterpret_cast<const uint32_t*>(sx + gq * sxw);
        const uint32_t* xr1 = reinterpret_cast<const uint32_t*>(sx + (gq + 8) * sxw);
        const uint32_t* xr2 = reinterpret_cast<const uint32_t*>(sx + (gq + 16) * sxw);
        const uint32_t* xr3 = reinterpret_cast<const uint32_t*>(sx + (gq + 24) * sxw);
        #pragma unroll 1
        for (int kk = 0; kk < kts; kk += 2)
        {
            const int s0 = kk % PP, s1 = s0 + 1;
            const uint32_t par = (kk / PP) & 1;
            mbar_wait(&full[s0], par);
            mbar_wait(&full[s1], par);
            half2 f0[2], f1[2], e0[2], e1[2];
            decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s0 * SB), cfg, f0, f1);
            decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s1 * SB), cfg, e0, e1);
            mma32(c0, c1, c2, c3, xr0, xr1, xr2, xr3, kk * 8, tq, f0, f1, e0, e1);
            __syncwarp();                    // every lane has read both stages before they are handed back
            if (lane == 0) { mbar_arrive(&empty[s0]); mbar_arrive(&empty[s1]); }
        }
    }
    else
    {
        // 1 + 2, RING: the input of k block bb + 1 (this warp's rows warp + 8j < m, and its suh) is loaded before block
        // bb's mma and transformed after it into the other stage; one consumer barrier per block publishes it and
        // frees the stage block bb read. So the weight stream waits for one block's transform per item, not the
        // slice's, and the shared memory does not grow with the k-slice.
        uint2 xv[4], hv;
        auto xr_load = [&](int bb)
        {
            const int c = k0 + bb * 128 + lane * 4;
            hv = *reinterpret_cast<const uint2*>(P.suh + c);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (warp + j * WARPS < m)
                    xv[j] = *reinterpret_cast<const uint2*>(x + (size_t) (warp + j * WARPS) * x_stride + P.x_col + c);
        };
        auto xr_store = [&](int bb)
        {
            half* st = sx + (bb & 1) * XR_STAGE;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (warp + j * WARPS < m) xform_store(st + (warp + j * WARPS) * XRW, xv[j], hv, lane);
        };
        xr_load(0);
        xr_store(0);
        consumer_sync();
        #pragma unroll 1
        for (int bb = 0; bb < kblocks; ++bb)
        {
            const bool more = bb + 1 < kblocks;
            if (more) xr_load(bb + 1);
            const half* st = sx + (bb & 1) * XR_STAGE;
            const uint32_t* xr0 = reinterpret_cast<const uint32_t*>(st + gq * XRW);
            const uint32_t* xr1 = reinterpret_cast<const uint32_t*>(st + (gq + 8) * XRW);
            const uint32_t* xr2 = reinterpret_cast<const uint32_t*>(st + (gq + 16) * XRW);
            const uint32_t* xr3 = reinterpret_cast<const uint32_t*>(st + (gq + 24) * XRW);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const int kk = bb * 8 + 2 * j;
                const int s0 = kk % PP, s1 = s0 + 1;
                const uint32_t par = (kk / PP) & 1;
                mbar_wait(&full[s0], par);
                mbar_wait(&full[s1], par);
                half2 f0[2], f1[2], e0[2], e1[2];
                decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s0 * SB), cfg, f0, f1);
                decode_tile<K>(reinterpret_cast<const uint32_t*>(wtile + s1 * SB), cfg, e0, e1);
                mma32(c0, c1, c2, c3, xr0, xr1, xr2, xr3, 16 * j, tq, f0, f1, e0, e1);
                __syncwarp();
                if (lane == 0) { mbar_arrive(&empty[s0]); mbar_arrive(&empty[s1]); }
            }
            if (more) xr_store(bb + 1);
            consumer_sync();                 // block bb + 1 is in place for every warp; stage bb & 1 is free
        }
    }

    // 3. output Hadamard + svh on the partial, then write (one slice) or keep for the k-split sum
    consumer_sync();                     // sy reuses the shared memory: every warp is past its last stage and input
    const int col = warp * 16 + 2 * tq;
    if constexpr (V == V_DRAFT)
    {
        sy[gq * YW + col] = c0[0];            sy[gq * YW + col + 1] = c0[1];
        sy[gq * YW + col + 8] = c1[0];        sy[gq * YW + col + 9] = c1[1];
        sy[(gq + 8) * YW + col] = c0[2];      sy[(gq + 8) * YW + col + 1] = c0[3];
        sy[(gq + 8) * YW + col + 8] = c1[2];  sy[(gq + 8) * YW + col + 9] = c1[3];
        sy[(gq + 16) * YW + col] = c2[0];     sy[(gq + 16) * YW + col + 1] = c2[1];
        sy[(gq + 16) * YW + col + 8] = c3[0]; sy[(gq + 16) * YW + col + 9] = c3[1];
        sy[(gq + 24) * YW + col] = c2[2];     sy[(gq + 24) * YW + col + 1] = c2[3];
        sy[(gq + 24) * YW + col + 8] = c3[2]; sy[(gq + 24) * YW + col + 9] = c3[3];
    }
    else
    {
        *reinterpret_cast<float2*>(sy + gq * YW + col) = make_float2(c0[0], c0[1]);
        *reinterpret_cast<float2*>(sy + gq * YW + col + 8) = make_float2(c1[0], c1[1]);
        *reinterpret_cast<float2*>(sy + (gq + 8) * YW + col) = make_float2(c0[2], c0[3]);
        *reinterpret_cast<float2*>(sy + (gq + 8) * YW + col + 8) = make_float2(c1[2], c1[3]);
        *reinterpret_cast<float2*>(sy + (gq + 16) * YW + col) = make_float2(c2[0], c2[1]);
        *reinterpret_cast<float2*>(sy + (gq + 16) * YW + col + 8) = make_float2(c3[0], c3[1]);
        *reinterpret_cast<float2*>(sy + (gq + 24) * YW + col) = make_float2(c2[2], c2[3]);
        *reinterpret_cast<float2*>(sy + (gq + 24) * YW + col + 8) = make_float2(c3[2], c3[3]);
    }
    consumer_sync();

    const int oc = n0 + lane * 4;
    for (int r = warp; r < m; r += WARPS)
    {
        float v[4];
        if constexpr (V == V_DRAFT)
        {
            #pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = sy[r * YW + lane * 4 + j];
        }
        else
        {
            const float4 t = *reinterpret_cast<const float4*>(sy + r * YW + lane * 4);
            v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
        }
        had128(v, lane);
        const uint2 sv = *reinterpret_cast<const uint2*>(P.svh + oc);
        const float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sv.x));
        const float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sv.y));
        const float y0 = v[0] * RSQRT128 * s01.x, y1 = v[1] * RSQRT128 * s01.y;
        const float y2 = v[2] * RSQRT128 * s23.x, y3 = v[3] * RSQRT128 * s23.y;
        if (P.n_slices == 1)
            store_bf16x4(out + (size_t) r * out_stride + P.out_col + oc, y0, y1, y2, y3);
        else
            *reinterpret_cast<float4*>(ws + P.ws_off + ((size_t) sl * m + r) * P.n + oc) = make_float4(y0, y1, y2, y3);
    }
    if (P.n_slices == 1) return;

    // 4. k-split: the last block of this output block sums the slices in order (no waiting on anyone)
    __threadfence();
    consumer_sync();
    if (threadIdx.x == 0)
    {
        const int prev = atomicAdd(&ctr[P.ctr_off + nb], 1);
        s_last = prev == P.n_slices - 1;
    }
    consumer_sync();
    if (!s_last) return;
    __threadfence();
    if constexpr (V == V_DRAFT)
    {
        for (int r = warp; r < m; r += WARPS)
        {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int q = 0; q < P.n_slices; ++q)
            {
                const float4 t = __ldcg(reinterpret_cast<const float4*>(ws + P.ws_off + ((size_t) q * m + r) * P.n + oc));
                acc.x += t.x; acc.y += t.y; acc.z += t.z; acc.w += t.w;
            }
            store_bf16x4(out + (size_t) r * out_stride + P.out_col + oc, acc.x, acc.y, acc.z, acc.w);
        }
    }
    else
    {
        // two rows per warp at a time, the slices still summed in order; the loads run ahead of the adds
        const size_t qs = (size_t) m * P.n;                                   // floats from one slice to the next
        for (int r0 = warp; r0 < m; r0 += 2 * WARPS)
        {
            const int r1 = r0 + WARPS < m ? r0 + WARPS : r0;
            const float* w0 = ws + P.ws_off + (size_t) r0 * P.n + oc;
            const float* w1 = ws + P.ws_off + (size_t) r1 * P.n + oc;
            float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = a0;
            #pragma unroll 4
            for (int q = 0; q < P.n_slices; ++q)
            {
                const float4 t0 = __ldcg(reinterpret_cast<const float4*>(w0 + q * qs));
                const float4 t1 = __ldcg(reinterpret_cast<const float4*>(w1 + q * qs));
                a0.x += t0.x; a0.y += t0.y; a0.z += t0.z; a0.w += t0.w;
                a1.x += t1.x; a1.y += t1.y; a1.z += t1.z; a1.w += t1.w;
            }
            store_bf16x4(out + (size_t) r0 * out_stride + P.out_col + oc, a0.x, a0.y, a0.z, a0.w);
            if (r1 != r0) store_bf16x4(out + (size_t) r1 * out_stride + P.out_col + oc, a1.x, a1.y, a1.z, a1.w);
        }
    }
    if (threadIdx.x == 0) ctr[P.ctr_off + nb] = 0;
}

struct Handle
{
    Group g;
    int K;
    at::Tensor ws, ctr;
    std::vector<at::Tensor> keep;   // keep the weights alive while registered
    // dsv41 0042b: the 17..32-row plan (own k-slices, workspace, counters), its kernel variant and shared memory
    Group g32;
    at::Tensor ws32, ctr32;
    bool has32 = false;
    int v32 = V_RING, smem32 = 0, min32 = MAX_M + 1;
};

std::vector<Handle>& registry()
{
    static std::vector<Handle> r;
    return r;
}

int pick_ks(int k, int n_blocks, int want_items, int max_ks = MAX_KS)
{
    int best = 128;
    for (int ks = 128; ks <= std::min(k, max_ks); ks += 128)
    {
        if (k % ks) continue;
        if ((long long) n_blocks * (k / ks) >= want_items) best = ks;   // largest slice that still fills the GPU
    }
    if ((long long) n_blocks * (k / best) < want_items)                  // nothing fills it: the most slices we can
        for (int ks = 128; ks <= std::min(k, max_ks); ks += 128)
            if (k % ks == 0) { best = ks; break; }
    return best;
}

template <int K, bool M16>
void launch(const Group& g, const __nv_bfloat16* xp, int xs, __nv_bfloat16* op, int os, int m, float* ws, int* ctr,
            cudaStream_t stream)
{
    const int smem = smem_bytes(K, M16 ? 16 : 8, g.ks_max);
    static int smem_set = 48 * 1024;      // dynamic shared memory past 48 KB needs the opt-in, once per size
    if (smem > smem_set)
    {
        C10_CUDA_CHECK(cudaFuncSetAttribute(mrow_kernel<K, M16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        smem_set = smem;
    }
    launch_k(mrow_kernel<K, M16>, dim3(g.n_items), dim3(THREADS), smem, stream, xp, xs, op, os, m, ws, ctr, g);
}

template <int K, int V>
void launch32_v(const Group& g, const __nv_bfloat16* xp, int xs, __nv_bfloat16* op, int os, int m, float* ws, int* ctr,
                cudaStream_t stream)
{
    const int smem = smem32_bytes(K, V, g.ks_max);
    static int smem_set = 48 * 1024;      // dynamic shared memory past 48 KB needs the opt-in, once per size
    static bool carveout = false;
    if (smem > smem_set)
    {
        C10_CUDA_CHECK(cudaFuncSetAttribute(mrow32_kernel<K, V>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        smem_set = smem;
    }
    if (!carveout)                        // ring: two blocks per SM need the carveout as shared memory, not as L1
    {
        C10_CUDA_CHECK(cudaFuncSetAttribute(mrow32_kernel<K, V>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                            (int) cudaSharedmemCarveoutMaxShared));
        carveout = true;
    }
    launch_k(mrow32_kernel<K, V>, dim3(g.n_items), dim3(THREADS), smem, stream, xp, xs, op, os, m, ws, ctr, g);
}

template <int K>
void launch32(int v, const Group& g, const __nv_bfloat16* xp, int xs, __nv_bfloat16* op, int os, int m, float* ws,
              int* ctr, cudaStream_t stream)
{
    switch (v)
    {
#if MROW32_VARIANTS & 1
        case V_DRAFT: launch32_v<K, V_DRAFT>(g, xp, xs, op, os, m, ws, ctr, stream); break;
#endif
#if MROW32_VARIANTS & 2
        case V_LEAN: launch32_v<K, V_LEAN>(g, xp, xs, op, os, m, ws, ctr, stream); break;
#endif
#if MROW32_VARIANTS & 4
        case V_RING: launch32_v<K, V_RING>(g, xp, xs, op, os, m, ws, ctr, stream); break;
#endif
#if MROW32_VARIANTS & 8
        case V_RING1: launch32_v<K, V_RING1>(g, xp, xs, op, os, m, ws, ctr, stream); break;
#endif
        default: TORCH_CHECK(false, "exl3_mrow: 32-row variant ", v, " not compiled (MROW32_VARIANTS)");
    }
}

int env_int(const char* name, int dflt)
{
    const char* s = std::getenv(name);
    return s && *s ? std::atoi(s) : dflt;
}

int env_variant()                         // EXL3_MROW32_VARIANT: draft / lean / ring / ring1 or 0..3 (default ring)
{
    const char* s = std::getenv("EXL3_MROW32_VARIANT");
    if (!s || !*s) return V_RING;
    static const char* const names[N_V32] = {"draft", "lean", "ring", "ring1"};
    for (int i = 0; i < N_V32; ++i)
        if (std::strcmp(s, names[i]) == 0) return i;
    TORCH_CHECK(s[0] >= '0' && s[0] < '0' + N_V32 && s[1] == 0, "exl3_mrow: EXL3_MROW32_VARIANT=", s,
                " (draft, lean, ring, ring1 or 0..3)");
    return s[0] - '0';
}

// resident blocks per SM (and registers / local bytes per thread) of a kernel at its dynamic shared memory: measurement
template <typename F>
int occ_of(F* kernel, int smem, int* regs, int* local)
{
    if (smem > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, THREADS, smem));
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
    *regs = a.numRegs;
    *local = (int) a.localSizeBytes;
    return n;
}

template <int K, int V>
int occ32_of(int smem, int* regs, int* local)
{
    C10_CUDA_CHECK(cudaFuncSetAttribute(mrow32_kernel<K, V>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        (int) cudaSharedmemCarveoutMaxShared));          // as launch32_v
    return occ_of(mrow32_kernel<K, V>, smem, regs, local);
}

// [blocks/SM 1..8 rows, 9..16 rows, 17..32 rows, registers 9..16, local 9..16, registers 17..32, local 17..32]
template <int K>
std::vector<int64_t> occupancy_k(const Group& g, int v, int smem32)
{
    int r8, l8, r16, l16, r32 = 0, l32 = 0, o32 = 0;
    const int o8 = occ_of(mrow_kernel<K, false>, smem_bytes(K, 8, g.ks_max), &r8, &l8);
    const int o16 = occ_of(mrow_kernel<K, true>, smem_bytes(K, 16, g.ks_max), &r16, &l16);
    switch (v)
    {
#if MROW32_VARIANTS & 1
        case V_DRAFT: o32 = occ32_of<K, V_DRAFT>(smem32, &r32, &l32); break;
#endif
#if MROW32_VARIANTS & 2
        case V_LEAN: o32 = occ32_of<K, V_LEAN>(smem32, &r32, &l32); break;
#endif
#if MROW32_VARIANTS & 4
        case V_RING: o32 = occ32_of<K, V_RING>(smem32, &r32, &l32); break;
#endif
#if MROW32_VARIANTS & 8
        case V_RING1: o32 = occ32_of<K, V_RING1>(smem32, &r32, &l32); break;
#endif
        default: break;
    }
    return {o8, o16, o32, r16, l16, r32, l32};
}

}  // namespace

int64_t mrow_register(std::vector<at::Tensor> trellis, std::vector<at::Tensor> suh, std::vector<at::Tensor> svh,
                      std::vector<int64_t> x_col, std::vector<int64_t> out_col, int64_t K, int64_t want_items)
{
    const int np = (int) trellis.size();
    TORCH_CHECK(np >= 1 && np <= MAX_P, "exl3_mrow: 1..", MAX_P, " linears per group");
    TORCH_CHECK(suh.size() == (size_t) np && svh.size() == (size_t) np && x_col.size() == (size_t) np &&
                out_col.size() == (size_t) np, "exl3_mrow: per-linear lists differ in length");
    TORCH_CHECK(K >= 2 && K <= 6, "exl3_mrow: K ", K, " (2..6 supported)");
    Handle h;
    h.K = (int) K;
    h.g.n_problems = np;
    const int per = (int) ((want_items + np - 1) / np);
    long long ws_floats = 0;
    int ctrs = 0, items = 0;
    for (int i = 0; i < np; ++i)
    {
        const at::Tensor& t = trellis[i];
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kShort && t.dim() == 3 && t.size(2) == 16 * K &&
                    t.is_contiguous(), "exl3_mrow: trellis ", i, " must be contiguous int16 [k/16, n/16, 16K]");
        const int k = (int) t.size(0) * 16, n = (int) t.size(1) * 16;
        TORCH_CHECK(k % 128 == 0 && n % 128 == 0, "exl3_mrow: k and n must be multiples of 128");
        TORCH_CHECK(suh[i].scalar_type() == at::kHalf && suh[i].numel() == k && suh[i].is_contiguous(), "suh ", i);
        TORCH_CHECK(svh[i].scalar_type() == at::kHalf && svh[i].numel() == n && svh[i].is_contiguous(), "svh ", i);
        TORCH_CHECK(x_col[i] % 4 == 0 && out_col[i] % 4 == 0, "exl3_mrow: column offsets must be multiples of 4");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "exl3_mrow: trellis ", i, " not 16-byte aligned");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(suh[i].data_ptr()) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(svh[i].data_ptr()) % 8 == 0, "exl3_mrow: suh / svh ", i, " not 8-byte aligned");
        Problem& p = h.g.p[i];
        p.trellis = reinterpret_cast<const uint16_t*>(t.data_ptr<int16_t>());
        p.suh = reinterpret_cast<const half*>(suh[i].data_ptr<at::Half>());
        p.svh = reinterpret_cast<const half*>(svh[i].data_ptr<at::Half>());
        p.k = k; p.n = n;
        p.x_col = (int) x_col[i]; p.out_col = (int) out_col[i];
        p.n_blocks = n / NB;
        p.ks = pick_ks(k, p.n_blocks, per);
        p.n_slices = k / p.ks;
        p.item_start = items;
        items += p.n_blocks * p.n_slices;
        p.ws_off = ws_floats;
        if (p.n_slices > 1) ws_floats += (long long) p.n_slices * MAX_M * n;
        p.ctr_off = ctrs;
        ctrs += p.n_blocks;
        h.keep.push_back(t); h.keep.push_back(suh[i]); h.keep.push_back(svh[i]);
    }
    h.g.n_items = items;
    h.g.ks_max = 0;
    for (int i = 0; i < np; ++i) h.g.ks_max = std::max(h.g.ks_max, h.g.p[i].ks);
    auto opts = trellis[0].options();
    h.ws = at::empty({std::max<long long>(ws_floats, 1)}, opts.dtype(at::kFloat));
    h.ctr = at::zeros({std::max(ctrs, 1)}, opts.dtype(at::kInt));
    // dsv41 0042b: the 17..32-row plan. Same problems. ring keeps the 16-row plan's k-slices (its shared memory does not
    // grow with them); draft / lean cap them so 32 input rows fit the opt-in shared memory (GB10 ~99 KB: ks <= 1280 at
    // K5), a slice that fits keeps the 16-row choice. EXL3_MROW32_KS / _ITEMS / _MIN_ROWS: tuning (head of the file).
    {
        const int v = env_variant();
        TORCH_CHECK((MROW32_VARIANTS >> v) & 1, "exl3_mrow: 32-row variant ", v, " not compiled (MROW32_VARIANTS)");
        const int ks_env = env_int("EXL3_MROW32_KS", 0);
        const int items_env = env_int("EXL3_MROW32_ITEMS", 0);
        int dev = 0, smax = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&smax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        int cap = MAX_KS;
        if (v < V_RING)
        {
            cap = 128;
            for (int ks = 128; ks <= MAX_KS; ks += 128)
                if (smem32_bytes((int) K, v, ks) + 512 <= smax) cap = ks;   // 512: static mbarriers + s_last
        }
        if (ks_env >= 128) cap = std::min(cap, ks_env / 128 * 128);
        const int per32 = items_env > 0 ? (items_env + np - 1) / np : per;
        h.v32 = v;
        h.min32 = std::max(1, std::min(env_int("EXL3_MROW32_MIN_ROWS", MAX_M + 1), MAX_M + 1));
        h.g32 = h.g;
        long long ws32 = 0;
        int items32 = 0;
        for (int i = 0; i < np; ++i)
        {
            Problem& q = h.g32.p[i];
            if (items_env > 0) q.ks = pick_ks(q.k, q.n_blocks, per32, cap);
            else
            {
                q.ks = std::min(h.g.p[i].ks, cap);
                if (q.k % q.ks) q.ks = pick_ks(q.k, q.n_blocks, per, cap);
            }
            q.n_slices = q.k / q.ks;
            q.item_start = items32;
            items32 += q.n_blocks * q.n_slices;
            q.ws_off = ws32;
            if (q.n_slices > 1) ws32 += (long long) q.n_slices * MAX_M32 * q.n;
        }
        h.g32.n_items = items32;
        h.g32.ks_max = 0;
        for (int i = 0; i < np; ++i) h.g32.ks_max = std::max(h.g32.ks_max, h.g32.p[i].ks);
        h.smem32 = smem32_bytes((int) K, v, h.g32.ks_max);
        h.has32 = h.smem32 + 512 <= smax;
        h.ws32 = at::empty({std::max<long long>(ws32, 1)}, opts.dtype(at::kFloat));
        h.ctr32 = at::zeros({std::max(ctrs, 1)}, opts.dtype(at::kInt));
    }
    registry().push_back(h);
    return (int64_t) registry().size() - 1;
}

std::vector<int64_t> mrow_info(int64_t handle)
{
    const Handle& h = registry().at(handle);
    std::vector<int64_t> r = {h.g.n_items};
    for (int i = 0; i < h.g.n_problems; ++i) r.push_back(h.g.p[i].ks);
    return r;
}

std::vector<int64_t> mrow_info32(int64_t handle)
{
    const Handle& h = registry().at(handle);
    std::vector<int64_t> r = {h.v32, h.has32 ? 1 : 0, h.g32.n_items, h.smem32, h.min32};
    for (int i = 0; i < h.g32.n_problems; ++i) r.push_back(h.g32.p[i].ks);
    return r;
}

std::vector<int64_t> mrow_occupancy(int64_t handle)
{
    const Handle& h = registry().at(handle);
    switch (h.K)
    {
        case 2: return occupancy_k<2>(h.g, h.v32, h.smem32);
        case 3: return occupancy_k<3>(h.g, h.v32, h.smem32);
        case 4: return occupancy_k<4>(h.g, h.v32, h.smem32);
        case 5: return occupancy_k<5>(h.g, h.v32, h.smem32);
        default: return occupancy_k<6>(h.g, h.v32, h.smem32);
    }
}

void mrow_run(int64_t handle, at::Tensor x, at::Tensor out)
{
    Handle& h = registry().at(handle);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "exl3_mrow: bf16 in and out");
    TORCH_CHECK((x.dim() == 2 || x.dim() == 3) && out.dim() == 2 && x.stride(-1) == 1 && out.stride(1) == 1,
                "exl3_mrow: x [m, *] or [m, problems, *] with unit last stride, out row-major [m, *]");
    const int m = (int) x.size(0);
    const bool m32 = h.has32 && m >= h.min32;                   // dsv41 0042b: 17..32 rows on plan g32
    TORCH_CHECK(m >= 1 && m <= (h.has32 ? MAX_M32 : MAX_M) && out.size(0) == m, "exl3_mrow: 1..16 rows (1..32 with plan g32)");
    TORCH_CHECK(x.stride(0) % 4 == 0 && out.stride(0) % 4 == 0, "exl3_mrow: row strides must be multiples of 4");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
                "exl3_mrow: x and out must be 8-byte aligned (4-element vector loads and stores)");
    // 3-D x: problem i reads x[:, i, :] wherever that sits (any row / problem strides), e.g. wo_a's groups straight from
    // the inverse-RoPE kernel's [G, T, D] buffer seen as [T, G, D], instead of a copy to [T, G * D] first. The
    // descriptor goes to the kernel by value, so a per-call copy with these offsets also holds under graph capture.
    Group g = m32 ? h.g32 : h.g;
    if (x.dim() == 3)
    {
        TORCH_CHECK(x.size(1) == h.g.n_problems && x.stride(1) % 4 == 0 && x.stride(1) * (h.g.n_problems - 1) < (1LL << 31),
                    "exl3_mrow: 3-D x needs one [m, *] slice per problem, problem stride a multiple of 4");
        for (int i = 0; i < h.g.n_problems; ++i)
        {
            TORCH_CHECK(h.g.p[i].x_col == i * h.g.p[0].k, "exl3_mrow: 3-D x needs problem i registered at column i * k");
            TORCH_CHECK(h.g.p[i].k <= x.size(2), "exl3_mrow: input narrower than problem ", i);
            g.p[i].x_col = (int) (i * x.stride(1));
        }
    }
    for (int i = 0; i < h.g.n_problems; ++i)
    {
        if (x.dim() == 2) TORCH_CHECK(h.g.p[i].x_col + h.g.p[i].k <= x.size(1), "exl3_mrow: input narrower than problem ", i);
        TORCH_CHECK(h.g.p[i].out_col + h.g.p[i].n <= out.size(1), "exl3_mrow: output narrower than problem ", i);
    }
    const c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();   // c10, not ATen/cuda: the image lacks cusparse.h
    auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    float* ws = (m32 ? h.ws32 : h.ws).data_ptr<float>();
    int* ctr = (m32 ? h.ctr32 : h.ctr).data_ptr<int>();
    const int xs = (int) x.stride(0), os = (int) out.stride(0);
    if (m32)
    {
        switch (h.K)
        {
            case 2: launch32<2>(h.v32, g, xp, xs, op, os, m, ws, ctr, stream); break;
            case 3: launch32<3>(h.v32, g, xp, xs, op, os, m, ws, ctr, stream); break;
            case 4: launch32<4>(h.v32, g, xp, xs, op, os, m, ws, ctr, stream); break;
            case 5: launch32<5>(h.v32, g, xp, xs, op, os, m, ws, ctr, stream); break;
            case 6: launch32<6>(h.v32, g, xp, xs, op, os, m, ws, ctr, stream); break;
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    const bool m16 = m > 8;
    switch (h.K)
    {
        case 2: m16 ? launch<2, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<2, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
        case 3: m16 ? launch<3, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<3, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
        case 4: m16 ? launch<4, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<4, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
        case 5: m16 ? launch<5, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<5, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
        case 6: m16 ? launch<6, true>(g, xp, xs, op, os, m, ws, ctr, stream) : launch<6, false>(g, xp, xs, op, os, m, ws, ctr, stream); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod)
{
    mod.def("register", &mrow_register, "register a group of EXL3 linears; returns a handle");
    mod.def("info", &mrow_info, "[work items, k-slice per linear...]");
    mod.def("info32", &mrow_info32, "17..32-row plan: [variant, usable, work items, shared memory, min rows, k-slice per linear...]");
    mod.def("occupancy", &mrow_occupancy, "[blocks/SM at 1..8, 9..16, 17..32 rows, regs/local 9..16, regs/local 17..32]");
    mod.def("run", &mrow_run, "run a registered group: x bf16 [m, *] (or [m, problems, *]) -> out bf16 [m, *]");
}
