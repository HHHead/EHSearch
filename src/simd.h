// simd.h - SIMD kernels: L2 distance, inner product, popcount, batched rotation, top-k
#pragma once
#include "common.h"
#include <immintrin.h>

namespace eh {

// runtime capability detection -------------------------------------------------
struct CpuCaps {
    bool avx2, avx512f, avx512vpopcnt, aes, vaes, fma, sse42;
};
inline const CpuCaps& caps() {
    static CpuCaps c = [] {
        CpuCaps x{};
        __builtin_cpu_init();
        x.avx2            = __builtin_cpu_supports("avx2");
        x.avx512f         = __builtin_cpu_supports("avx512f");
        x.avx512vpopcnt   = __builtin_cpu_supports("avx512vpopcntdq");
        x.aes             = __builtin_cpu_supports("aes");
        x.vaes            = __builtin_cpu_supports("vaes");
        x.fma             = __builtin_cpu_supports("fma");
        x.sse42           = __builtin_cpu_supports("sse4.2");
        return x;
    }();
    return c;
}
inline void print_cpu_caps() {
    const auto& c = caps();
    printf("[CPU] AVX2=%d AVX512F=%d AVX512VPOPCNTDQ=%d AES-NI=%d VAES=%d FMA=%d\n",
           c.avx2, c.avx512f, c.avx512vpopcnt, c.aes, c.vaes, c.fma);
}

// ------------------------------------------------------------------ L2 squared
// d is assumed to be a multiple of 16 for the AVX512 path (SIFT: 128).
inline float l2_sqr(const float* __restrict a, const float* __restrict b, int d) {
    if (caps().avx512f) {
        __m512 acc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= d; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 dd = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(dd, dd, acc);
        }
        float s = _mm512_reduce_add_ps(acc);
        for (; i < d; i++) { float t = a[i] - b[i]; s += t * t; }
        return s;
    }
    if (caps().avx2) {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= d; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 dd = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(dd, dd, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_hadd_ps(s4, s4); s4 = _mm_hadd_ps(s4, s4);
        float s = _mm_cvtss_f32(s4);
        for (; i < d; i++) { float t = a[i] - b[i]; s += t * t; }
        return s;
    }
    float s = 0;
    for (int i = 0; i < d; i++) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

// inner product, used by the "norms - 2*dot" rerank formulation
inline float ip(const float* __restrict a, const float* __restrict b, int d) {
    if (caps().avx512f) {
        __m512 acc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= d; i += 16)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
        float s = _mm512_reduce_add_ps(acc);
        for (; i < d; i++) s += a[i] * b[i];
        return s;
    }
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
}

// squared norm of each row of a (n x d), written to out[n]
inline void row_sqr_norms(const float* __restrict a, size_t n, int d, float* __restrict out) {
    if (caps().avx512f) {
        for (size_t i = 0; i < n; i++) {
            const float* p = a + i * (size_t)d;
            __m512 acc = _mm512_setzero_ps();
            int j = 0;
            for (; j + 16 <= d; j += 16) { __m512 v = _mm512_loadu_ps(p + j); acc = _mm512_fmadd_ps(v, v, acc); }
            float s = _mm512_reduce_add_ps(acc);
            for (; j < d; j++) s += p[j] * p[j];
            out[i] = s;
        }
        return;
    }
    for (size_t i = 0; i < n; i++) {
        const float* p = a + i * (size_t)d;
        float s = 0; for (int j = 0; j < d; j++) s += p[j] * p[j];
        out[i] = s;
    }
}

// ------------------------------------------------------------------ top-k
// returns indices of the k smallest entries of dist[0..n) in ascending order
inline void topk_smallest(const float* dist, size_t n, size_t k, std::vector<u32>& out) {
    out.clear();
    if (n == 0 || k == 0) return;
    k = std::min(k, n);
    std::vector<u32> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(),
                     [&](u32 a, u32 b) { return dist[a] < dist[b]; });
    idx.resize(k);
    std::sort(idx.begin(), idx.end(), [&](u32 a, u32 b) { return dist[a] < dist[b]; });
    out = std::move(idx);
}

// ------------------------------------------------------------------ batched rotation
// Y = X * R^T  where R is d x d row-major (rotation rows are R's rows) and X is n x d.
// We compute y[k] = dot(x, R_row_k).  Tiled + OpenMP parallel for cache friendliness.
void rotate_rows(const float* X, size_t n, int d, const float* R, float* Y);

} // namespace eh
