// rabitq.h - randomized binary quantization: orthogonal rotation + sign binarization + segment packing
//
// Dimension generality (needed by the dimension-ablation experiments)
// ----------------------------------------------------------------
// The original implementation could only handle d <= 128: the sign bits lived in
// two u64 words and a sub-segment value had to fit in a single u32 (so bps <= 32).
// Both restrictions are removed here:
//
//   * the sign mask is now ceil(used/64) u64 words            -> any d
//   * a sub-segment is kper = ceil(bps/32) u32 words          -> any bps
//   * `used = min(d, m*bps)` bits of the rotated vector are consumed; choosing
//     m*bps < d is a legitimate (and interesting) configuration: the code then
//     covers only a prefix of the rotated dimensions, i.e. a random subspace.
//
// Backward compatibility is exact.  With kper == 1 and bps == 16 the packing rule
// below places sign j of segment l at bit (bps-1-j) of the single word, which is
// exactly what the old code did (`v |= 1u << (bps-1-j)`), so d=128/m=8 produces
// bit-identical codes and hence identical PRF keys and identical recall.
#pragma once
#include "common.h"
#include "simd.h"
#include <random>

namespace eh {

struct RaBitQ {
    int d = 128;
    int m = 8;          // number of sub-segments
    int bps = 16;       // bits per sub-segment (explicit; no longer forced to d/m)
    int kper = 1;       // u32 words per sub-segment = ceil(bps/32)
    int used = 128;     // bits of the sign code actually consumed = min(d, m*bps)
    int nw = 2;         // u64 words holding `used` bits = ceil(used/64)
    std::vector<float> Q;    // d x d, orthonormal columns (row-major)
    std::vector<float> W;    // Q^T (row-major) -> y[k] = dot(x, W[k])

    // deterministic random orthogonal matrix built by QR of a seeded gaussian matrix
    // bps_ <= 0 means "auto" = max(1, d/m), preserving the historical behaviour
    void init(int d_, int m_, int bps_ = 0, int seed = 42) {
        d = d_; m = m_;
        bps = (bps_ > 0) ? bps_ : std::max(1, d / m);
        kper = (bps + 31) / 32;
        used = std::min(d, m * bps);
        nw = (used + 63) / 64;
        if (nw < 1) nw = 1;

        std::mt19937_64 rng(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<double> A((size_t)d * d);
        for (auto& x : A) x = nd(rng);

        // modified Gram-Schmidt -> Q with orthonormal columns
        std::vector<double> Qd((size_t)d * d, 0.0);
        for (int k = 0; k < d; k++) {
            for (int i = 0; i < d; i++) Qd[(size_t)i * d + k] = A[(size_t)i * d + k];
            for (int j = 0; j < k; j++) {
                double dot = 0;
                for (int i = 0; i < d; i++) dot += Qd[(size_t)i * d + j] * Qd[(size_t)i * d + k];
                for (int i = 0; i < d; i++) Qd[(size_t)i * d + k] -= dot * Qd[(size_t)i * d + j];
            }
            double nrm = 0;
            for (int i = 0; i < d; i++) nrm += Qd[(size_t)i * d + k] * Qd[(size_t)i * d + k];
            nrm = std::sqrt(nrm);
            if (nrm < 1e-12) nrm = 1e-12;
            for (int i = 0; i < d; i++) Qd[(size_t)i * d + k] /= nrm;
        }
        Q.assign((size_t)d * d, 0.f);
        W.assign((size_t)d * d, 0.f);
        for (int i = 0; i < d; i++)
            for (int k = 0; k < d; k++) {
                Q[(size_t)i * d + k] = (float)Qd[(size_t)i * d + k];
                W[(size_t)k * d + i] = (float)Qd[(size_t)i * d + k];   // W = Q^T
            }
    }

    // ---- sign mask: bit g <=> rotated component g is > 0, for g < used -------
    inline void sign_mask(const float* __restrict y, u64* __restrict mask) const {
        for (int w = 0; w < nw; w++) mask[w] = 0ull;
        if (caps().avx512f) {
            const __m512 z = _mm512_setzero_ps();
            int g = 0;
            for (; g + 16 <= used; g += 16) {
                __mmask16 mk = _mm512_cmp_ps_mask(_mm512_loadu_ps(y + g), z, _CMP_GT_OQ);
                mask[g >> 6] |= (u64)mk << (g & 63);
            }
            if (g < used) {
                const int rem = used - g;
                __m512 v = _mm512_setzero_ps();
                for (int j = 0; j < rem; j++) v[j] = y[g + j];
                __mmask16 mk = _mm512_cmp_ps_mask(v, z, _CMP_GT_OQ);
                mk &= (__mmask16)((1u << rem) - 1u);
                mask[g >> 6] |= (u64)mk << (g & 63);
            }
            return;
        }
        for (int g = 0; g < used; g++) if (y[g] > 0) mask[g >> 6] |= 1ull << (g & 63);
    }

    // ---- pack the mask into m segments of kper u32 words --------------------
    // Segment l holds sign bits [l*bps, l*bps+bps).  The bps bits are stored
    // right-aligned across the kper words: sign bit j goes to bit (bps-1-j) of the
    // little-endian word string, i.e. word (bps-1-j)/32, bit (bps-1-j)%32.
    inline void pack_segments(const u64* __restrict mask, u32* __restrict c) const {
        for (int l = 0; l < m; l++) {
            u32* seg = c + (size_t)l * kper;
            for (int w = 0; w < kper; w++) seg[w] = 0u;
            const int g0 = l * bps;
            const int lim = std::min(bps, used - g0);
            for (int j = 0; j < lim; j++) {
                const int g = g0 + j;
                if ((mask[g >> 6] >> (g & 63)) & 1ull) {
                    const int pos = bps - 1 - j;
                    seg[pos >> 5] |= 1u << (pos & 31);
                }
            }
        }
    }

    // encode n vectors into packed segment codes: kper words per (vector, segment)
    void encode(const float* x, size_t n, std::vector<u32>& codes) const {
        const int kp = kper;
        codes.assign(n * (size_t)m * kp, 0);
        std::vector<float> Y(n * (size_t)d);
        // unit-normalise then rotate
        std::vector<float> Xn(n * (size_t)d);
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)n; i++) {
            const float* p = x + (size_t)i * d;
            float nn = 0; for (int j = 0; j < d; j++) nn += p[j] * p[j];
            nn = std::sqrt(nn); if (nn < 1e-10f) nn = 1e-10f;
            float inv = 1.0f / nn;
            for (int j = 0; j < d; j++) Xn[(size_t)i * d + j] = p[j] * inv;
        }
        rotate_rows(Xn.data(), n, d, W.data(), Y.data());

        #pragma omp parallel
        {
            std::vector<u64> mask(nw);
            #pragma omp for schedule(static)
            for (long long i = 0; i < (long long)n; i++) {
                const float* y = Y.data() + (size_t)i * d;
                sign_mask(y, mask.data());
                pack_segments(mask.data(), codes.data() + (size_t)i * m * kp);
            }
        }
    }

    // encode a single vector
    void encode_one(const float* x, std::vector<u32>& code) const {
        std::vector<u32> c;
        encode(x, 1, c);
        code.assign(c.begin(), c.begin() + (size_t)m * kper);
    }

    // ---- hot path: single-query encoding, serial and allocation free ----
    // Spawning an OpenMP region for one vector costs ~4 ms per query, which
    // dominated the whole online latency.  This path keeps everything in
    // caller-provided scratch buffers and uses plain AVX-512 kernels.
    // `code` must hold m * kper u32 words.
    void encode_one_into(const float* __restrict x, float* __restrict xn,
                         float* __restrict rot, u32* __restrict code) const {
        float nn = 0;
        for (int j = 0; j < d; j++) nn += x[j] * x[j];
        nn = std::sqrt(nn);
        if (nn < 1e-10f) nn = 1e-10f;
        const float inv = 1.0f / nn;
        for (int j = 0; j < d; j++) xn[j] = x[j] * inv;

        const bool avx512 = caps().avx512f;
        for (int k = 0; k < d; k++) {
            const float* w = W.data() + (size_t)k * d;
            if (avx512) {
                __m512 acc = _mm512_setzero_ps();
                int j = 0;
                for (; j + 16 <= d; j += 16)
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(xn + j), _mm512_loadu_ps(w + j), acc);
                float s = _mm512_reduce_add_ps(acc);
                for (; j < d; j++) s += xn[j] * w[j];
                rot[k] = s;
            } else {
                float s = 0;
                for (int j = 0; j < d; j++) s += xn[j] * w[j];
                rot[k] = s;
            }
        }
        // stack-allocated mask: nw <= ceil(1536/64) = 24, the cap is only a guard
        u64 mask_st[64];
        u64* mask = (nw <= 64) ? mask_st : new u64[nw];
        sign_mask(rot, mask);
        pack_segments(mask, code);
        if (mask != mask_st) delete[] mask;
    }
};

} // namespace eh
