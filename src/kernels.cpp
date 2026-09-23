// kernels.cpp - out-of-line SIMD kernels
#include "simd.h"
#include <omp.h>

namespace eh {

// Y[i][k] = dot( X[i] , W[k] )   with X: n x d row-major, W: d x d row-major
void rotate_rows(const float* X, size_t n, int d, const float* W, float* Y) {
    const bool avx512 = caps().avx512f;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; i++) {
        const float* x = X + (size_t)i * d;
        float* y = Y + (size_t)i * d;
        if (avx512) {
            for (int k = 0; k < d; k++) {
                const float* w = W + (size_t)k * d;
                __m512 acc = _mm512_setzero_ps();
                int j = 0;
                for (; j + 16 <= d; j += 16)
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + j), _mm512_loadu_ps(w + j), acc);
                float s = _mm512_reduce_add_ps(acc);
                for (; j < d; j++) s += x[j] * w[j];
                y[k] = s;
            }
        } else {
            for (int k = 0; k < d; k++) {
                const float* w = W + (size_t)k * d;
                float s = 0;
                for (int j = 0; j < d; j++) s += x[j] * w[j];
                y[k] = s;
            }
        }
    }
}

} // namespace eh
