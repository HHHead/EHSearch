// crypto.h - AES-128 with AES-NI / VAES hardware acceleration
//   * aes_expand   : AES-NI key schedule
//   * aes_prf_u64  : single-block AES based pseudo-random function (deterministic
//                    encryption for keys / ids / trapdoors)
//   * aes_cbc_encrypt : chained CBC encryption (AES-NI)
//   * aes_cbc_decrypt : CBC decryption parallelised with VAES (4 blocks/instruction)
#pragma once
#include "common.h"
#include "simd.h"
#include <immintrin.h>
#include <wmmintrin.h>

namespace eh {

struct AesKey {
    __m128i enc[11];
    __m128i dec[11];   // equivalent-inverse-cipher round keys
    int rounds = 10;
};

inline __m128i aes128_step(__m128i key, __m128i assist) {
    assist = _mm_shuffle_epi32(assist, _MM_SHUFFLE(3, 3, 3, 3));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, assist);
}

inline void aes_expand(const u8* key16, AesKey& k) {
    k.rounds = 10;
    k.enc[0] = _mm_loadu_si128((const __m128i*)key16);
    static const u8 rc[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};
    for (int i = 1; i <= 10; i++)
        k.enc[i] = aes128_step(k.enc[i - 1], _mm_aeskeygenassist_si128(k.enc[i - 1], rc[i - 1]));
    k.dec[0] = k.enc[10];
    for (int i = 1; i < 10; i++) k.dec[i] = _mm_aesimc_si128(k.enc[10 - i]);
    k.dec[10] = k.enc[0];
}

inline __m128i aes_enc_block(const AesKey& k, __m128i b) {
    b = _mm_xor_si128(b, k.enc[0]);
    for (int i = 1; i < 10; i++) b = _mm_aesenc_si128(b, k.enc[i]);
    return _mm_aesenclast_si128(b, k.enc[10]);
}
inline __m128i aes_dec_block(const AesKey& k, __m128i b) {
    b = _mm_xor_si128(b, k.dec[0]);
    for (int i = 1; i < 10; i++) b = _mm_aesdec_si128(b, k.dec[i]);
    return _mm_aesdeclast_si128(b, k.dec[10]);
}

// deterministic PRF used for table keys, ids and query trapdoors
inline u64 aes_prf_u64(const AesKey& k, u64 x) {
    __m128i b = _mm_set_epi64x((long long)(x * 0x9E3779B97F4A7C15ull + 0x1234567ull), (long long)x);
    b = aes_enc_block(k, b);
    return (u64)_mm_cvtsi128_si64(b) ^ (u64)_mm_extract_epi64(b, 1);
}

// ---------------------------------------------------------------- CBC
// encrypts nblocks * 16 bytes, chained (serial by definition)
inline void aes_cbc_encrypt(const AesKey& k, const u8* iv, const u8* in, size_t nblocks, u8* out) {
    __m128i prev = _mm_loadu_si128((const __m128i*)iv);
    for (size_t i = 0; i < nblocks; i++) {
        __m128i p = _mm_loadu_si128((const __m128i*)(in + i * 16));
        __m128i c = aes_enc_block(k, _mm_xor_si128(p, prev));
        _mm_storeu_si128((__m128i*)(out + i * 16), c);
        prev = c;
    }
}

// CBC decryption is embarrassingly parallel: P_i = D(C_i) XOR C_{i-1}
inline void aes_cbc_decrypt(const AesKey& k, const u8* iv, const u8* in, size_t nblocks, u8* out) {
    if (nblocks == 0) return;
    // first block scalar (depends on IV)
    {
        __m128i c0 = _mm_loadu_si128((const __m128i*)in);
        __m128i p0 = _mm_xor_si128(aes_dec_block(k, c0), _mm_loadu_si128((const __m128i*)iv));
        _mm_storeu_si128((__m128i*)out, p0);
    }
    size_t i = 1;
    if (caps().vaes && caps().avx512f) {
        // 4 independent blocks per VAES instruction; round key broadcast to all lanes
        const __m512i rk0 = _mm512_broadcast_i32x4(k.dec[0]);
        const __m512i rkL = _mm512_broadcast_i32x4(k.dec[10]);
        __m512i rkm[9];
        for (int rr = 1; rr < 10; rr++) rkm[rr - 1] = _mm512_broadcast_i32x4(k.dec[rr]);
        for (; i + 4 <= nblocks; i += 4) {
            __m512i c    = _mm512_loadu_si512((const void*)(in + i * 16));
            __m512i prev = _mm512_loadu_si512((const void*)(in + (i - 1) * 16));
            __m512i r = _mm512_xor_si512(c, rk0);
            for (int rr = 0; rr < 9; rr++) r = _mm512_aesdec_epi128(r, rkm[rr]);
            r = _mm512_aesdeclast_epi128(r, rkL);
            r = _mm512_xor_si512(r, prev);
            _mm512_storeu_si512((void*)(out + i * 16), r);
        }
    }
    for (; i < nblocks; i++) {
        __m128i c = _mm_loadu_si128((const __m128i*)(in + i * 16));
        __m128i prev = _mm_loadu_si128((const __m128i*)(in + (i - 1) * 16));
        __m128i p = _mm_xor_si128(aes_dec_block(k, c), prev);
        _mm_storeu_si128((__m128i*)(out + i * 16), p);
    }
}

} // namespace eh
