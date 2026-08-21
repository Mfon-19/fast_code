/*
  Hand-written AVX-512 SGEMM

  This kernel computes four rows by 64 columns at a time. Sixteen ZMM
  registers hold the 4x64 C micro-tile while each group of four B vectors is
  loaded once and reused across all four rows. C is written only after all 64
  k contributions have been accumulated.
*/

#include "sgemm.h"

#include <immintrin.h>

#define TILE_SIZE 64
#define ROW_BLOCK 4

void sgemm_04(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N) {

  for (int io = 0; io < N; io += TILE_SIZE) {
    for (int ko = 0; ko < N; ko += TILE_SIZE) {
      for (int jo = 0; jo < N; jo += TILE_SIZE) {
        for (int i = io; i < io + TILE_SIZE; i += ROW_BLOCK) {
          float *c0p = C + (i + 0) * N + jo;
          float *c1p = C + (i + 1) * N + jo;
          float *c2p = C + (i + 2) * N + jo;
          float *c3p = C + (i + 3) * N + jo;

          // Sixteen accumulators hold four rows of 64 C values.
          __m512 c00 = _mm512_loadu_ps(c0p + 0);
          __m512 c01 = _mm512_loadu_ps(c0p + 16);
          __m512 c02 = _mm512_loadu_ps(c0p + 32);
          __m512 c03 = _mm512_loadu_ps(c0p + 48);
          __m512 c10 = _mm512_loadu_ps(c1p + 0);
          __m512 c11 = _mm512_loadu_ps(c1p + 16);
          __m512 c12 = _mm512_loadu_ps(c1p + 32);
          __m512 c13 = _mm512_loadu_ps(c1p + 48);
          __m512 c20 = _mm512_loadu_ps(c2p + 0);
          __m512 c21 = _mm512_loadu_ps(c2p + 16);
          __m512 c22 = _mm512_loadu_ps(c2p + 32);
          __m512 c23 = _mm512_loadu_ps(c2p + 48);
          __m512 c30 = _mm512_loadu_ps(c3p + 0);
          __m512 c31 = _mm512_loadu_ps(c3p + 16);
          __m512 c32 = _mm512_loadu_ps(c3p + 32);
          __m512 c33 = _mm512_loadu_ps(c3p + 48);

          for (int k = ko; k < ko + TILE_SIZE; ++k) {
            const float *b = B + (size_t)k * N + jo;

            // Load this section of B once, then reuse it for four A rows.
            const __m512 b0 = _mm512_loadu_ps(b + 0);
            const __m512 b1 = _mm512_loadu_ps(b + 16);
            const __m512 b2 = _mm512_loadu_ps(b + 32);
            const __m512 b3 = _mm512_loadu_ps(b + 48);

            const __m512 a0 =
                _mm512_set1_ps(A[(size_t)(i + 0) * N + k]);
            c00 = _mm512_fmadd_ps(a0, b0, c00);
            c01 = _mm512_fmadd_ps(a0, b1, c01);
            c02 = _mm512_fmadd_ps(a0, b2, c02);
            c03 = _mm512_fmadd_ps(a0, b3, c03);

            const __m512 a1 =
                _mm512_set1_ps(A[(size_t)(i + 1) * N + k]);
            c10 = _mm512_fmadd_ps(a1, b0, c10);
            c11 = _mm512_fmadd_ps(a1, b1, c11);
            c12 = _mm512_fmadd_ps(a1, b2, c12);
            c13 = _mm512_fmadd_ps(a1, b3, c13);

            const __m512 a2 =
                _mm512_set1_ps(A[(size_t)(i + 2) * N + k]);
            c20 = _mm512_fmadd_ps(a2, b0, c20);
            c21 = _mm512_fmadd_ps(a2, b1, c21);
            c22 = _mm512_fmadd_ps(a2, b2, c22);
            c23 = _mm512_fmadd_ps(a2, b3, c23);

            const __m512 a3 =
                _mm512_set1_ps(A[(size_t)(i + 3) * N + k]);
            c30 = _mm512_fmadd_ps(a3, b0, c30);
            c31 = _mm512_fmadd_ps(a3, b1, c31);
            c32 = _mm512_fmadd_ps(a3, b2, c32);
            c33 = _mm512_fmadd_ps(a3, b3, c33);
          }

          // Store once per 64-value k tile, rather than after every k.
          _mm512_storeu_ps(c0p + 0, c00);
          _mm512_storeu_ps(c0p + 16, c01);
          _mm512_storeu_ps(c0p + 32, c02);
          _mm512_storeu_ps(c0p + 48, c03);
          _mm512_storeu_ps(c1p + 0, c10);
          _mm512_storeu_ps(c1p + 16, c11);
          _mm512_storeu_ps(c1p + 32, c12);
          _mm512_storeu_ps(c1p + 48, c13);
          _mm512_storeu_ps(c2p + 0, c20);
          _mm512_storeu_ps(c2p + 16, c21);
          _mm512_storeu_ps(c2p + 32, c22);
          _mm512_storeu_ps(c2p + 48, c23);
          _mm512_storeu_ps(c3p + 0, c30);
          _mm512_storeu_ps(c3p + 16, c31);
          _mm512_storeu_ps(c3p + 32, c32);
          _mm512_storeu_ps(c3p + 48, c33);
        }
      }
    }
  }
}
