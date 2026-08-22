/*
  Two-level cache-blocked SGEMM

  The 128x128 inner tiles from sgemm_03 are grouped into 512x512 outer
  regions. The inner level retains the compiler-vectorized loop shape, while
  the outer level attempts to keep a larger working set resident in L2/L3.
*/

#include "sgemm.h"

#define TILE_SIZE 128
#define CACHE_TILE_SIZE 512

static inline int min(const int a, const int b) { return a < b ? a : b; }

void sgemm_04(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N) {
  for (int ic = 0; ic < N; ic += CACHE_TILE_SIZE) {
    const int ic_end = min(ic + CACHE_TILE_SIZE, N);
    for (int kc = 0; kc < N; kc += CACHE_TILE_SIZE) {
      const int kc_end = min(kc + CACHE_TILE_SIZE, N);
      for (int jc = 0; jc < N; jc += CACHE_TILE_SIZE) {
        const int jc_end = min(jc + CACHE_TILE_SIZE, N);

        for (int io = ic; io < ic_end; io += TILE_SIZE) {
          for (int ko = kc; ko < kc_end; ko += TILE_SIZE) {
            for (int jo = jc; jo < jc_end; jo += TILE_SIZE) {
              for (int i = io; i < io + TILE_SIZE; i++) {
                for (int k = ko; k < ko + TILE_SIZE; k++) {
                  for (int j = jo; j < jo + TILE_SIZE; j++) {
                    C[i * N + j] += A[i * N + k] * B[k * N + j];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
