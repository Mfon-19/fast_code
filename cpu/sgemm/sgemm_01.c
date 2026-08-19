/*
  Base naive SGEMM
*/

#include "sgemm.h"

void sgemm_01(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}
