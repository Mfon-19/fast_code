/*
  Loop reordering SGEMM

  sgemm_01 had good locality for A and C, but terrible for B. This is because
  each increment of k in the inner loop caused a walk along a row of A, each
  element separated by 1; each increment of k walked down a column of B, and
  since the arrays are arranged in row-major order, each k accessed a new row,
  separating elements by N.

  As it turns out, it doesn't matter what order we put the for loops in, the 
  results remain the same. And so we went with the i -> k -> j loop. In the
  inner loop, as j increases, we walk along a row of C and also along a row 
  of B. There is good spatial locality for all accesses to each array.
*/

#include "sgemm.h"

void sgemm_02(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}
