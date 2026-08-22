#include "sgemm.h"

#ifdef HAVE_MKL

#include <mkl_cblas.h>

void sgemm_mkl(const float *__restrict A, const float *__restrict B,
               float *__restrict C, const int N) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A,
              N, B, N, 0.0f, C, N);
}

#endif
