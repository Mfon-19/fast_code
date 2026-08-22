#ifndef SGEMM_H
#define SGEMM_H

#include <stddef.h>

// Standard SGEMM (Single Precision General Matrix Multiplication) Function
// Pointer A: N x N input matrix (row-major) B: N x N input matrix (row-major)
// C: N x N output matrix (row-major)
// N: Dimension size
typedef void (*sgemm_fn)(const float *__restrict A, const float *__restrict B,
                         float *__restrict C, const int N);

typedef struct {
  const char *name;
  sgemm_fn fn;
} SgemmImpl;

// Implementation prototypes
void sgemm_01(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N);
void sgemm_02(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N);
void sgemm_03(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N);
void sgemm_04(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N);
void sgemm_05(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N);
void sgemm_06(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N);
#ifdef HAVE_MKL
void sgemm_mkl(const float *__restrict A, const float *__restrict B,
               float *__restrict C, const int N);
#endif

#endif // SGEMM_H
