#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

// Standard Matrix Multiplication Function Pointer
// A: N x N input matrix (row-major)
// B: N x N input matrix (row-major)
// C: N x N output matrix (row-major)
// N: Dimension size
typedef void (*matmul_fn)(const float *A, const float *B, float *C,
                          const int N);

typedef struct {
  const char *name;
  matmul_fn fn;
} MatmulImpl;

// Implementation prototypes
void matmul_01(const float *A, const float *B, float *C, const int N);
void matmul_02(const float *A, const float *B, float *C, const int N);

#endif // MATMUL_H
