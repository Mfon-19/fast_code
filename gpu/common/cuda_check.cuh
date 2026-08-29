#pragma once

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

static inline void cuda_check(cudaError_t status, const char *expression,
                              const char *file, int line) {
  if (status == cudaSuccess) {
    return;
  }

  fprintf(stderr, "CUDA error at %s:%d\n  %s\n  %s\n", file, line,
          expression, cudaGetErrorString(status));
  exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                               \
  cuda_check((expression), #expression, __FILE__, __LINE__)
