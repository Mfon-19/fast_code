#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define FILTER_RADIUS 2
#define SIDE_LENGTH (2 * FILTER_RADIUS + 1)
// will be placed into device constant memory
__constant__ float F[SIDE_LENGTH][SIDE_LENGTH];

/**
  We observe the following:
  - F is typically quite small and the radius of most
  convolutions is 7 or less.
  - The contents of F do not change throughout the execution
  of the kernel
  - All threads access the filter elements in the same order
  starting from F[0][0] and moving one element at a time

  These observations make F a good candidate for constant memory
  and caching so we declare it as a `__constant__` at the top of 
  the file. The GPU caches this in the constant cache

  As a result of the cache, we have doubled the arithmetic intensity
  as we no longer go to DRAM for `F`. 2 flops / 4 bytes
*/
__global__ void convolution_2d(float *N, float *P, int r, int width,
                               int height) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;

  float pvalue = 0.0f; // the accumulator
  for (int frow = 0; frow < 2 * r + 1; frow++) {
    for (int fcol = 0; fcol < 2 * r + 1; fcol++) {
      int inrow = outRow - r + frow;
      int incol = outCol - r + fcol;

      if (inrow >= 0 && inrow < height && incol >= 0 && incol < width) {
        pvalue += F[frow][fcol] * N[inrow * width + incol];
      }
    }
  }
  P[outRow * width + outCol] = pvalue;
}

void conv_stub(const float *F_h) {
  
  // ..... need to complete this ......

  cudaMemcpyToSymbol(F, F_h, SIDE_LENGTH * SIDE_LENGTH * sizeof(float));
}