#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

/**
  Base convolution kernel

  N = input array
  F = convolution filter (also called convolution kernel)
  P = output array
  r = radius

  This performs 2 floating operations for every 8 bytes read
  `pvalue += F[frow * width + fcol] * N[inrow * width + incol];`
  Hence, the arithmetic intensity is 0.25 OP/B
*/
__global__ void convolution_2d(float *N, float *F, float *P, int r, int width,
                               int height) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;

  float pvalue = 0.0f; // the accumulator
  for (int frow = 0; frow < 2 * r + 1; frow++) {
    for (int fcol = 0; fcol < 2 * r + 1; fcol++) {
      int inrow = outRow - r + frow;
      int incol = outCol - r + fcol;

      if (inrow >= 0 && inrow < height && incol >= 0 && incol < width) {
        pvalue += F[frow * width + fcol] * N[inrow * width + incol];
      }
    }
  }
  P[outRow * width + outCol] = pvalue;
}

void conv_stub(const float *F_h) {
  
  // ..... need to complete this ......

}