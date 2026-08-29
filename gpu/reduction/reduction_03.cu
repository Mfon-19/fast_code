#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

#define BLOCK_DIM 1024

/**
  Shared memory. The previous kernel is excellent for decreasing
  control divergence, but there remains the problem that each
  thread performs two reads + a write to global memory. Here,
  each thread first loads the input it will need into shared
  memory, then subsequent excecutions read from and write to
  shared memory
*/
__global__ void reduction_03(float *input, float *output) {
  __shared__ float input_s[BLOCK_DIM];
  unsigned int i = threadIdx.x;
  input_s[i] = input[i] + input[i + BLOCK_DIM];
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      input_s[i] += input_s[i + stride];
    }
  }
  if (threadIdx.x == 0) {
    *output = input_s[0];
  }
}