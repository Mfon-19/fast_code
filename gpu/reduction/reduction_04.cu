#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

#define BLOCK_DIM 1024

/**
  Segmented reduction. The previous kernels had the disadvantage
  of being launched on just one block, limiting input. This kernel
  first segments the input into N segments, has each block work on
  a segment, then at the end atomically add result into an address
*/
__global__ void reduction_04(float *input, float *output) {
  __shared__ float input_s[BLOCK_DIM];
  // each block works on size blockDim.x * 2 of the input
  unsigned int segment = 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;
  input_s[t] = input[i] + input[i + BLOCK_DIM];
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  if (t == 0) {
    atomicAdd(output, input_s[0]);
  }
}