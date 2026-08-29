#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 3

/**
  Thread coarsening. Use less threads to do the same work
*/
__global__ void reduction_05(float *input, float *output) {
  __shared__ float input_s[BLOCK_DIM];
  // each block works on size blockDim.x * 2 of the input
  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;
  float sum = input[i];
  for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++) {
    sum += input[i + tile * BLOCK_DIM];
  }

  input_s[t] = sum;
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