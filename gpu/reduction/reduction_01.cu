#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

/**
  Reduction requires a lot of corporation among threads. Threads
  in different grids can't communicate, hence can't communicate
  and so we perform a reduction in just one thread block. A thread
  block can contain up to 1024 threads, meaning this kernel can
  process up to 2048 input elements
*/
__global__ void reduction_01(float *input, float *output) {
  // threads are assigned to even locations in the input array
  // input[0]...input[2]...input[4]...input[6]...
  unsigned int i = 2 * threadIdx.x;
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}