#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

/**
  Reducing control divergence. In the previous kernel, active
  threads are spread out accross the entire block; in every 
  warp, on each iteration, half the threads are active, increasing
  control divergence significantly within warps.

  This kernel has brilliantly avoids intra-warp divergence; it
  basically lets the kernel exhibit an all-threads-active or 
  no-threads-active within warps. When stride == blockDim.x, all
  threads in all warps are active, on the next iteration, 
  the if (threadIdx.x < stride) turns off all warps where the lowest
  thread id is less than the stride, this continues till the first
  warp, warp 0 where it goes 32 -> 16 -> 8 -> 4 -> 2 -> 1
*/
__global__ void reduction_02(float *input, float *output) {
  unsigned int i = threadIdx.x;
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}