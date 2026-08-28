#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>

#define NUM_BINS 256

/**
  Privatization + shared memory
*/
__global__ void hist_03(char *data, unsigned int length, unsigned int *hist) {
  __shared__ unsigned int hist_s[NUM_BINS];
  for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    hist_s[bin] = 0u;
  }
  __syncthreads();

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int pos = data[i] - 'a';
    if (pos >= 0 and pos < 26) {
      atomicAdd(&hist_s[blockIdx.x * NUM_BINS + pos / 4], 1);
    }
  }

  if (blockIdx.x > 0) {
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
      unsigned int bin_val = hist_s[blockIdx.x * NUM_BINS + bin];
      if (bin_val > 0) {
        atomicAdd(&hist[bin], bin_val);
      }
    }
  }
}