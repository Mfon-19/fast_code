#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>

#define NUM_BINS 4

/**
  Privatization. Each thread works on a copy of its block's private
  histogram, reducing contention to global hist array. At the end,
  threads from each block update block 0's private copy.  
*/
__global__ void hist_02(char *data, unsigned int length, unsigned int *hist) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int pos = data[i] - 'a';
    if (pos >= 0 and pos < 26) {
      atomicAdd(&hist[blockIdx.x * NUM_BINS + pos / 4], 1);
    }
  }

  if (blockIdx.x > 0) {
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
      unsigned int bin_val = hist[blockIdx.x * NUM_BINS + bin];
      if (bin_val > 0) {
        atomicAdd(&hist[bin], bin_val);
      }
    }
  }
}