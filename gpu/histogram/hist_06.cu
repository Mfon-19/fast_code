#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>

#define NUM_BINS 256

/**
  Aggregation. If consecutive iterations are updating the same bin,
  just aggregate them, then perform atomic update later. This reduces
  the number of atomic updates to make, reducing contention on that
  bin.
*/
__global__ void hist_05(char *data, unsigned int length, unsigned int *hist) {
  __shared__ unsigned int hist_s[NUM_BINS];
  for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    hist_s[bin] = 0u;
  }
  __syncthreads();

  unsigned int accumulator;
  int prev_bin_idx = -1;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
    int pos = data[i] - 'a';
    if (pos >= 0 and pos < 26) {
      int bin = pos / 4;
      if (bin == prev_bin_idx) {
        accumulator++;
      } else {
        if (accumulator > 0) {
          atomicAdd(&hist_s[prev_bin_idx], accumulator);
        }
        accumulator = 1;
        prev_bin_idx = bin;
      }
    }
  }
  if (accumulator > 0) {
    atomicAdd(&hist_s[prev_bin_idx], accumulator);
  }

  __syncthreads();
  for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned int bin_val = hist_s[blockIdx.x * NUM_BINS + bin];
    if (bin_val > 0) {
      atomicAdd(&hist[bin], bin_val);
    }
  }
}