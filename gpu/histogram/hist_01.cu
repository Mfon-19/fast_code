#include <cuda_runtime.h>

/**
  Base kernel using atomicAdd for atomic increments to a shared
  hist array. Tp of atomic updates to address N is limited by the total
  latency of a read-modify-write (read + write latencies)
*/
__global__ void hist_01(char *data, unsigned int length, unsigned int *hist) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int pos = data[i] - 'a';
    if (pos >= 0 and pos < 26) {
      atomicAdd(&hist[pos / 4], 1);
    }
  }
}