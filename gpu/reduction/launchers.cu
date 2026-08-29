#include "reduction.h"

#include <cuda_runtime.h>

__global__ void reduction_01(float *input, float *output);
__global__ void reduction_02(float *input, float *output);
__global__ void reduction_03(float *input, float *output);
__global__ void reduction_04(float *input, float *output);
__global__ void reduction_05(float *input, float *output);

#define BENCHMARK_BLOCK_DIM 1024
#define BASIC_ELEMENTS_PER_BLOCK (2 * BENCHMARK_BLOCK_DIM)
#define COARSENED_ELEMENTS_PER_BLOCK (3 * 2 * BENCHMARK_BLOCK_DIM)

int supports_2048(size_t N) { return N == BASIC_ELEMENTS_PER_BLOCK; }

int supports_multiple_2048(size_t N) {
  return N != 0 && N % BASIC_ELEMENTS_PER_BLOCK == 0;
}

int supports_multiple_6144(size_t N) {
  return N != 0 && N % COARSENED_ELEMENTS_PER_BLOCK == 0;
}

cudaError_t launch_reduction_01(float *input, float *output, size_t,
                                cudaStream_t stream) {
  reduction_01<<<1, BENCHMARK_BLOCK_DIM, 0, stream>>>(input, output);
  return cudaGetLastError();
}

cudaError_t launch_reduction_02(float *input, float *output, size_t,
                                cudaStream_t stream) {
  reduction_02<<<1, BENCHMARK_BLOCK_DIM, 0, stream>>>(input, output);
  return cudaGetLastError();
}

cudaError_t launch_reduction_03(float *input, float *output, size_t,
                                cudaStream_t stream) {
  reduction_03<<<1, BENCHMARK_BLOCK_DIM, 0, stream>>>(input, output);
  return cudaGetLastError();
}

cudaError_t launch_reduction_04(float *input, float *output, size_t N,
                                cudaStream_t stream) {
  const unsigned int blocks =
      (unsigned int)(N / BASIC_ELEMENTS_PER_BLOCK);
  reduction_04<<<blocks, BENCHMARK_BLOCK_DIM, 0, stream>>>(input, output);
  return cudaGetLastError();
}

cudaError_t launch_reduction_05(float *input, float *output, size_t N,
                                cudaStream_t stream) {
  const unsigned int blocks =
      (unsigned int)(N / COARSENED_ELEMENTS_PER_BLOCK);
  reduction_05<<<blocks, BENCHMARK_BLOCK_DIM, 0, stream>>>(input, output);
  return cudaGetLastError();
}
