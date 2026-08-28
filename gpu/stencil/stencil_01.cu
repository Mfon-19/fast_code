#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

__constant__ float c[7];

/*
  Arithmetic intensity: 13 flops / 28 mem accesses = 0.46op/b
*/
__global__ void stencil_01(float *in, float *out, unsigned int N) {
  unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[i * N * N + j * N + k] = c[0] * in[i * N * N + j * N + k] +
                                 c[1] * in[i * N * N + j * N + (k - 1)] +
                                 c[2] * in[i * N * N + j * N + (k + 1)] +
                                 c[3] * in[i * N * N + (j - 1) * N + k] +
                                 c[4] * in[i * N * N + (j + 1) * N + k] +
                                 c[5] * in[(i - 1) * N * N + j * N + k] +
                                 c[6] * in[(i + 1) * N * N + j * N + k];
  }
}

void stencil_stub(float *in_h, float *out_h, unsigned int N) {
  float hostC[7] = {-6, 1, 1, 1, 1, 1, 1};
  cudaMemcpyToSymbol(c, hostC, sizeof(hostC));
}