#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define THREADS_PER_BLOCK 256

/*
  Base matmul kernel
*/
__global__ void sgemm(const float *A, const float *B,
                      float *C, int N) {
  int idx = blockDim.x;
}

void sgemm_stub(const float *A_h, const float *B_h,
                float *C_h, int N) {
  float *A_d, *B_d, *C_d;

  // Allocate on device
  cudaMalloc((void **)&A_d, sizeof(float) * N);
  cudaMalloc((void **)&B_d, sizeof(float) * N);
  cudaMalloc((void **)&C_d, sizeof(float) * N);

  // Copy A_h and B_h to device arrays
  cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, sizeof(float) * N, cudaMemcpyHostToDevice);

  int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  sgemm<<<blocks, THREADS_PER_BLOCK>>>(A_d, B_d, C_d, N);

  cudaMemcpy(C_h, C_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
}