#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 1)

__constant__ float c[7];

/*
  Thread coarsening. Instead of threads operating on single elemtns,
  a thread operates on a column
*/
__global__ void stencil_03(float *in, float *out, unsigned int N) {
  int iStart = blockIdx.z * OUT_TILE_DIM;
  int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

  __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
  __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
  __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
  if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
    inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
  }
  if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
    inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
  }

  for (int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
    if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
      inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
      if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
          threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
        out[i * N * N + j * N + k] =
            c[0] * inCurr_s[threadIdx.y][threadIdx.x] +
            c[1] * inCurr_s[threadIdx.y][threadIdx.x - 1] +
            c[2] * inCurr_s[threadIdx.y][threadIdx.x + 1] +
            c[3] * inCurr_s[threadIdx.y - 1][threadIdx.x] +
            c[4] * inCurr_s[threadIdx.y + 1][threadIdx.x] +
            c[5] * inPrev_s[threadIdx.y][threadIdx.x] +
            c[6] * inNext_s[threadIdx.y][threadIdx.x];
      }
    }
    __syncthreads();
    inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
    inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
  }
}

void stencil_stub(float *in_h, float *out_h, unsigned int N) {
  float hostC[7] = {-6, 1, 1, 1, 1, 1, 1};
  cudaMemcpyToSymbol(c, hostC, sizeof(hostC));
}