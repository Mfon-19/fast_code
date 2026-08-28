#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 1)

__constant__ float c[7];

/*
  Tiling. Block size is the size of the input 3D block, some threads
  are deactivated when calculating the output
*/
__global__ void stencil_02(float *in, float *out, unsigned int N) {
  unsigned int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
  unsigned int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  unsigned int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
  }
  __syncthreads();

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >= 1 &&
        threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
        threadIdx.x < IN_TILE_DIM - 1) {
      out[i * N * N + j * N + k] =
          c[0] * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
          c[1] * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
          c[2] * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
          c[3] * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
          c[4] * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
          c[5] * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
          c[6] * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
    }
  }
}

void stencil_stub(float *in_h, float *out_h, unsigned int N) {
  float hostC[7] = {-6, 1, 1, 1, 1, 1, 1};
  cudaMemcpyToSymbol(c, hostC, sizeof(hostC));
}