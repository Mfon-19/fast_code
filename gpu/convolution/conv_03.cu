#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

#define FILTER_RADIUS 2
#define SIDE_LENGTH (2 * FILTER_RADIUS + 1)

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

/**
  Here we use the classic tiling strategy to reduce DRAM traffic

  The one complication for a tiled convolution kernel is that to
  calculate an n x n output, we need to load in an (n + n)^2 input
  taking into account the radius. Thus, there are two ways to determine
  block sizes: block size equal to the input tile, which will make
  loading into shared memory easy, but some threads would have to be
  deactivated when calculating the output; the other method is to make
  the block size equal to the output tile size, making output calculation
  straightforward, but complicating shared memory loading. Here, we
  use the first method
*/
__global__ void convolution_2d(float *N, float *P, int width,
                               int height) {
  int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
  int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

  // Each thread in the block loads in a portion of the
  // shared N
  if (row >= 0 && row < height && col >= 0 && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads(); // Wait till input tile is loaded

  int tileCol = threadIdx.x - FILTER_RADIUS; // get into the output tile
  int tileRow = threadIdx.y - FILTER_RADIUS;

  if (col >= 0 && col < width && row >= 0 && row < height) {
    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 &&
        tileRow < OUT_TILE_DIM) {
      float pvalue = 0.0f; // the accumulator
      for (int frow = 0; frow < 2 * FILTER_RADIUS + 1; frow++) {
        for (int fcol = 0; fcol < 2 * FILTER_RADIUS + 1; fcol++) {
          pvalue += F[frow][fcol] * N_s[tileRow + frow][tileCol + fcol];
        }
      }
      P[row * width + col] = pvalue;
    }
  }
}

void conv_stub(const float *F_h) {
  
  // ..... need to complete this ......

  cudaMemcpyToSymbol(F, F_h, SIDE_LENGTH * SIDE_LENGTH * sizeof(float));
}