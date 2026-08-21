#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

#define FILTER_RADIUS 2
#define SIDE_LENGTH (2 * FILTER_RADIUS + 1)

#define TILE_DIM 32

__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

/*
  In the previous tiled convolutin kernel, we had complexity due to
  loading of halo cells; the input tile was larger than the output.
  However, we can observe that the halo cells of some input tile
  overlaps with the internal input cells of a neighboring block(s).
  There is a good probability that by the time a block needs its halo
  cells, they are already in the L2 cache since a neighboring block
  accessed it recently causing accesses to be served from L2 without
  causing additional DRAM traffic. This means that we forgo the
  complexity of having different input and output tile sizes; if we
  need a halo cell, just access from N instead of loading halos into
  the shared memory.
*/
__global__ void convolution_2d(float *N, float *P, int width, int height) {
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int row = blockIdx.y * TILE_DIM + threadIdx.y;

  __shared__ float N_s[TILE_DIM][TILE_DIM];

  // Each thread in the block loads in a portion of the
  // shared N
  if (row < height && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads(); // Wait till input tile is loaded

  if (col < width && row < height) {
    float pvalue = 0.0f;
    for (int frow = 0; frow < 2 * FILTER_RADIUS + 1; frow++) {
      for (int fcol = 0; fcol < 2 * FILTER_RADIUS + 1; fcol++) {
        // test if within the input tile
        if (threadIdx.x - FILTER_RADIUS + fcol >= 0 &&
            threadIdx.x - FILTER_RADIUS + fcol < TILE_DIM &&
            threadIdx.y - FILTER_RADIUS + frow >= 0 &&
            threadIdx.y - FILTER_RADIUS + frow < TILE_DIM) {
          pvalue += F[frow][fcol] * N_s[threadIdx.y + frow][threadIdx.x + fcol];
        } else {
          // we are in a halo cell, test if this is a ghost
          // cell. if not, take halo cell from N
          if (row - FILTER_RADIUS + frow >= 0 &&
              row - FILTER_RADIUS + frow < height &&
              col - FILTER_RADIUS + fcol >= 0 &&
              col - FILTER_RADIUS + fcol < width) {
            pvalue += F[frow][fcol] * N[(row - FILTER_RADIUS + frow) * width +
                                        col - FILTER_RADIUS + fcol];
          }
        }
      }
    }
    P[row * width + col] = pvalue;
  }
}

void conv_stub(const float *F_h) {

  // ..... need to complete this ......

  cudaMemcpyToSymbol(F, F_h, SIDE_LENGTH * SIDE_LENGTH * sizeof(float));
}