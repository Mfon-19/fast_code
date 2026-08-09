/*
  Tiling SGEMM

  sgemm_02 with loop reordering gave us great data locality, but it has the
  problem where computing a row for C requires us to read an entire row of A
  and the entirety of B. It has poor temporal locality for B. If B doesn't
  fit in the cache, we must fetch it again for every row of C

  What we can do is to process C in blocks; process TILE_SIZE rows of C together.
  A small block of B can remain in the cache while it is reused for TILE_SIZE
  different rows of C, giving us better temporal locality. Tiling reduces cache
  misses, and subseqently, accesses to slower memory in the heirarchy. Ideally,
  B is streamed N/TILE_SIZE times instead of N times giving us up to a TILE_SIZE
  reduction in slow-memory traffic for B.
*/

#include "sgemm.h"

#define TILE_SIZE 64

void sgemm_03(const float *__restrict A, const float *__restrict B,
              float *__restrict C, const int N) {
  for (int io = 0; io < N; io += TILE_SIZE) {
    for (int ko = 0; ko < N; ko += TILE_SIZE) {
      for (int jo = 0; jo < N; jo += TILE_SIZE) {
        for (int i = io; i < io + TILE_SIZE; i++) {
          for (int k = ko; k < ko + TILE_SIZE; k++) {
            for (int j = jo; j < jo + TILE_SIZE; j++) {
              C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
