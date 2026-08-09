#define _POSIX_C_SOURCE 200809L

#include "matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Alignment for SIMD operations (64 bytes for AVX-512)
#define ALIGNMENT 64
#define DEFAULT_N 1024
#define WARMUP_RUNS 1
#define BENCHMARK_RUNS 3
#define TOLERANCE 1e-4f

static void *aligned_alloc_mem(size_t size) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, ALIGNMENT, size) != 0) {
    fprintf(stderr, "Error: Memory allocation failed!\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

static double get_time_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void init_matrix(float *matrix, int N) {
  for (int i = 0; i < N * N; i++) {
    matrix[i] = (float)rand() / (float)RAND_MAX;
  }
}

static int verify_result(const float *C_ref, const float *C_test, int N) {
  float max_diff = 0.0f;
  for (int i = 0; i < N * N; i++) {
    float diff = fabsf(C_ref[i] - C_test[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  if (max_diff > TOLERANCE) {
    printf(" [FAIL]");
    return 0;
  }
  printf(" [PASS]");
  return 1;
}

int main(int argc, char **argv) {
  int N = DEFAULT_N;
  if (argc > 1) {
    N = atoi(argv[1]);
    if (N <= 0) {
      fprintf(stderr, "Invalid matrix dimension N: %d\n", N);
      return EXIT_FAILURE;
    }
  }

  printf("=========================================================\n");
  printf(" CPU Matrix Multiplication Benchmark Harness (N = %d)\n", N);
  printf("=========================================================\n\n");

  // Total floating point operations: 2 * N^3 (N^3 multiplies + N^3 adds)
  double gflops_factor = (2.0 * (double)N * (double)N * (double)N) / 1e9;
  size_t matrix_bytes = (size_t)N * N * sizeof(float);

  float *A = (float *)aligned_alloc_mem(matrix_bytes);
  float *B = (float *)aligned_alloc_mem(matrix_bytes);
  float *C_ref = (float *)aligned_alloc_mem(matrix_bytes);
  float *C_test = (float *)aligned_alloc_mem(matrix_bytes);

  // Seed RNG for reproducible matrices
  srand(42);
  init_matrix(A, N);
  init_matrix(B, N);

  // List of implementations to benchmark
  MatmulImpl implementations[] = {
      {"matmul_01 (Base Naive)", matmul_01},
      // Future implementations go here:
      // {"matmul_02 (Loop Reordered)", matmul_02},
      // {"matmul_03 (Tiling)", matmul_03},
  };
  int num_impls = sizeof(implementations) / sizeof(implementations[0]);

  // Compute reference output using baseline (matmul_01)
  printf("Computing reference output using %s...\n", implementations[0].name);
  memset(C_ref, 0, matrix_bytes);
  implementations[0].fn(A, B, C_ref, N);
  printf("Reference computation complete.\n\n");

  printf("%-30s | %-12s | %-12s | %-20s\n", "Implementation", "Time (ms)",
         "GFLOPS", "Verification");
  printf("---------------------------------------------------------------------"
         "--------------------\n");

  for (int i = 0; i < num_impls; i++) {
    MatmulImpl impl = implementations[i];

    // Warmup runs
    for (int w = 0; w < WARMUP_RUNS; w++) {
      memset(C_test, 0, matrix_bytes);
      impl.fn(A, B, C_test, N);
    }

    // Benchmark runs
    double min_time = 1e9;
    for (int r = 0; r < BENCHMARK_RUNS; r++) {
      memset(C_test, 0, matrix_bytes);
      double start = get_time_sec();
      impl.fn(A, B, C_test, N);
      double end = get_time_sec();
      double elapsed = end - start;
      if (elapsed < min_time) {
        min_time = elapsed;
      }
    }

    double time_ms = min_time * 1000.0;
    double gflops = gflops_factor / min_time;

    printf("%-30s | %12.2f | %12.2f |", impl.name, time_ms, gflops);

    // Verification (for baseline, compare against itself; for future impls,
    // compare against C_ref)
    verify_result(C_ref, C_test, N);
    printf("\n");
  }

  printf("---------------------------------------------------------------------"
         "--------------------\n");

  free(A);
  free(B);
  free(C_ref);
  free(C_test);

  return EXIT_SUCCESS;
}