#define _POSIX_C_SOURCE 200809L

#include "sgemm.h"
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
#define ABS_TOLERANCE 1e-4f
#define REL_TOLERANCE 1e-5f

// Nominal all-core AVX-512 peak for this i5-1135G7:
// 4 cores * 3.8 GHz * 1 FMA/cycle * 16 floats/FMA * 2 FLOPs/float.
#define PEAK_GFLOPS 486.4

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
  int passed = 1;
  for (int i = 0; i < N * N; i++) {
    float diff = fabsf(C_ref[i] - C_test[i]);
    float tolerance = ABS_TOLERANCE + REL_TOLERANCE * fabsf(C_ref[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > tolerance) {
      passed = 0;
    }
  }
  if (!passed) {
    printf(" FAIL");
    return 0;
  }
  printf(" PASS");
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
  printf(" CPU SGEMM Benchmark Harness (N = %d)\n", N);
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
  SgemmImpl implementations[] = {
      {"sgemm_01 (Base Naive)", sgemm_01},
      {"sgemm_02 (Loop Reordered)", sgemm_02},
      {"sgemm_03 (Tiling)", sgemm_03},
      {"sgemm_04 (AVX-512 4x64)", sgemm_04},
      {"sgemm_05 (Parallel)", sgemm_05},
  };
  int num_impls = sizeof(implementations) / sizeof(implementations[0]);

  // Compute reference output using baseline (sgemm_01)
  printf("Computing reference output using %s...\n", implementations[0].name);
  memset(C_ref, 0, matrix_bytes);
  double reference_start = get_time_sec();
  implementations[0].fn(A, B, C_ref, N);
  double reference_time = get_time_sec() - reference_start;
  printf("Reference computation complete.\n\n");

  printf("%-30s | %-12s | %-12s | %-12s | %-14s\n", "Implementation", "Time (ms)",
         "GFLOPS", "% Peak", "Verification");
  printf("---------------------------------------------------------------------------------------------------\n");

  for (int i = 0; i < num_impls; i++) {
    SgemmImpl impl = implementations[i];
    double min_time = reference_time;
    const float *result = C_ref;

    // The reference computation is also the benchmark run for the baseline.
    if (i != 0) {
      result = C_test;

      // Warmup runs
      for (int w = 0; w < WARMUP_RUNS; w++) {
        memset(C_test, 0, matrix_bytes);
        impl.fn(A, B, C_test, N);
      }

      // Benchmark runs
      min_time = 1e9;
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
    }

    double time_ms = min_time * 1000.0;
    double gflops = gflops_factor / min_time;
    double pct_peak = (gflops / PEAK_GFLOPS) * 100.0;

    printf("%-30s | %12.2f | %12.2f | %11.2f%% |", impl.name, time_ms, gflops, pct_peak);

    verify_result(C_ref, result, N);
    printf("\n");
  }

  printf("---------------------------------------------------------------------------------------------------\n");

  free(A);
  free(B);
  free(C_ref);
  free(C_test);

  return EXIT_SUCCESS;
}
