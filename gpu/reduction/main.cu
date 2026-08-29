#include "reduction.h"
#include "../common/cuda_benchmark.cuh"
#include "../common/cuda_check.cuh"

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_N 2048
#define DEFAULT_WARMUPS 3
#define DEFAULT_RUNS 10
#define ABS_TOLERANCE 1e-2
#define REL_TOLERANCE 2e-5

static size_t parse_size(const char *text, const char *name) {
  char *end = NULL;
  unsigned long long value = strtoull(text, &end, 10);
  if (text[0] == '\0' || *end != '\0' || value == 0 || value > SIZE_MAX) {
    fprintf(stderr, "Invalid %s: %s\n", name, text);
    exit(EXIT_FAILURE);
  }
  return (size_t)value;
}

static int parse_count(const char *text, const char *name) {
  size_t value = parse_size(text, name);
  if (value > 1000) {
    fprintf(stderr, "%s is unreasonably large: %zu\n", name, value);
    exit(EXIT_FAILURE);
  }
  return (int)value;
}

static double initialize_input(float *input, size_t N) {
  uint32_t state = 42;
  double reference = 0.0;
  size_t i;

  for (i = 0; i < N; ++i) {
    state = state * 1664525u + 1013904223u;
    input[i] = (float)((state >> 8) & 0xffffu) / 65535.0f;
    reference += (double)input[i];
  }
  return reference;
}

static int verify(double reference, float result, double *absolute_error,
                  double *relative_error) {
  *absolute_error = fabs(reference - (double)result);
  *relative_error = *absolute_error / fabs(reference);
  return *absolute_error <=
         ABS_TOLERANCE + REL_TOLERANCE * fabs(reference);
}

static int compare_float(const void *left, const void *right) {
  float a = *(const float *)left;
  float b = *(const float *)right;
  return (a > b) - (a < b);
}

static float median_ms(float *samples, int count) {
  int middle = count / 2;
  qsort(samples, (size_t)count, sizeof(float), compare_float);
  if (count % 2 != 0) {
    return samples[middle];
  }
  return 0.5f * (samples[middle - 1] + samples[middle]);
}

static void prepare_run(float *input_d, const float *input_h, size_t bytes,
                        float *output_d, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemsetAsync(output_d, 0, sizeof(float), stream));
  // Keep input reset and H2D transfer outside the kernel-only measurement.
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

int main(int argc, char **argv) {
  size_t N = argc > 1 ? parse_size(argv[1], "N") : DEFAULT_N;
  int runs = argc > 2 ? parse_count(argv[2], "runs") : DEFAULT_RUNS;
  int warmups =
      argc > 3 ? parse_count(argv[3], "warmups") : DEFAULT_WARMUPS;
  size_t bytes;
  int device_count = 0;
  cudaError_t device_status;
  cudaDeviceProp properties;
  float *input_h;
  float *input_d = NULL;
  float *output_d = NULL;
  cudaStream_t stream;
  CudaEventTimer timer;
  double reference;
  int all_passed = 1;
  size_t implementation_index;

  if (argc > 4) {
    fprintf(stderr, "Usage: %s [N] [runs] [warmups]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (N > SIZE_MAX / sizeof(float)) {
    fprintf(stderr, "N is too large\n");
    return EXIT_FAILURE;
  }
  bytes = N * sizeof(float);

  device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    fprintf(stderr,
            "No CUDA GPU is available. Build and run this directory in a "
            "GPU-enabled Google Colab runtime.\n");
    if (device_status != cudaSuccess) {
      fprintf(stderr, "CUDA reported: %s\n",
              cudaGetErrorString(device_status));
    }
    return EXIT_FAILURE;
  }

  memset(&properties, 0, sizeof(properties));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  CUDA_CHECK(cudaSetDevice(0));

  input_h = (float *)malloc(bytes);
  if (input_h == NULL) {
    fprintf(stderr, "Host allocation failed\n");
    return EXIT_FAILURE;
  }
  reference = initialize_input(input_h, N);

  CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
  CUDA_CHECK(cudaMalloc((void **)&output_d, sizeof(float)));
  CUDA_CHECK(cudaStreamCreate(&stream));
  cuda_event_timer_create(&timer);

  ReductionImpl implementations[] = {
      {"reduction_01 (Interleaved Global)", launch_reduction_01,
       supports_2048, "N == 2048"},
      {"reduction_02 (Sequential Global)", launch_reduction_02,
       supports_2048, "N == 2048"},
      {"reduction_03 (Shared Memory)", launch_reduction_03, supports_2048,
       "N == 2048"},
      {"reduction_04 (Segmented Atomic)", launch_reduction_04,
       supports_multiple_2048, "N must be a multiple of 2048"},
      {"reduction_05 (Coarsened Atomic)", launch_reduction_05,
       supports_multiple_6144, "N must be a multiple of 6144"},
  };
  size_t implementation_count =
      sizeof(implementations) / sizeof(implementations[0]);

  printf("============================================================\n");
  printf(" GPU Reduction Benchmark (N = %zu)\n", N);
  printf(" Device: %s (SM %d.%d, %d SMs)\n", properties.name,
         properties.major, properties.minor, properties.multiProcessorCount);
  printf(" Timing: median of %d runs after %d warmups (kernel only)\n",
         runs, warmups);
  printf("============================================================\n\n");
  printf("%-38s | %10s | %11s | %10s | %12s\n", "Implementation",
         "Time (ms)", "GElements/s", "Input GB/s", "Verification");
  printf("--------------------------------------------------------------------------"
         "-------------\n");

  for (implementation_index = 0;
       implementation_index < implementation_count; ++implementation_index) {
    ReductionImpl *implementation = &implementations[implementation_index];
    float *samples;
    float result = 0.0f;
    float milliseconds;
    double seconds;
    double gelements;
    double input_gbytes;
    double absolute_error = 0.0;
    double relative_error = 0.0;
    int passed;
    int i;

    if (!implementation->supports(N)) {
      printf("%-38s | %10s | %11s | %10s | SKIP (%s)\n",
             implementation->name, "-", "-", "-",
             implementation->size_requirement);
      continue;
    }

    for (i = 0; i < warmups; ++i) {
      prepare_run(input_d, input_h, bytes, output_d, stream);
      CUDA_CHECK(implementation->launch(input_d, output_d, N, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    samples = (float *)malloc((size_t)runs * sizeof(float));
    if (samples == NULL) {
      fprintf(stderr, "Timing allocation failed\n");
      exit(EXIT_FAILURE);
    }

    for (i = 0; i < runs; ++i) {
      prepare_run(input_d, input_h, bytes, output_d, stream);
      cuda_event_timer_start(&timer, stream);
      CUDA_CHECK(implementation->launch(input_d, output_d, N, stream));
      samples[i] = cuda_event_timer_stop(&timer, stream);
    }

    CUDA_CHECK(cudaMemcpy(&result, output_d, sizeof(float),
                          cudaMemcpyDeviceToHost));
    passed = verify(reference, result, &absolute_error, &relative_error);
    all_passed = all_passed && passed;

    milliseconds = median_ms(samples, runs);
    seconds = milliseconds * 1e-3;
    gelements = (double)N / seconds / 1e9;
    // Lower-bound effective bandwidth based on one read of the input.
    input_gbytes = (double)bytes / seconds / 1e9;

    if (passed) {
      printf("%-38s | %10.4f | %11.3f | %10.2f | PASS (%.2e rel)\n",
             implementation->name, milliseconds, gelements, input_gbytes,
             relative_error);
    } else {
      printf("%-38s | %10.4f | %11.3f | %10.2f | FAIL (%.2e abs)\n",
             implementation->name, milliseconds, gelements, input_gbytes,
             absolute_error);
    }
    free(samples);
  }

  printf("--------------------------------------------------------------------------"
         "-------------\n");

  cuda_event_timer_destroy(&timer);
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(output_d));
  CUDA_CHECK(cudaFree(input_d));
  free(input_h);

  return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
