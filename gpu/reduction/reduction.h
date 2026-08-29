#pragma once

#include <cuda_runtime.h>

#include <stddef.h>

typedef cudaError_t (*ReductionLaunchFn)(float *input, float *output, size_t N,
                                         cudaStream_t stream);
typedef int (*ReductionSupportsFn)(size_t N);

typedef struct {
  const char *name;
  ReductionLaunchFn launch;
  ReductionSupportsFn supports;
  const char *size_requirement;
} ReductionImpl;

cudaError_t launch_reduction_01(float *, float *, size_t, cudaStream_t);
cudaError_t launch_reduction_02(float *, float *, size_t, cudaStream_t);
cudaError_t launch_reduction_03(float *, float *, size_t, cudaStream_t);
cudaError_t launch_reduction_04(float *, float *, size_t, cudaStream_t);
cudaError_t launch_reduction_05(float *, float *, size_t, cudaStream_t);

int supports_2048(size_t N);
int supports_multiple_2048(size_t N);
int supports_multiple_6144(size_t N);
