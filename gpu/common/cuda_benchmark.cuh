#pragma once

#include "cuda_check.cuh"

typedef struct {
  cudaEvent_t start;
  cudaEvent_t stop;
} CudaEventTimer;

static inline void cuda_event_timer_create(CudaEventTimer *timer) {
  CUDA_CHECK(cudaEventCreate(&timer->start));
  CUDA_CHECK(cudaEventCreate(&timer->stop));
}

static inline void cuda_event_timer_destroy(CudaEventTimer *timer) {
  CUDA_CHECK(cudaEventDestroy(timer->start));
  CUDA_CHECK(cudaEventDestroy(timer->stop));
}

static inline void cuda_event_timer_start(CudaEventTimer *timer,
                                          cudaStream_t stream) {
  CUDA_CHECK(cudaEventRecord(timer->start, stream));
}

static inline float cuda_event_timer_stop(CudaEventTimer *timer,
                                          cudaStream_t stream) {
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventRecord(timer->stop, stream));
  CUDA_CHECK(cudaEventSynchronize(timer->stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer->start, timer->stop));
  return elapsed_ms;
}
