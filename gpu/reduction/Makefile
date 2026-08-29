NVCC ?= nvcc
NVCCFLAGS ?= -O3 -lineinfo -Xcompiler=-Wall,-Wextra
GPU_ARCH ?= native

N ?= 2048
RUNS ?= 10
WARMUPS ?= 3

TARGET := main
SRCS := main.cu launchers.cu $(wildcard reduction_*.cu)
OBJS := $(SRCS:.cu=.o)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -arch=$(GPU_ARCH) $^ -o $@

%.o: %.cu reduction.h ../common/cuda_check.cuh ../common/cuda_benchmark.cuh
	$(NVCC) $(NVCCFLAGS) -arch=$(GPU_ARCH) -I../common -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(N) $(RUNS) $(WARMUPS)

clean:
	rm -f $(OBJS) $(TARGET)
