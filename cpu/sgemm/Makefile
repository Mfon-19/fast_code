CC ?= gcc
CFLAGS ?= -O3 -march=native -mprefer-vector-width=512 -Wall -Wextra -std=c99 -fopenmp
LDFLAGS ?= -lm

ONEAPI_MKL_ROOT ?= $(if $(MKLROOT),$(MKLROOT),$(HOME)/intel/oneapi/mkl/latest)
ONEAPI_MKL_PC := $(ONEAPI_MKL_ROOT)/lib/pkgconfig
MKL_PKG_CONFIG := PKG_CONFIG_PATH="$(ONEAPI_MKL_PC):$(PKG_CONFIG_PATH)" pkg-config

MKL_PKG ?= $(shell $(MKL_PKG_CONFIG) --exists mkl-dynamic-lp64-gomp && \
	echo mkl-dynamic-lp64-gomp || \
	($(MKL_PKG_CONFIG) --exists mkl-sdl-lp64 && echo mkl-sdl-lp64))
MKL_CFLAGS ?= $(shell test -n "$(MKL_PKG)" && $(MKL_PKG_CONFIG) --cflags $(MKL_PKG))
MKL_LIBS ?= $(shell test -n "$(MKL_PKG)" && $(MKL_PKG_CONFIG) --libs $(MKL_PKG))

ifneq ($(strip $(MKL_LIBS)),)
CFLAGS += -DHAVE_MKL $(MKL_CFLAGS)
LDFLAGS += $(MKL_LIBS)
ifneq ($(wildcard $(ONEAPI_MKL_ROOT)/lib/libmkl_core.so),)
LDFLAGS += -Wl,-rpath,$(ONEAPI_MKL_ROOT)/lib
endif
endif

N ?= 512
THREADS ?= 4

TARGET = main
SRCS = main.c $(wildcard sgemm_*.c)
OBJS = $(SRCS:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c sgemm.h
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE OMP_NUM_THREADS=$(THREADS) \
	OMP_PLACES=cores OMP_PROC_BIND=close ./$(TARGET) $(N)

clean:
	rm -f $(OBJS) $(TARGET)
