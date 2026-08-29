BENCH ?= reduction
N ?= 2048
RUNS ?= 10
WARMUPS ?= 3

.PHONY: all run clean

all:
	$(MAKE) -C $(BENCH)

run:
	$(MAKE) -C $(BENCH) run N=$(N) RUNS=$(RUNS) WARMUPS=$(WARMUPS)

clean:
	$(MAKE) -C $(BENCH) clean
