CC ?= gcc
CFLAGS ?= -O3 -march=native -mprefer-vector-width=512 -Wall -Wextra -std=c99
LDFLAGS ?= -lm

N ?= 512

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
	./$(TARGET) $(N)

clean:
	rm -f $(OBJS) $(TARGET)
