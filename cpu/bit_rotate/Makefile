CC ?= gcc
CFLAGS ?= -O3 -march=native -mprefer-vector-width=512 -Wall -Wextra -std=c99
LDFLAGS ?=

LIMIT ?= 1.0

TARGET = main
SRCS = main.c $(wildcard bit_rotate_*.c)
OBJS = $(SRCS:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c bit_rotate.h
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(LIMIT)

clean:
	rm -f $(OBJS) $(TARGET)
