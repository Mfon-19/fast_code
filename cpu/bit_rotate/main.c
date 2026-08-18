#define _POSIX_C_SOURCE 200809L

#include "bit_rotate.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_TIME_LIMIT 1.0
#define DEFAULT_MAX_MB 2048
#define VERIFY_CASES 200
#define VERIFY_MAX_BITS 1024

// Measured single-core DRAM ceiling on 11th Gen i5 (in-place read+writeback
// streams).  The GB/s column charges the 2*bytes minimum-traffic floor (every
// bit must be read once and written once).  Adjust per machine.
#define PEAK_GBPS 40.0

// Tier t rotates offset=fib[t], amount=fib[t+1], length=fib[t+2] in a buffer
// of fib[t+3] bits.
static const uint64_t fibs[53] = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233,
    377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025,
    121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578,
    5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155,
    165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903,
    2971215073, 4807526976, 7778742049, 12586269025, 20365011074, 32951280099,
    53316291173, 86267571272};

static double get_time_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t rng_state;
static uint64_t xrand(void) {
  uint64_t x = rng_state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return rng_state = x;
}

static void fill_random(unsigned char *buf, size_t bytes) {
  size_t i = 0;
  for (; i + 8 <= bytes; i += 8) {
    uint64_t r = xrand();
    memcpy(buf + i, &r, 8);
  }
  for (; i < bytes; i++) {
    buf[i] = (unsigned char)xrand();
  }
}

// Random small rotations compared bit-for-bit (whole buffer, so bits outside
// the range are checked too) against the reference implementation.
static int verify_impl(const BitRotateImpl *impl, const BitRotateImpl *ref) {
  int failures = 0;
  for (int c = 0; c < VERIFY_CASES; c++) {
    rng_state = 0x9e3779b97f4a7c15ULL + c;
    size_t sz = 1 + xrand() % VERIFY_MAX_BITS;
    size_t bytes = (sz + 7) / 8;
    unsigned char *a = malloc(bytes);
    unsigned char *b = malloc(bytes);
    fill_random(a, bytes);
    memcpy(b, a, bytes);
    size_t off = xrand() % sz;
    size_t len = xrand() % (sz - off + 1);
    ssize_t amount = (ssize_t)(xrand() % (3 * sz + 1)) - (ssize_t)(3 * sz / 2);
    ref->fn(a, off, len, amount);
    impl->fn(b, off, len, amount);
    if (memcmp(a, b, bytes) != 0) {
      if (failures == 0) {
        printf("    first failure: sz=%zu off=%zu len=%zu amount=%zd\n", sz,
               off, len, amount);
      }
      failures++;
    }
    free(a);
    free(b);
  }
  return failures;
}

static void run_ladder(const BitRotateImpl *impl, double time_limit,
                       uint64_t max_bytes) {
  printf("%s\n", impl->name);
  int last_passed = -1;
  for (int t = 0; t + 3 < 53; t++) {
    uint64_t bit_sz = fibs[t + 3];
    uint64_t bytes = (bit_sz + 7) / 8;
    if (bytes > max_bytes) {
      printf("  stopped at tier %d: buffer %.0f MB exceeds cap\n", t,
             bytes / 1e6);
      break;
    }
    unsigned char *buf = calloc(1, bytes);
    if (buf == NULL) {
      printf("  stopped at tier %d: allocation failed\n", t);
      break;
    }
    rng_state = 6172;
    fill_random(buf, bytes);

    double t0 = get_time_sec();
    impl->fn(buf, fibs[t], fibs[t + 2], (ssize_t)fibs[t + 1]);
    double elapsed = get_time_sec() - t0;
    free(buf);

    bool passed = elapsed < time_limit;
    if (elapsed >= 1e-3 || !passed) {
      double gbps = 2.0 * (fibs[t + 2] / 8.0) / elapsed / 1e9;
      printf("  tier %2d (%8.2f MB) %10.4f ms  %6.1f GB/s  %5.1f%% peak  %s\n",
             t, fibs[t + 2] / 8e6, elapsed * 1e3, gbps,
             100.0 * gbps / PEAK_GBPS, passed ? "pass" : "FAIL");
    }
    if (!passed) {
      break;
    }
    last_passed = t;
  }
  printf("  => highest tier passed: %d\n\n", last_passed);
}

int main(int argc, char **argv) {
  double time_limit = argc > 1 ? atof(argv[1]) : DEFAULT_TIME_LIMIT;
  uint64_t max_bytes =
      (argc > 2 ? (uint64_t)atoll(argv[2]) : DEFAULT_MAX_MB) << 20;
  if (time_limit <= 0) {
    fprintf(stderr, "Invalid time limit\n");
    return EXIT_FAILURE;
  }

  BitRotateImpl implementations[] = {
      {"bit_rotate_01 (Base Naive)", bit_rotate_01},
      // Future implementations go here:
      // {"bit_rotate_02 (...)", bit_rotate_02},
  };
  int num_impls = sizeof(implementations) / sizeof(implementations[0]);

  printf("=========================================================\n");
  printf(" bit_rotate benchmark (limit %.3fs, buffers <= %llu MB)\n",
         time_limit, (unsigned long long)(max_bytes >> 20));
  printf("=========================================================\n\n");

  printf("Correctness (%d randomized cases vs %s):\n", VERIFY_CASES,
         implementations[0].name);
  for (int i = 0; i < num_impls; i++) {
    if (i == 0) {
      printf("  %-32s REF\n", implementations[i].name);
      continue;
    }
    int fails = verify_impl(&implementations[i], &implementations[0]);
    printf("  %-32s %s (%d/%d)\n", implementations[i].name,
           fails ? "FAIL" : "PASS", VERIFY_CASES - fails, VERIFY_CASES);
  }
  printf("\nTier ladder:\n");
  for (int i = 0; i < num_impls; i++) {
    run_ladder(&implementations[i], time_limit, max_bytes);
  }
  return EXIT_SUCCESS;
}
