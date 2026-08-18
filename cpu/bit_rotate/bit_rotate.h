#ifndef BIT_ROTATE_H
#define BIT_ROTATE_H

#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>

// In-place bit-range rotation over a packed bit buffer.
// buf: bits packed 8 per byte, LSB-first (bit i = bit i%8 of byte i/8)
// bit_offset: index of the first bit of the range to rotate
// bit_length: number of bits in the range
// bit_right_amount: signed rotation distance (negative rotates left)
// Bits outside [bit_offset, bit_offset + bit_length) must not change.
// Accessors derive from the MIT-licensed 6.172 everybit starter code.
typedef void (*bit_rotate_fn)(unsigned char *buf, size_t bit_offset,
                              size_t bit_length, ssize_t bit_right_amount);

typedef struct {
  const char *name;
  bit_rotate_fn fn;
} BitRotateImpl;

// Packed-bit accessors shared by all implementations.
static inline bool bit_get(const unsigned char *buf, size_t bit_index) {
  return (buf[bit_index / 8] >> (bit_index % 8)) & 1;
}

static inline void bit_set(unsigned char *buf, size_t bit_index, bool value) {
  const unsigned char mask = (unsigned char)(1u << (bit_index % 8));
  buf[bit_index / 8] =
      (unsigned char)((buf[bit_index / 8] & ~mask) | (value ? mask : 0));
}

// Implementation prototypes
void bit_rotate_01(unsigned char *buf, size_t bit_offset, size_t bit_length,
                   ssize_t bit_right_amount);

#endif // BIT_ROTATE_H
