/**
 * bit_rotate_01: the naive baseline — rotate left by one, repeated.
 * O(bit_length * amount) bit operations.
 *
 * Adapted from the MIT 6.172 "everybit" starter code:
 * Copyright (c) 2012 MIT License by 6.172 Staff
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/

#include "bit_rotate.h"

#include <assert.h>

// Portable modulo that is nonnegative for negative dividends, unlike C's %.
static size_t modulo(const ssize_t n, const size_t m) {
  const ssize_t signed_m = (ssize_t)m;
  assert(signed_m > 0);
  const ssize_t result = ((n % signed_m) + signed_m) % signed_m;
  assert(result >= 0);
  return (size_t)result;
}

// Grab the first bit in the range, shift everything left by one, and stick
// the first bit at the end.
static void rotate_left_one(unsigned char *buf, const size_t bit_offset,
                            const size_t bit_length) {
  const bool first_bit = bit_get(buf, bit_offset);
  size_t i;
  for (i = bit_offset; i + 1 < bit_offset + bit_length; i++) {
    bit_set(buf, i, bit_get(buf, i + 1));
  }
  bit_set(buf, i, first_bit);
}

void bit_rotate_01(unsigned char *buf, const size_t bit_offset,
                   const size_t bit_length, const ssize_t bit_right_amount) {
  if (bit_length == 0) {
    return;
  }

  // Convert any rotation into an equivalent left rotation and eliminate
  // full rotations.
  const size_t bit_left_amount = modulo(-bit_right_amount, bit_length);
  for (size_t i = 0; i < bit_left_amount; i++) {
    rotate_left_one(buf, bit_offset, bit_length);
  }
}
