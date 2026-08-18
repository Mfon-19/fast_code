# bit_rotate

Rotate a subrange of a packed bit array, in place.

Extracted from MIT 6.172 (Performance Engineering) Project 1, "everybit"
(MIT-licensed staff code; naive baseline kept, all optimization work removed).

## Problem

A bit array packs bits 8 per byte: bit `i` is bit `i % 8` (LSB-first) of byte
`i / 8`. Given a buffer, a `bit_offset`, a `bit_length`, and a signed
`bit_right_amount`, cyclically rotate the `bit_length`-bit subrange starting at
`bit_offset` right by `bit_right_amount` bits (negative = rotate left). Every
bit outside the subrange must be left untouched. Nothing is byte- or
word-aligned: offset, length, and amount are all arbitrary bit counts.

Correct means: buffer contents identical to what the reference implementation
(`bit_rotate_01`, rotate-left-by-one repeated) produces, for any parameters.

## Benchmark

The harness climbs a tier ladder with adversarial parameters: tier `t` rotates
with `offset = Fib(t)`, `amount = Fib(t+1)`, `length = Fib(t+2)` inside a
buffer of `Fib(t+3)` bits prefilled with random data. One rotation is timed
per tier; a tier passes if it finishes within the time limit (default 1.0s).
Score = highest passing tier. The Fibonacci family is deliberately worst-case:
`gcd(length, amount) = 1` and the split is the golden ratio.

```
make run                 # 1.0s limit, buffers capped at 2 GB
./main 0.1               # custom time limit
./main 1.0 4096          # custom buffer cap (MB)
```

## Rules variants

- **classic** (original MIT 6.172 rules): single thread, standard C only (no
  intrinsics, no asm), O(1) extra memory (small stack/BSS buffers only).
- **unrestricted CPU**: anything goes.
- **gpu**: same contract on a device buffer; time the kernel only.

## Layout

- `bit_rotate.h` — contract + packed-bit accessors
- `bit_rotate_01.c` — reference/naive baseline, O(n·k)
- `main.c` — randomized correctness check vs the reference + tier ladder
