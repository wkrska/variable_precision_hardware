#pragma once

#include <cstdint>
#include <cstddef>

namespace vph {

/**
 * @brief Convert a 4-byte array to an unsigned 32-bit integer (little-endian).
 */
inline uint32_t bytes_to_u32(const uint8_t b[4]) {
    return static_cast<uint32_t>(b[0]) |
          (static_cast<uint32_t>(b[1]) << 8) |
          (static_cast<uint32_t>(b[2]) << 16) |
          (static_cast<uint32_t>(b[3]) << 24);
}

/**
 * @brief Pack an unsigned 32-bit integer into a 4-byte array (little-endian).
 */
inline void u32_to_bytes(uint32_t val, uint8_t b[4]) {
    b[0] = static_cast<uint8_t>(val & 0xFF);
    b[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    b[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    b[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
}

/**
 * @brief Synthesizable, branchless Count Leading Zeros (CLZ) for 32-bit words.
 *
 * Implements a 5-level binary priority tree that synthesizes directly into
 * efficient FPGA ALM/LUT logic.
 */
inline uint8_t clz32(uint32_t x) {
    if (x == 0) return 32;
    uint8_t n = 0;
    if ((x & 0xFFFF0000) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF000000) == 0) { n += 8;  x <<= 8;  }
    if ((x & 0xF0000000) == 0) { n += 4;  x <<= 4;  }
    if ((x & 0xC0000000) == 0) { n += 2;  x <<= 2;  }
    if ((x & 0x80000000) == 0) { n += 1; }
    return n;
}

/**
 * @brief Synthesizable, branchless Count Leading Zeros (CLZ) for 64-bit words.
 */
inline uint8_t clz64(uint64_t x) {
    if (x == 0) return 64;
    uint32_t hi = static_cast<uint32_t>(x >> 32);
    if (hi != 0) {
        return clz32(hi);
    } else {
        return 32 + clz32(static_cast<uint32_t>(x & 0xFFFFFFFFULL));
    }
}

/**
 * @brief Explicit bit-cast helpers for host/verification use.
 */
inline float u32_to_float(uint32_t u) {
    union { uint32_t u; float f; } c;
    c.u = u;
    return c.f;
}

inline uint32_t float_to_u32(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    return c.u;
}

} // namespace vph
