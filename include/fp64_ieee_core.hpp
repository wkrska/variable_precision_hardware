#pragma once

#include <cstdint>
#include "fpu_types.hpp"
#include "bit_ops.hpp"

namespace vph {

/**
 * @brief Unpacked FP64 Representation for Bit-Level ALU
 */
struct Fp64Unpacked {
    uint8_t  sign;          ///< 1-bit sign: 0 = positive, 1 = negative
    int16_t  exp;           ///< 11-bit biased exponent (0..2047)
    uint64_t mant;          ///< 53-bit significand with implicit hidden bit at bit 52
    bool     is_zero;       ///< Operand is +/- 0.0
    bool     is_subnormal;  ///< Exponent == 0 and fraction != 0
    bool     is_inf;        ///< Exponent == 2047 and fraction == 0
    bool     is_nan;        ///< Exponent == 2047 and fraction != 0
};

/**
 * @brief Convert an 8-byte array to an unsigned 64-bit integer (little-endian).
 */
inline uint64_t bytes_to_u64(const uint8_t b[8]) {
    return static_cast<uint64_t>(b[0]) |
          (static_cast<uint64_t>(b[1]) << 8) |
          (static_cast<uint64_t>(b[2]) << 16) |
          (static_cast<uint64_t>(b[3]) << 24) |
          (static_cast<uint64_t>(b[4]) << 32) |
          (static_cast<uint64_t>(b[5]) << 40) |
          (static_cast<uint64_t>(b[6]) << 48) |
          (static_cast<uint64_t>(b[7]) << 56);
}

/**
 * @brief Pack an unsigned 64-bit integer into an 8-byte array (little-endian).
 */
inline void u64_to_bytes(uint64_t val, uint8_t b[8]) {
    b[0] = static_cast<uint8_t>(val & 0xFF);
    b[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    b[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    b[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
    b[4] = static_cast<uint8_t>((val >> 32) & 0xFF);
    b[5] = static_cast<uint8_t>((val >> 40) & 0xFF);
    b[6] = static_cast<uint8_t>((val >> 48) & 0xFF);
    b[7] = static_cast<uint8_t>((val >> 56) & 0xFF);
}

/**
 * @brief Bit-cast helpers for host double conversions.
 */
inline double u64_to_double(uint64_t u) {
    union { uint64_t u; double d; } c;
    c.u = u;
    return c.d;
}

inline uint64_t double_to_u64(double d) {
    union { double d; uint64_t u; } c;
    c.d = d;
    return c.u;
}

/**
 * @brief Unpack raw IEEE-754 binary64 word into constituent bit fields.
 */
inline Fp64Unpacked fp64_unpack(uint64_t raw) {
    Fp64Unpacked u;
    u.sign = static_cast<uint8_t>((raw >> 63) & 0x1);
    u.exp  = static_cast<int16_t>((raw >> 52) & 0x7FF);
    uint64_t frac = raw & 0x000FFFFFFFFFFFFFULL;

    u.is_nan       = (u.exp == 0x7FF) && (frac != 0);
    u.is_inf       = (u.exp == 0x7FF) && (frac == 0);
    u.is_zero      = (u.exp == 0) && (frac == 0);
    u.is_subnormal = (u.exp == 0) && (frac != 0);

    if (u.exp == 0) {
        u.mant = frac;
        u.exp  = 1;
    } else {
        u.mant = (1ULL << 52) | frac;
    }

    return u;
}

/**
 * @brief Pack sign, exponent, and fraction into an IEEE-754 binary64 word.
 */
inline uint64_t fp64_pack(uint8_t sign, int16_t exp, uint64_t frac) {
    uint64_t raw = (static_cast<uint64_t>(sign & 0x1) << 63);
    raw |= (static_cast<uint64_t>(exp & 0x7FF) << 52);
    raw |= (frac & 0x000FFFFFFFFFFFFFULL);
    return raw;
}

constexpr uint64_t FP64_POS_ZERO = 0x0000000000000000ULL;
constexpr uint64_t FP64_NEG_ZERO = 0x8000000000000000ULL;
constexpr uint64_t FP64_POS_INF  = 0x7FF0000000000000ULL;
constexpr uint64_t FP64_NEG_INF  = 0xFFF0000000000000ULL;
constexpr uint64_t FP64_QNAN     = 0x7FF8000000000000ULL;

/**
 * @brief Core Bit-Level Combinational FP64 ALU
 */
struct Fp64AluResult {
    uint64_t result;
    uint64_t residual;
    uint8_t  status;
};

inline Fp64AluResult fp64_eval_alu(uint64_t raw_a, uint64_t raw_b, uint8_t opcode) {
    Fp64AluResult out{0, 0, FpuStatus::OK};

    // Unary Operations (ABS, NEG)
    if (opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        out.result = raw_a & 0x7FFFFFFFFFFFFFFFULL;
        return out;
    }
    if (opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        out.result = raw_a ^ 0x8000000000000000ULL;
        return out;
    }

    Fp64Unpacked a = fp64_unpack(raw_a);
    Fp64Unpacked b = fp64_unpack(raw_b);

    if (opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    // MULTIPLICATION (MUL, MUL_HILO)
    if (opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;

        if (a.is_nan || b.is_nan) {
            out.result = FP64_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
            out.result = FP64_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if (a.is_inf || b.is_inf) {
            out.result = (res_sign ? FP64_NEG_INF : FP64_POS_INF);
            out.status = FpuStatus::IS_INF;
            return out;
        }
        if (a.is_zero || b.is_zero) {
            out.result = (res_sign ? FP64_NEG_ZERO : FP64_POS_ZERO);
            out.status = FpuStatus::IS_ZERO;
            return out;
        }

        // 53x53-bit multiplication decomposed into four 27/26-bit partial products
        uint64_t ah = a.mant >> 26;
        uint64_t al = a.mant & 0x3FFFFFFULL;
        uint64_t bh = b.mant >> 26;
        uint64_t bl = b.mant & 0x3FFFFFFULL;

        uint64_t pll = al * bl;
        uint64_t phl = ah * bl;
        uint64_t plh = al * bh;
        uint64_t phh = ah * bh;

        uint64_t mid_sum  = phl + plh + (pll >> 26);
        uint64_t high_sum = phh + (mid_sum >> 26);
        uint64_t low_bits = (pll & 0x3FFFFFFULL) | ((mid_sum & 0x3FFFFFFULL) << 26);

        int16_t exp_res = a.exp + b.exp - 1023;

        uint64_t mant_res;
        uint8_t guard, round, sticky;

        if (high_sum & (1ULL << 53)) {
            // Product in [2.0, 4.0): bit 105 is 1
            exp_res += 1;
            mant_res = high_sum >> 1; // 53 bits
            guard    = static_cast<uint8_t>(high_sum & 1);
            round    = static_cast<uint8_t>((low_bits >> 51) & 1);
            sticky   = (low_bits & 0x7FFFFFFFFFFFFULL) ? 1 : 0;
        } else {
            // Product in [1.0, 2.0): bit 104 is 1
            mant_res = high_sum & 0x001FFFFFFFFFFFFFULL; // 53 bits
            guard    = static_cast<uint8_t>((low_bits >> 51) & 1);
            round    = static_cast<uint8_t>((low_bits >> 50) & 1);
            sticky   = (low_bits & 0x3FFFFFFFFFFFFULL) ? 1 : 0;
        }

        bool lsb = (mant_res & 1);
        bool round_up = guard && (round || sticky || lsb);
        if (round_up) {
            mant_res++;
            if (mant_res & (1ULL << 53)) {
                mant_res >>= 1;
                exp_res++;
            }
        }

        if (guard || round || sticky) {
            out.status |= FpuStatus::INEXACT;
        }

        if (exp_res >= 2047) {
            out.result = (res_sign ? FP64_NEG_INF : FP64_POS_INF);
            out.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
        } else if (exp_res <= 0) {
            out.result = (res_sign ? FP64_NEG_ZERO : FP64_POS_ZERO);
            out.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
        } else {
            out.result = fp64_pack(res_sign, exp_res, mant_res & 0x000FFFFFFFFFFFFFULL);
        }

        return out;
    }

    // ADDITION / SUBTRACTION
    if (a.is_nan || b.is_nan) {
        out.result = FP64_QNAN;
        out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return out;
    }
    if (a.is_inf && b.is_inf) {
        if (a.sign != b.sign) {
            out.result = FP64_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        } else {
            out.result = (a.sign ? FP64_NEG_INF : FP64_POS_INF);
            out.status = FpuStatus::IS_INF;
        }
        return out;
    }
    if (a.is_inf) {
        out.result = (a.sign ? FP64_NEG_INF : FP64_POS_INF);
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (b.is_inf) {
        out.result = (b.sign ? FP64_NEG_INF : FP64_POS_INF);
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (a.is_zero && b.is_zero) {
        uint8_t res_sign = (a.sign && b.sign) ? 1 : 0;
        out.result = (res_sign ? FP64_NEG_ZERO : FP64_POS_ZERO);
        out.status = FpuStatus::IS_ZERO;
        return out;
    }
    if (a.is_zero) {
        out.result = fp64_pack(b.sign, b.exp, b.mant & 0x000FFFFFFFFFFFFFULL);
        return out;
    }
    if (b.is_zero) {
        out.result = fp64_pack(a.sign, a.exp, a.mant & 0x000FFFFFFFFFFFFFULL);
        return out;
    }

    bool a_is_larger = false;
    if (a.exp > b.exp) {
        a_is_larger = true;
    } else if (a.exp == b.exp) {
        a_is_larger = (a.mant >= b.mant);
    }

    Fp64Unpacked large = a_is_larger ? a : b;
    Fp64Unpacked small = a_is_larger ? b : a;
    bool eff_sub = (large.sign != small.sign);

    int16_t exp_diff = large.exp - small.exp;

    // Extend 53-bit significands by 3 bits: [Guard, Round, Sticky] -> 56 bits (fits in uint64_t)
    uint64_t mant_large_ext = large.mant << 3;
    uint64_t mant_small_ext = small.mant << 3;

    uint8_t sticky = 0;
    if (exp_diff > 0) {
        if (exp_diff >= 60) {
            sticky = (small.mant != 0) ? 1 : 0;
            mant_small_ext = 0;
        } else {
            uint64_t mask = (1ULL << exp_diff) - 1;
            sticky = ((small.mant & mask) != 0) ? 1 : 0;
            mant_small_ext >>= exp_diff;
        }
    }
    mant_small_ext |= sticky;

    uint64_t mant_res;
    if (eff_sub) {
        mant_res = mant_large_ext - mant_small_ext;
    } else {
        mant_res = mant_large_ext + mant_small_ext;
    }

    if (mant_res == 0) {
        out.result = FP64_POS_ZERO;
        out.status = FpuStatus::IS_ZERO;
        return out;
    }

    int16_t exp_res = large.exp;

    if (!eff_sub) {
        // Carry out into bit 56
        if (mant_res & (1ULL << 56)) {
            sticky |= static_cast<uint8_t>(mant_res & 1);
            mant_res >>= 1;
            exp_res++;
        }
    } else {
        uint64_t upper = mant_res >> 3;
        if (upper != 0) {
            uint8_t lz = clz64(upper) - 11; // Bit 52 is target leading 1
            if (lz > 0) {
                if (exp_res > lz) {
                    mant_res <<= lz;
                    exp_res -= lz;
                } else {
                    mant_res <<= (exp_res - 1);
                    exp_res = 0;
                }
            }
        }
    }

    uint64_t final_mant = mant_res >> 3; // 53-bit significand
    uint8_t g = static_cast<uint8_t>((mant_res >> 2) & 1);
    uint8_t r = static_cast<uint8_t>((mant_res >> 1) & 1);
    uint8_t s = static_cast<uint8_t>(mant_res & 1) | sticky;

    bool round_up = g && (r || s || (final_mant & 1));
    if (round_up) {
        final_mant++;
        if (final_mant & (1ULL << 53)) {
            final_mant >>= 1;
            exp_res++;
        }
    }

    if (g || r || s) {
        out.status |= FpuStatus::INEXACT;
    }

    if (exp_res >= 2047) {
        out.result = (large.sign ? FP64_NEG_INF : FP64_POS_INF);
        out.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
    } else if (exp_res <= 0) {
        out.result = (large.sign ? FP64_NEG_ZERO : FP64_POS_ZERO);
        out.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
    } else {
        out.result = fp64_pack(large.sign, exp_res, final_mant & 0x000FFFFFFFFFFFFFULL);
    }

    return out;
}

} // namespace vph
