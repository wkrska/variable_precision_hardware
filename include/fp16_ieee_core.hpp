#pragma once

#include <cstdint>
#include "fpu_types.hpp"
#include "bit_ops.hpp"

namespace vph {

/**
 * @brief Convert a 2-byte array to an unsigned 16-bit integer (little-endian).
 */
inline uint16_t bytes_to_u16(const uint8_t b[2]) {
    return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}

/**
 * @brief Pack an unsigned 16-bit integer into a 2-byte array (little-endian).
 */
inline void u16_to_bytes(uint16_t val, uint8_t b[2]) {
    b[0] = static_cast<uint8_t>(val & 0xFF);
    b[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
}

/**
 * @brief Unpacked FP16 Representation for Bit-Level ALU
 */
struct Fp16Unpacked {
    uint8_t  sign;          ///< 1-bit sign: 0 = positive, 1 = negative
    int16_t  exp;           ///< 5-bit biased exponent (0..31)
    uint16_t mant;          ///< 11-bit significand with implicit hidden bit at bit 10
    bool     is_zero;       ///< Operand is +/- 0.0
    bool     is_subnormal;  ///< Exponent == 0 and fraction != 0
    bool     is_inf;        ///< Exponent == 31 and fraction == 0
    bool     is_nan;        ///< Exponent == 31 and fraction != 0
};

inline Fp16Unpacked fp16_unpack(uint16_t raw) {
    Fp16Unpacked u;
    u.sign = static_cast<uint8_t>((raw >> 15) & 0x1);
    u.exp  = static_cast<int16_t>((raw >> 10) & 0x1F);
    uint16_t frac = raw & 0x3FF;

    u.is_nan       = (u.exp == 0x1F) && (frac != 0);
    u.is_inf       = (u.exp == 0x1F) && (frac == 0);
    u.is_zero      = (u.exp == 0) && (frac == 0);
    u.is_subnormal = (u.exp == 0) && (frac != 0);

    if (u.exp == 0) {
        u.mant = frac;
        u.exp  = 1;
    } else {
        u.mant = (1U << 10) | frac;
    }

    return u;
}

inline uint16_t fp16_pack(uint8_t sign, int16_t exp, uint16_t frac) {
    uint16_t raw = (static_cast<uint16_t>(sign & 0x1) << 15);
    raw |= (static_cast<uint16_t>(exp & 0x1F) << 10);
    raw |= (frac & 0x3FF);
    return raw;
}

constexpr uint16_t FP16_POS_ZERO = 0x0000;
constexpr uint16_t FP16_NEG_ZERO = 0x8000;
constexpr uint16_t FP16_POS_INF  = 0x7C00;
constexpr uint16_t FP16_NEG_INF  = 0xFC00;
constexpr uint16_t FP16_QNAN     = 0x7E00;

struct Fp16AluResult {
    uint16_t result;
    uint16_t residual;
    uint8_t  status;
};

inline Fp16AluResult fp16_eval_alu(uint16_t raw_a, uint16_t raw_b, uint8_t opcode) {
    Fp16AluResult out{0, 0, FpuStatus::OK};

    if (opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        out.result = raw_a & 0x7FFF;
        return out;
    }
    if (opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        out.result = raw_a ^ 0x8000;
        return out;
    }

    Fp16Unpacked a = fp16_unpack(raw_a);
    Fp16Unpacked b = fp16_unpack(raw_b);

    if (opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    // MULTIPLICATION
    if (opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;

        if (a.is_nan || b.is_nan) {
            out.result = FP16_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
            out.result = FP16_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if (a.is_inf || b.is_inf) {
            out.result = (res_sign ? FP16_NEG_INF : FP16_POS_INF);
            out.status = FpuStatus::IS_INF;
            return out;
        }
        if (a.is_zero || b.is_zero) {
            out.result = (res_sign ? FP16_NEG_ZERO : FP16_POS_ZERO);
            out.status = FpuStatus::IS_ZERO;
            return out;
        }

        // 11x11-bit mantissa multiplication -> 22 bits
        uint32_t prod = static_cast<uint32_t>(a.mant) * static_cast<uint32_t>(b.mant);
        int16_t exp_res = a.exp + b.exp - 15;

        uint16_t mant_res;
        uint8_t guard, round, sticky;

        if (prod & (1U << 21)) {
            // Product in [2.0, 4.0): bit 21 is 1
            exp_res += 1;
            mant_res = static_cast<uint16_t>(prod >> 11); // 11 bits
            guard    = static_cast<uint8_t>((prod >> 10) & 1);
            round    = static_cast<uint8_t>((prod >> 9) & 1);
            sticky   = (prod & 0x1FF) ? 1 : 0;
        } else {
            // Product in [1.0, 2.0): bit 20 is 1
            mant_res = static_cast<uint16_t>(prod >> 10); // 11 bits
            guard    = static_cast<uint8_t>((prod >> 9) & 1);
            round    = static_cast<uint8_t>((prod >> 8) & 1);
            sticky   = (prod & 0xFF) ? 1 : 0;
        }

        bool lsb = (mant_res & 1);
        bool round_up = guard && (round || sticky || lsb);
        if (round_up) {
            mant_res++;
            if (mant_res & (1U << 11)) {
                mant_res >>= 1;
                exp_res++;
            }
        }

        if (guard || round || sticky) {
            out.status |= FpuStatus::INEXACT;
        }

        if (exp_res >= 31) {
            out.result = (res_sign ? FP16_NEG_INF : FP16_POS_INF);
            out.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
        } else if (exp_res <= 0) {
            out.result = (res_sign ? FP16_NEG_ZERO : FP16_POS_ZERO);
            out.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
        } else {
            out.result = fp16_pack(res_sign, exp_res, mant_res & 0x3FF);
        }

        return out;
    }

    // ADDITION / SUBTRACTION
    if (a.is_nan || b.is_nan) {
        out.result = FP16_QNAN;
        out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return out;
    }
    if (a.is_inf && b.is_inf) {
        if (a.sign != b.sign) {
            out.result = FP16_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        } else {
            out.result = (a.sign ? FP16_NEG_INF : FP16_POS_INF);
            out.status = FpuStatus::IS_INF;
        }
        return out;
    }
    if (a.is_inf) {
        out.result = (a.sign ? FP16_NEG_INF : FP16_POS_INF);
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (b.is_inf) {
        out.result = (b.sign ? FP16_NEG_INF : FP16_POS_INF);
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (a.is_zero && b.is_zero) {
        uint8_t res_sign = (a.sign && b.sign) ? 1 : 0;
        out.result = (res_sign ? FP16_NEG_ZERO : FP16_POS_ZERO);
        out.status = FpuStatus::IS_ZERO;
        return out;
    }
    if (a.is_zero) {
        out.result = fp16_pack(b.sign, b.exp, b.mant & 0x3FF);
        return out;
    }
    if (b.is_zero) {
        out.result = fp16_pack(a.sign, a.exp, a.mant & 0x3FF);
        return out;
    }

    bool a_larger = false;
    if (a.exp > b.exp) {
        a_larger = true;
    } else if (a.exp == b.exp) {
        a_larger = (a.mant >= b.mant);
    }

    Fp16Unpacked large = a_larger ? a : b;
    Fp16Unpacked small = a_larger ? b : a;
    bool eff_sub = (large.sign != small.sign);

    int16_t exp_diff = large.exp - small.exp;

    // Extend 11-bit significand by 3 bits [Guard, Round, Sticky] -> 14 bits
    uint32_t mant_large_ext = static_cast<uint32_t>(large.mant) << 3;
    uint32_t mant_small_ext = static_cast<uint32_t>(small.mant) << 3;

    uint8_t sticky = 0;
    if (exp_diff > 0) {
        if (exp_diff >= 14) {
            sticky = (small.mant != 0) ? 1 : 0;
            mant_small_ext = 0;
        } else {
            uint32_t mask = (1U << exp_diff) - 1;
            sticky = ((small.mant & mask) != 0) ? 1 : 0;
            mant_small_ext >>= exp_diff;
        }
    }
    mant_small_ext |= sticky;

    uint32_t mant_res;
    if (eff_sub) {
        mant_res = mant_large_ext - mant_small_ext;
    } else {
        mant_res = mant_large_ext + mant_small_ext;
    }

    if (mant_res == 0) {
        out.result = FP16_POS_ZERO;
        out.status = FpuStatus::IS_ZERO;
        return out;
    }

    int16_t exp_res = large.exp;

    if (!eff_sub) {
        if (mant_res & (1U << 14)) {
            sticky |= static_cast<uint8_t>(mant_res & 1);
            mant_res >>= 1;
            exp_res++;
        }
    } else {
        uint32_t upper = mant_res >> 3;
        if (upper != 0) {
            uint8_t lz = clz32(upper) - 21; // Bit 10 is target leading 1
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

    uint16_t final_mant = static_cast<uint16_t>(mant_res >> 3); // 11-bit significand
    uint8_t g = static_cast<uint8_t>((mant_res >> 2) & 1);
    uint8_t r = static_cast<uint8_t>((mant_res >> 1) & 1);
    uint8_t s = static_cast<uint8_t>(mant_res & 1) | sticky;

    bool round_up = g && (r || s || (final_mant & 1));
    if (round_up) {
        final_mant++;
        if (final_mant & (1U << 11)) {
            final_mant >>= 1;
            exp_res++;
        }
    }

    if (g || r || s) {
        out.status |= FpuStatus::INEXACT;
    }

    if (exp_res >= 31) {
        out.result = (large.sign ? FP16_NEG_INF : FP16_POS_INF);
        out.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
    } else if (exp_res <= 0) {
        out.result = (large.sign ? FP16_NEG_ZERO : FP16_POS_ZERO);
        out.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
    } else {
        out.result = fp16_pack(large.sign, exp_res, final_mant & 0x3FF);
    }

    return out;
}

// Convert FP16 bit-pattern to float for host tests
inline float fp16_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t f = h & 0x3FF;
    uint32_t out;
    if (e == 0x1F) {
        out = (s << 31) | 0x7F800000 | (f << 13);
    } else if (e == 0) {
        if (f == 0) {
            out = (s << 31);
        } else {
            // subnormal
            while (!(f & 0x400)) { f <<= 1; e--; }
            e++; f &= ~0x400;
            out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
        }
    } else {
        out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
    }
    return u32_to_float(out);
}

// Convert float to FP16 bit-pattern for host tests
inline uint16_t float_to_fp16(float val) {
    uint32_t x = float_to_u32(val);
    uint32_t s = (x >> 31) & 1;
    int32_t  e = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t f = x & 0x7FFFFF;

    if (((x >> 23) & 0xFF) == 0xFF) {
        return (s << 15) | 0x7C00 | (f ? 0x200 : 0);
    }
    if (e >= 31) {
        return (s << 15) | 0x7C00;
    }
    if (e <= 0) {
        return (s << 15);
    }
    return (s << 15) | (e << 10) | (f >> 13);
}

} // namespace vph
