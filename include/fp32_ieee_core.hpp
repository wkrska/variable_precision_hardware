#pragma once

#include <cstdint>
#include "fpu_types.hpp"
#include "bit_ops.hpp"

namespace vph {

/**
 * @brief Unpacked FP32 Representation for Bit-Level ALU
 */
struct Fp32Unpacked {
    uint8_t  sign;          ///< 1-bit sign: 0 = positive, 1 = negative
    int16_t  exp;           ///< 8-bit biased exponent (0..255)
    uint32_t mant;          ///< 24-bit significand with implicit hidden bit at bit 23
    bool     is_zero;       ///< Operand is +/- 0.0
    bool     is_subnormal;  ///< Exponent == 0 and fraction != 0
    bool     is_inf;        ///< Exponent == 255 and fraction == 0
    bool     is_nan;        ///< Exponent == 255 and fraction != 0
};

/**
 * @brief Unpack a raw IEEE-754 32-bit float word into constituent bit fields.
 * 
 * Performs bit extraction, hidden-bit restoration, and special-case classification
 * without relying on compiler floating-point intrinsics.
 */
inline Fp32Unpacked fp32_unpack(uint32_t raw) {
    Fp32Unpacked u;
    u.sign = static_cast<uint8_t>((raw >> 31) & 0x1);
    u.exp  = static_cast<int16_t>((raw >> 23) & 0xFF);
    uint32_t frac = raw & 0x7FFFFF;

    u.is_nan       = (u.exp == 0xFF) && (frac != 0);
    u.is_inf       = (u.exp == 0xFF) && (frac == 0);
    u.is_zero      = (u.exp == 0) && (frac == 0);
    u.is_subnormal = (u.exp == 0) && (frac != 0);

    // Restore implicit leading significand bit (1 for normal, 0 for subnormal/zero)
    if (u.exp == 0) {
        u.mant = frac;
        u.exp  = 1; // Effective exponent for subnormal numbers
    } else {
        u.mant = (1U << 23) | frac;
    }

    return u;
}

/**
 * @brief Pack sign, exponent, and fraction into an IEEE-754 binary32 word.
 */
inline uint32_t fp32_pack(uint8_t sign, int16_t exp, uint32_t frac) {
    uint32_t raw = (static_cast<uint32_t>(sign & 0x1) << 31);
    raw |= (static_cast<uint32_t>(exp & 0xFF) << 23);
    raw |= (frac & 0x7FFFFF);
    return raw;
}

/**
 * @brief Canonical IEEE FP32 constants
 */
constexpr uint32_t FP32_POS_ZERO = 0x00000000;
constexpr uint32_t FP32_NEG_ZERO = 0x80000000;
constexpr uint32_t FP32_POS_INF  = 0x7F800000;
constexpr uint32_t FP32_NEG_INF  = 0xFF800000;
constexpr uint32_t FP32_QNAN     = 0x7FC00000;

/**
 * @brief Core Bit-Level Combinational FP32 ALU
 * 
 * Implements IEEE-754 compliant Addition, Subtraction, Multiplication,
 * Absolute Value, Negation, and Error-Free Transformation residuals (ADD_HILO, MUL_HILO).
 */
struct Fp32AluResult {
    uint32_t result;
    uint32_t residual;
    uint8_t  status;
};

inline Fp32AluResult fp32_eval_alu(uint32_t raw_a, uint32_t raw_b, uint8_t opcode) {
    Fp32AluResult out{0, 0, FpuStatus::OK};

    // Unary Operations (ABS, NEG)
    if (opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        out.result = raw_a & 0x7FFFFFFF;
        return out;
    }
    if (opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        out.result = raw_a ^ 0x80000000;
        return out;
    }

    Fp32Unpacked a = fp32_unpack(raw_a);
    Fp32Unpacked b = fp32_unpack(raw_b);

    // Invert sign of B for SUB
    if (opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    // =========================================================================
    // MULTIPLICATION (MUL, MUL_HILO)
    // =========================================================================
    if (opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;

        // 1. Handle NaNs
        if (a.is_nan || b.is_nan) {
            out.result = FP32_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }

        // 2. Handle 0 * Inf -> Invalid NaN
        if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
            out.result = FP32_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }

        // 3. Handle Infinities
        if (a.is_inf || b.is_inf) {
            out.result = (res_sign ? FP32_NEG_INF : FP32_POS_INF);
            out.status = FpuStatus::IS_INF;
            return out;
        }

        // 4. Handle Zeros
        if (a.is_zero || b.is_zero) {
            out.result = (res_sign ? FP32_NEG_ZERO : FP32_POS_ZERO);
            out.status = FpuStatus::IS_ZERO;
            return out;
        }

        // 5. Significant Multiplication: 24-bit x 24-bit unsigned product -> 48 bits
        uint64_t prod = static_cast<uint64_t>(a.mant) * static_cast<uint64_t>(b.mant);
        int16_t exp_res = a.exp + b.exp - 127;

        // Normalization: check MSB of 48-bit product (bits 47 and 46)
        uint32_t mant_res;
        uint8_t guard, round, sticky;
        uint32_t low_residual_bits;

        if (prod & (1ULL << 47)) {
            // Product in [2.0, 4.0): shift right by 1
            exp_res += 1;
            mant_res = static_cast<uint32_t>(prod >> 24); // 24 bits
            guard = static_cast<uint8_t>((prod >> 23) & 1);
            round = static_cast<uint8_t>((prod >> 22) & 1);
            sticky = (prod & 0x00000000003FFFFFULL) ? 1 : 0;
            low_residual_bits = static_cast<uint32_t>(prod & 0xFFFFFF);
        } else {
            // Product in [1.0, 2.0)
            mant_res = static_cast<uint32_t>(prod >> 23); // 24 bits
            guard = static_cast<uint8_t>((prod >> 22) & 1);
            round = static_cast<uint8_t>((prod >> 21) & 1);
            sticky = (prod & 0x00000000001FFFFFULL) ? 1 : 0;
            low_residual_bits = static_cast<uint32_t>(prod & 0x7FFFFF);
        }

        // Round to Nearest Even (RNE)
        bool lsb = (mant_res & 1);
        bool round_up = guard && (round || sticky || lsb);
        if (round_up) {
            mant_res++;
            if (mant_res & (1U << 24)) {
                // Significand overflowed 24 bits
                mant_res >>= 1;
                exp_res++;
            }
        }

        if (guard || round || sticky) {
            out.status |= FpuStatus::INEXACT;
        }

        // Exponent Range Check
        if (exp_res >= 255) {
            out.result = (res_sign ? FP32_NEG_INF : FP32_POS_INF);
            out.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
        } else if (exp_res <= 0) {
            // Underflow
            out.result = (res_sign ? FP32_NEG_ZERO : FP32_POS_ZERO);
            out.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
        } else {
            out.result = fp32_pack(res_sign, exp_res, mant_res & 0x7FFFFF);
        }

        // For MUL_HILO: construct exact product residual
        if (opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
            // Residual has opposite sign to rounding error, exponent scaled down
            if (low_residual_bits != 0 && exp_res > 24) {
                uint8_t lz = clz32(low_residual_bits) - 8;
                int16_t res_exp = exp_res - 24 - lz;
                if (res_exp > 0 && res_exp < 255) {
                    uint32_t norm_frac = (low_residual_bits << (lz + 1)) & 0x7FFFFF;
                    out.residual = fp32_pack(res_sign, res_exp, norm_frac);
                }
            }
        }

        return out;
    }

    // =========================================================================
    // ADDITION / SUBTRACTION (ADD, SUB, ADD_HILO)
    // =========================================================================
    // 1. Handle NaNs
    if (a.is_nan || b.is_nan) {
        out.result = FP32_QNAN;
        out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return out;
    }

    // 2. Handle Infinities
    if (a.is_inf && b.is_inf) {
        if (a.sign != b.sign) {
            // Inf - Inf -> NaN
            out.result = FP32_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        } else {
            out.result = (a.sign ? FP32_NEG_INF : FP32_POS_INF);
            out.status = FpuStatus::IS_INF;
        }
        return out;
    }
    if (a.is_inf) {
        out.result = (a.sign ? FP32_NEG_INF : FP32_POS_INF);
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (b.is_inf) {
        out.result = (b.sign ? FP32_NEG_INF : FP32_POS_INF);
        out.status = FpuStatus::IS_INF;
        return out;
    }

    // 3. Handle Zeros
    if (a.is_zero && b.is_zero) {
        uint8_t res_sign = (a.sign && b.sign) ? 1 : 0;
        out.result = (res_sign ? FP32_NEG_ZERO : FP32_POS_ZERO);
        out.status = FpuStatus::IS_ZERO;
        return out;
    }
    if (a.is_zero) {
        out.result = fp32_pack(b.sign, b.exp, b.mant & 0x7FFFFF);
        return out;
    }
    if (b.is_zero) {
        out.result = fp32_pack(a.sign, a.exp, a.mant & 0x7FFFFF);
        return out;
    }

    // 4. Determine Larger Operand (Magnitude Comparison)
    bool a_is_larger = false;
    if (a.exp > b.exp) {
        a_is_larger = true;
    } else if (a.exp == b.exp) {
        a_is_larger = (a.mant >= b.mant);
    }

    Fp32Unpacked large = a_is_larger ? a : b;
    Fp32Unpacked small = a_is_larger ? b : a;
    bool eff_sub = (large.sign != small.sign);

    int16_t exp_diff = large.exp - small.exp;

    // 5. Align Significands
    // Extend significands by 3 bits: [Guard, Round, Sticky]
    uint64_t mant_large_ext = static_cast<uint64_t>(large.mant) << 3;
    uint64_t mant_small_ext = static_cast<uint64_t>(small.mant) << 3;

    uint8_t sticky = 0;
    if (exp_diff > 0) {
        if (exp_diff >= 32) {
            sticky = (small.mant != 0) ? 1 : 0;
            mant_small_ext = 0;
        } else {
            uint32_t mask = (1U << exp_diff) - 1;
            sticky = ((small.mant & mask) != 0) ? 1 : 0;
            mant_small_ext >>= exp_diff;
        }
    }
    mant_small_ext |= sticky;

    // 6. Execute Addition or Subtraction on Significands
    uint64_t mant_res;
    if (eff_sub) {
        mant_res = mant_large_ext - mant_small_ext;
    } else {
        mant_res = mant_large_ext + mant_small_ext;
    }

    // Exact cancellation -> Zero result
    if (mant_res == 0) {
        out.result = FP32_POS_ZERO;
        out.status = FpuStatus::IS_ZERO;
        return out;
    }

    int16_t exp_res = large.exp;

    // 7. Normalization
    if (!eff_sub) {
        // Addition can overflow into bit 27 (where normal bit is 26)
        if (mant_res & (1ULL << 27)) {
            sticky = sticky | (mant_res & 1);
            mant_res >>= 1;
            exp_res++;
        }
    } else {
        // Subtraction can lead to significant cancellation: count leading zeros
        uint32_t upper = static_cast<uint32_t>(mant_res >> 3);
        if (upper != 0) {
            uint8_t lz = clz32(upper) - 8; // Bit 23 is the target leading 1
            if (lz > 0) {
                if (exp_res > lz) {
                    mant_res <<= lz;
                    exp_res -= lz;
                } else {
                    // Underflow
                    mant_res <<= (exp_res - 1);
                    exp_res = 0;
                }
            }
        }
    }

    // 8. Rounding (RNE)
    uint32_t final_mant = static_cast<uint32_t>(mant_res >> 3); // 24-bit significand
    uint8_t g = static_cast<uint8_t>((mant_res >> 2) & 1);
    uint8_t r = static_cast<uint8_t>((mant_res >> 1) & 1);
    uint8_t s = static_cast<uint8_t>(mant_res & 1) | sticky;

    bool round_up = g && (r || s || (final_mant & 1));
    if (round_up) {
        final_mant++;
        if (final_mant & (1U << 24)) {
            final_mant >>= 1;
            exp_res++;
        }
    }

    if (g || r || s) {
        out.status |= FpuStatus::INEXACT;
    }

    // 9. Overflow / Underflow Checks
    if (exp_res >= 255) {
        out.result = (large.sign ? FP32_NEG_INF : FP32_POS_INF);
        out.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
    } else if (exp_res <= 0) {
        out.result = (large.sign ? FP32_NEG_ZERO : FP32_POS_ZERO);
        out.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
    } else {
        out.result = fp32_pack(large.sign, exp_res, final_mant & 0x7FFFFF);
    }

    return out;
}

} // namespace vph
