#pragma once

#include <cstdint>
#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp16_ieee_core.hpp"

namespace vph {

/**
 * @brief Unpacked BF16 Representation for Bit-Level ALU
 */
struct Bf16Unpacked {
    uint8_t  sign;          ///< 1-bit sign: 0 = positive, 1 = negative
    int16_t  exp;           ///< 8-bit biased exponent (0..255)
    uint16_t mant;          ///< 8-bit significand with implicit hidden bit at bit 7
    bool     is_zero;
    bool     is_subnormal;
    bool     is_inf;
    bool     is_nan;
};

inline Bf16Unpacked bf16_unpack(uint16_t raw) {
    Bf16Unpacked u;
    u.sign = static_cast<uint8_t>((raw >> 15) & 0x1);
    u.exp  = static_cast<int16_t>((raw >> 7) & 0xFF);
    uint16_t frac = raw & 0x7F;

    u.is_nan       = (u.exp == 0xFF) && (frac != 0);
    u.is_inf       = (u.exp == 0xFF) && (frac == 0);
    u.is_zero      = (u.exp == 0) && (frac == 0);
    u.is_subnormal = (u.exp == 0) && (frac != 0);

    if (u.exp == 0) {
        u.mant = frac;
        u.exp  = 1;
    } else {
        u.mant = (1U << 7) | frac;
    }

    return u;
}

inline uint16_t bf16_pack(uint8_t sign, int16_t exp, uint16_t frac) {
    uint16_t raw = (static_cast<uint16_t>(sign & 0x1) << 15);
    raw |= (static_cast<uint16_t>(exp & 0xFF) << 7);
    raw |= (frac & 0x7F);
    return raw;
}

constexpr uint16_t BF16_POS_ZERO = 0x0000;
constexpr uint16_t BF16_NEG_ZERO = 0x8000;
constexpr uint16_t BF16_POS_INF  = 0x7F80;
constexpr uint16_t BF16_NEG_INF  = 0xFF80;
constexpr uint16_t BF16_QNAN     = 0x7FC0;

struct Bf16AluResult {
    uint16_t result;
    uint16_t residual;
    uint8_t  status;
};

inline Bf16AluResult bf16_eval_alu(uint16_t raw_a, uint16_t raw_b, uint8_t opcode) {
    Bf16AluResult out{0, 0, FpuStatus::OK};

    if (opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        out.result = raw_a & 0x7FFF;
        return out;
    }
    if (opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        out.result = raw_a ^ 0x8000;
        return out;
    }

    Bf16Unpacked a = bf16_unpack(raw_a);
    Bf16Unpacked b = bf16_unpack(raw_b);

    if (opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    // MULTIPLICATION
    if (opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;

        if (a.is_nan || b.is_nan) {
            out.result = BF16_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
            out.result = BF16_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if (a.is_inf || b.is_inf) {
            out.result = (res_sign ? BF16_NEG_INF : BF16_POS_INF);
            out.status = FpuStatus::IS_INF;
            return out;
        }
        if (a.is_zero || b.is_zero) {
            out.result = (res_sign ? BF16_NEG_ZERO : BF16_POS_ZERO);
            out.status = FpuStatus::IS_ZERO;
            return out;
        }

        // 8x8-bit mantissa multiplication -> 16 bits
        uint32_t prod = static_cast<uint32_t>(a.mant) * static_cast<uint32_t>(b.mant);
        int16_t exp_res = a.exp + b.exp - 127;

        uint16_t mant_res;
        uint8_t guard, round, sticky;

        if (prod & (1U << 15)) {
            exp_res += 1;
            mant_res = static_cast<uint16_t>(prod >> 8); // 8 bits
            guard    = static_cast<uint8_t>((prod >> 7) & 1);
            round    = static_cast<uint8_t>((prod >> 6) & 1);
            sticky   = (prod & 0x3F) ? 1 : 0;
        } else {
            mant_res = static_cast<uint16_t>(prod >> 7); // 8 bits
            guard    = static_cast<uint8_t>((prod >> 6) & 1);
            round    = static_cast<uint8_t>((prod >> 5) & 1);
            sticky   = (prod & 0x1F) ? 1 : 0;
        }

        bool lsb = (mant_res & 1);
        bool round_up = guard && (round || sticky || lsb);
        if (round_up) {
            mant_res++;
            if (mant_res & (1U << 8)) {
                mant_res >>= 1;
                exp_res++;
            }
        }

        if (guard || round || sticky) {
            out.status |= FpuStatus::INEXACT;
        }

        if (exp_res >= 255) {
            out.result = (res_sign ? BF16_NEG_INF : BF16_POS_INF);
            out.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
        } else if (exp_res <= 0) {
            out.result = (res_sign ? BF16_NEG_ZERO : BF16_POS_ZERO);
            out.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
        } else {
            out.result = bf16_pack(res_sign, exp_res, mant_res & 0x7F);
        }

        return out;
    }

    // ADDITION / SUBTRACTION
    if (a.is_nan || b.is_nan) {
        out.result = BF16_QNAN;
        out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return out;
    }
    if (a.is_inf && b.is_inf) {
        if (a.sign != b.sign) {
            out.result = BF16_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        } else {
            out.result = (a.sign ? BF16_NEG_INF : BF16_POS_INF);
            out.status = FpuStatus::IS_INF;
        }
        return out;
    }
    if (a.is_inf) {
        out.result = (a.sign ? BF16_NEG_INF : BF16_POS_INF);
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (b.is_inf) {
        out.result = (b.sign ? BF16_NEG_INF : BF16_POS_INF);
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (a.is_zero && b.is_zero) {
        uint8_t res_sign = (a.sign && b.sign) ? 1 : 0;
        out.result = (res_sign ? BF16_NEG_ZERO : BF16_POS_ZERO);
        out.status = FpuStatus::IS_ZERO;
        return out;
    }
    if (a.is_zero) {
        out.result = bf16_pack(b.sign, b.exp, b.mant & 0x7F);
        return out;
    }
    if (b.is_zero) {
        out.result = bf16_pack(a.sign, a.exp, a.mant & 0x7F);
        return out;
    }

    bool a_larger = false;
    if (a.exp > b.exp) {
        a_larger = true;
    } else if (a.exp == b.exp) {
        a_larger = (a.mant >= b.mant);
    }

    Bf16Unpacked large = a_larger ? a : b;
    Bf16Unpacked small = a_larger ? b : a;
    bool eff_sub = (large.sign != small.sign);

    int16_t exp_diff = large.exp - small.exp;

    // Extend 8-bit significand by 3 bits [Guard, Round, Sticky] -> 11 bits
    uint32_t mant_large_ext = static_cast<uint32_t>(large.mant) << 3;
    uint32_t mant_small_ext = static_cast<uint32_t>(small.mant) << 3;

    uint8_t sticky = 0;
    if (exp_diff > 0) {
        if (exp_diff >= 11) {
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
        out.result = BF16_POS_ZERO;
        out.status = FpuStatus::IS_ZERO;
        return out;
    }

    int16_t exp_res = large.exp;

    if (!eff_sub) {
        if (mant_res & (1U << 11)) {
            sticky |= static_cast<uint8_t>(mant_res & 1);
            mant_res >>= 1;
            exp_res++;
        }
    } else {
        uint32_t upper = mant_res >> 3;
        if (upper != 0) {
            uint8_t lz = clz32(upper) - 24; // Bit 7 is target leading 1
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

    uint16_t final_mant = static_cast<uint16_t>(mant_res >> 3); // 8-bit significand
    uint8_t g = static_cast<uint8_t>((mant_res >> 2) & 1);
    uint8_t r = static_cast<uint8_t>((mant_res >> 1) & 1);
    uint8_t s = static_cast<uint8_t>(mant_res & 1) | sticky;

    bool round_up = g && (r || s || (final_mant & 1));
    if (round_up) {
        final_mant++;
        if (final_mant & (1U << 8)) {
            final_mant >>= 1;
            exp_res++;
        }
    }

    if (g || r || s) {
        out.status |= FpuStatus::INEXACT;
    }

    if (exp_res >= 255) {
        out.result = (large.sign ? BF16_NEG_INF : BF16_POS_INF);
        out.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
    } else if (exp_res <= 0) {
        out.result = (large.sign ? BF16_NEG_ZERO : BF16_POS_ZERO);
        out.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
    } else {
        out.result = bf16_pack(large.sign, exp_res, final_mant & 0x7F);
    }

    return out;
}

inline float bf16_to_float(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    return u32_to_float(u);
}

inline uint16_t float_to_bf16(float f) {
    uint32_t u = float_to_u32(f);
    // Round to nearest even on the 16-bit truncation
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

} // namespace vph
