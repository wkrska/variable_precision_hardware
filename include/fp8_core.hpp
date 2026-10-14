#pragma once

#include <cstdint>
#include "fpu_types.hpp"
#include "bit_ops.hpp"

namespace vph {

// ============================================================================
// FP8 Formats:
// 1. E4M3 (OCP / NVIDIA / Intel standard): 1 sign bit, 4 exp bits (bias 7), 3 mant bits
// 2. E5M2 (IEEE 754 style): 1 sign bit, 5 exp bits (bias 15), 2 mant bits
// ============================================================================

enum class Fp8Type : uint8_t {
    E4M3 = 0,
    E5M2 = 1
};

// ----------------------------------------------------------------------------
// FP8 E4M3 (1, 4, 3, bias=7)
// ----------------------------------------------------------------------------
struct Fp8E4M3Unpacked {
    uint8_t sign;
    int16_t exp;
    uint8_t mant; // 4 bits: hidden bit at bit 3 + 3 frac bits
    bool    is_zero;
    bool    is_subnormal;
    bool    is_nan;
};

inline Fp8E4M3Unpacked fp8_e4m3_unpack(uint8_t raw) {
    Fp8E4M3Unpacked u;
    u.sign = (raw >> 7) & 0x1;
    uint8_t exp_bits = (raw >> 3) & 0x0F;
    uint8_t frac = raw & 0x07;

    // In OCP E4M3: 0x7F (0 1111 111) and 0xFF (1 1111 111) are NaNs; no infinities.
    u.is_nan = (exp_bits == 0x0F) && (frac == 0x07);
    u.is_zero = (exp_bits == 0) && (frac == 0);
    u.is_subnormal = (exp_bits == 0) && (frac != 0);

    if (exp_bits == 0) {
        u.mant = frac;
        u.exp  = 1 - 7; // unbiased exponent = -6
    } else {
        u.mant = (1U << 3) | frac;
        u.exp  = static_cast<int16_t>(exp_bits) - 7;
    }
    return u;
}

inline uint8_t fp8_e4m3_pack(uint8_t sign, int16_t exp, uint8_t frac) {
    uint8_t raw = (sign & 0x1) << 7;
    raw |= ((exp + 7) & 0x0F) << 3;
    raw |= (frac & 0x07);
    return raw;
}

constexpr uint8_t FP8_E4M3_POS_ZERO = 0x00;
constexpr uint8_t FP8_E4M3_NEG_ZERO = 0x80;
constexpr uint8_t FP8_E4M3_QNAN     = 0x7F;

struct Fp8AluResult {
    uint8_t result;
    uint8_t residual;
    uint8_t status;
};

inline Fp8AluResult fp8_e4m3_eval_alu(uint8_t raw_a, uint8_t raw_b, uint8_t opcode) {
    Fp8AluResult out{0, 0, FpuStatus::OK};

    if (opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        out.result = raw_a & 0x7F;
        return out;
    }
    if (opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        out.result = raw_a ^ 0x80;
        return out;
    }

    Fp8E4M3Unpacked a = fp8_e4m3_unpack(raw_a);
    Fp8E4M3Unpacked b = fp8_e4m3_unpack(raw_b);

    if (opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    // MULTIPLY
    if (opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;
        if (a.is_nan || b.is_nan) {
            out.result = FP8_E4M3_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if (a.is_zero || b.is_zero) {
            out.result = (res_sign ? FP8_E4M3_NEG_ZERO : FP8_E4M3_POS_ZERO);
            out.status = FpuStatus::IS_ZERO;
            return out;
        }

        int16_t exp_sum = a.exp + b.exp;
        // 4 bits * 4 bits = 8 bits product
        uint16_t prod = static_cast<uint16_t>(a.mant) * static_cast<uint16_t>(b.mant);
        // prod has implicit bits at bit 6 or 7
        uint8_t norm_shift = 0;
        if (prod & (1U << 7)) {
            norm_shift = 1;
        }

        int16_t final_exp = exp_sum + norm_shift;
        int16_t biased_exp = final_exp + 7;

        uint8_t frac = 0;
        if (norm_shift == 1) {
            frac = (prod >> 4) & 0x07;
            // rounding bit at bit 3
            if ((prod >> 3) & 1) {
                frac++;
            }
        } else {
            frac = (prod >> 3) & 0x07;
            if ((prod >> 2) & 1) {
                frac++;
            }
        }

        if (biased_exp >= 15) {
            // Saturate to max representable (448 = 0x7E) or NaN
            out.result = (res_sign << 7) | 0x7E;
            out.status = FpuStatus::OVERFLOW | FpuStatus::INEXACT;
            return out;
        }
        if (biased_exp <= 0) {
            out.result = (res_sign ? FP8_E4M3_NEG_ZERO : FP8_E4M3_POS_ZERO);
            out.status = FpuStatus::UNDERFLOW | FpuStatus::INEXACT;
            return out;
        }

        out.result = (res_sign << 7) | (static_cast<uint8_t>(biased_exp & 0x0F) << 3) | (frac & 0x07);
        return out;
    }

    // ADD / SUB
    if (a.is_nan || b.is_nan) {
        out.result = FP8_E4M3_QNAN;
        out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return out;
    }
    if (a.is_zero) {
        out.result = raw_b ^ ((opcode == static_cast<uint8_t>(FpuOpcode::SUB)) ? 0x80 : 0x00);
        return out;
    }
    if (b.is_zero) {
        out.result = raw_a;
        return out;
    }

    // Alignment
    bool a_ge_b = (a.exp > b.exp) || ((a.exp == b.exp) && (a.mant >= b.mant));
    Fp8E4M3Unpacked big = a_ge_b ? a : b;
    Fp8E4M3Unpacked sml = a_ge_b ? b : a;

    int16_t exp_diff = big.exp - sml.exp;
    if (exp_diff > 8) exp_diff = 8;

    uint16_t big_mant = static_cast<uint16_t>(big.mant) << 6;
    uint16_t sml_mant = (static_cast<uint16_t>(sml.mant) << 6) >> exp_diff;

    uint16_t res_mant = 0;
    uint8_t  res_sign = big.sign;

    if (big.sign == sml.sign) {
        res_mant = big_mant + sml_mant;
    } else {
        res_mant = big_mant - sml_mant;
    }

    if (res_mant == 0) {
        out.result = FP8_E4M3_POS_ZERO;
        out.status = FpuStatus::IS_ZERO;
        return out;
    }

    // Normalization
    int16_t res_exp = big.exp;
    if (res_mant & (1U << 10)) { // overflowed bit 9
        res_mant >>= 1;
        res_exp++;
    } else {
        // Find leading one (clz on 16-bit)
        uint8_t lz = clz32(static_cast<uint32_t>(res_mant) << 16);
        // bit 9 corresponds to lz = 6
        if (lz > 6) {
            uint8_t shift = lz - 6;
            res_mant <<= shift;
            res_exp -= shift;
        }
    }

    // Rounding (RNE)
    uint8_t frac = (res_mant >> 6) & 0x07;
    bool guard = (res_mant >> 5) & 1;
    bool round_bit = (res_mant >> 4) & 1;
    bool sticky = (res_mant & 0x0F) != 0;
    if (guard && (round_bit || sticky || (frac & 1))) {
        frac++;
        if (frac == 8) {
            frac = 0;
            res_exp++;
        }
    }

    int16_t biased_exp = res_exp + 7;
    if (biased_exp >= 15) {
        out.result = (res_sign << 7) | 0x7E;
        out.status = FpuStatus::OVERFLOW | FpuStatus::INEXACT;
        return out;
    }
    if (biased_exp <= 0) {
        out.result = (res_sign ? FP8_E4M3_NEG_ZERO : FP8_E4M3_POS_ZERO);
        out.status = FpuStatus::UNDERFLOW | FpuStatus::INEXACT;
        return out;
    }

    out.result = (res_sign << 7) | (static_cast<uint8_t>(biased_exp & 0x0F) << 3) | (frac & 0x07);
    return out;
}

// ----------------------------------------------------------------------------
// FP8 E5M2 (1, 5, 2, bias=15) - Standard IEEE 754-like
// ----------------------------------------------------------------------------
struct Fp8E5M2Unpacked {
    uint8_t sign;
    int16_t exp;
    uint8_t mant; // 3 bits: hidden bit at bit 2 + 2 frac bits
    bool    is_zero;
    bool    is_subnormal;
    bool    is_inf;
    bool    is_nan;
};

inline Fp8E5M2Unpacked fp8_e5m2_unpack(uint8_t raw) {
    Fp8E5M2Unpacked u;
    u.sign = (raw >> 7) & 0x1;
    uint8_t exp_bits = (raw >> 2) & 0x1F;
    uint8_t frac = raw & 0x03;

    u.is_nan = (exp_bits == 0x1F) && (frac != 0);
    u.is_inf = (exp_bits == 0x1F) && (frac == 0);
    u.is_zero = (exp_bits == 0) && (frac == 0);
    u.is_subnormal = (exp_bits == 0) && (frac != 0);

    if (exp_bits == 0) {
        u.mant = frac;
        u.exp  = 1 - 15;
    } else {
        u.mant = (1U << 2) | frac;
        u.exp  = static_cast<int16_t>(exp_bits) - 15;
    }
    return u;
}

constexpr uint8_t FP8_E5M2_POS_ZERO = 0x00;
constexpr uint8_t FP8_E5M2_NEG_ZERO = 0x80;
constexpr uint8_t FP8_E5M2_POS_INF  = 0x7C;
constexpr uint8_t FP8_E5M2_NEG_INF  = 0xFC;
constexpr uint8_t FP8_E5M2_QNAN     = 0x7E;

inline Fp8AluResult fp8_e5m2_eval_alu(uint8_t raw_a, uint8_t raw_b, uint8_t opcode) {
    Fp8AluResult out{0, 0, FpuStatus::OK};

    if (opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        out.result = raw_a & 0x7F;
        return out;
    }
    if (opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        out.result = raw_a ^ 0x80;
        return out;
    }

    Fp8E5M2Unpacked a = fp8_e5m2_unpack(raw_a);
    Fp8E5M2Unpacked b = fp8_e5m2_unpack(raw_b);

    if (opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    // MULTIPLY
    if (opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;
        if (a.is_nan || b.is_nan) {
            out.result = FP8_E5M2_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
            out.result = FP8_E5M2_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if (a.is_inf || b.is_inf) {
            out.result = res_sign ? FP8_E5M2_NEG_INF : FP8_E5M2_POS_INF;
            out.status = FpuStatus::IS_INF;
            return out;
        }
        if (a.is_zero || b.is_zero) {
            out.result = res_sign ? FP8_E5M2_NEG_ZERO : FP8_E5M2_POS_ZERO;
            out.status = FpuStatus::IS_ZERO;
            return out;
        }

        int16_t exp_sum = a.exp + b.exp;
        uint16_t prod = static_cast<uint16_t>(a.mant) * static_cast<uint16_t>(b.mant);
        uint8_t norm_shift = 0;
        if (prod & (1U << 5)) {
            norm_shift = 1;
        }

        int16_t final_exp = exp_sum + norm_shift;
        int16_t biased_exp = final_exp + 15;

        uint8_t frac = 0;
        if (norm_shift == 1) {
            frac = (prod >> 3) & 0x03;
            if ((prod >> 2) & 1) frac++;
        } else {
            frac = (prod >> 2) & 0x03;
            if ((prod >> 1) & 1) frac++;
        }

        if (biased_exp >= 31) {
            out.result = res_sign ? FP8_E5M2_NEG_INF : FP8_E5M2_POS_INF;
            out.status = FpuStatus::OVERFLOW | FpuStatus::INEXACT;
            return out;
        }
        if (biased_exp <= 0) {
            out.result = res_sign ? FP8_E5M2_NEG_ZERO : FP8_E5M2_POS_ZERO;
            out.status = FpuStatus::UNDERFLOW | FpuStatus::INEXACT;
            return out;
        }

        out.result = (res_sign << 7) | (static_cast<uint8_t>(biased_exp & 0x1F) << 2) | (frac & 0x03);
        return out;
    }

    // ADD / SUB
    if (a.is_nan || b.is_nan) {
        out.result = FP8_E5M2_QNAN;
        out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return out;
    }
    if (a.is_inf && b.is_inf) {
        if (a.sign != b.sign) {
            out.result = FP8_E5M2_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        out.result = a.sign ? FP8_E5M2_NEG_INF : FP8_E5M2_POS_INF;
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (a.is_inf) {
        out.result = a.sign ? FP8_E5M2_NEG_INF : FP8_E5M2_POS_INF;
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (b.is_inf) {
        out.result = b.sign ? FP8_E5M2_NEG_INF : FP8_E5M2_POS_INF;
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (a.is_zero) {
        out.result = raw_b ^ ((opcode == static_cast<uint8_t>(FpuOpcode::SUB)) ? 0x80 : 0x00);
        return out;
    }
    if (b.is_zero) {
        out.result = raw_a;
        return out;
    }

    bool a_ge_b = (a.exp > b.exp) || ((a.exp == b.exp) && (a.mant >= b.mant));
    Fp8E5M2Unpacked big = a_ge_b ? a : b;
    Fp8E5M2Unpacked sml = a_ge_b ? b : a;

    int16_t exp_diff = big.exp - sml.exp;
    if (exp_diff > 6) exp_diff = 6;

    uint16_t big_mant = static_cast<uint16_t>(big.mant) << 4;
    uint16_t sml_mant = (static_cast<uint16_t>(sml.mant) << 4) >> exp_diff;

    uint16_t res_mant = 0;
    uint8_t  res_sign = big.sign;

    if (big.sign == sml.sign) {
        res_mant = big_mant + sml_mant;
    } else {
        res_mant = big_mant - sml_mant;
    }

    if (res_mant == 0) {
        out.result = FP8_E5M2_POS_ZERO;
        out.status = FpuStatus::IS_ZERO;
        return out;
    }

    int16_t res_exp = big.exp;
    if (res_mant & (1U << 7)) {
        res_mant >>= 1;
        res_exp++;
    } else {
        uint8_t lz = clz32(static_cast<uint32_t>(res_mant) << 16);
        if (lz > 9) {
            uint8_t shift = lz - 9;
            res_mant <<= shift;
            res_exp -= shift;
        }
    }

    uint8_t frac = (res_mant >> 4) & 0x03;
    if ((res_mant >> 3) & 1) {
        frac++;
        if (frac == 4) {
            frac = 0;
            res_exp++;
        }
    }

    int16_t biased_exp = res_exp + 15;
    if (biased_exp >= 31) {
        out.result = res_sign ? FP8_E5M2_NEG_INF : FP8_E5M2_POS_INF;
        out.status = FpuStatus::OVERFLOW | FpuStatus::INEXACT;
        return out;
    }
    if (biased_exp <= 0) {
        out.result = res_sign ? FP8_E5M2_NEG_ZERO : FP8_E5M2_POS_ZERO;
        out.status = FpuStatus::UNDERFLOW | FpuStatus::INEXACT;
        return out;
    }

    out.result = (res_sign << 7) | (static_cast<uint8_t>(biased_exp & 0x1F) << 2) | (frac & 0x03);
    return out;
}

} // namespace vph
