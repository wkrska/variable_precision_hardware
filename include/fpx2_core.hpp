#pragma once

#include <cstdint>
#include "fpu_types.hpp"
#include "bit_ops.hpp"

namespace vph {

/**
 * @brief FPX2 Shared-Exponent Multi-Limb Floating-Point Format
 * 
 * Based on architectural specification from 'MIxed Float Hardware Support.md':
 * 64-bit composite word:
 *   - Bit 63:       1-bit sign
 *   - Bits 62..52: 11-bit shared exponent (bias = 1023)
 *   - Bits 51..48:  4-bit metadata / class flags
 *   - Bits 47..0:  48-bit significand, decomposed into two 24-bit limbs:
 *                    - M_hi = bits 47..24 (replaying 24-bit FP32 significand)
 *                    - M_lo = bits 23..0  (replaying 24-bit FP32 significand)
 */
struct Fpx2Unpacked {
    uint8_t  sign;
    int16_t  exp;       // 11-bit biased exponent (0..2047)
    uint8_t  meta;      // 4-bit metadata
    uint64_t mant48;    // 48-bit significand (with implicit/explicit bits)
    uint32_t limb_hi;   // 24-bit upper limb (M_hi)
    uint32_t limb_lo;   // 24-bit lower limb (M_lo)
    bool     is_zero;
    bool     is_inf;
    bool     is_nan;
};

inline Fpx2Unpacked fpx2_unpack(uint64_t raw) {
    Fpx2Unpacked u;
    u.sign   = static_cast<uint8_t>((raw >> 63) & 0x1);
    u.exp    = static_cast<int16_t>((raw >> 52) & 0x7FF);
    u.meta   = static_cast<uint8_t>((raw >> 48) & 0x0F);
    u.mant48 = raw & 0x0000FFFFFFFFFFFFULL;

    u.is_nan  = (u.exp == 0x7FF) && (u.mant48 != 0);
    u.is_inf  = (u.exp == 0x7FF) && (u.mant48 == 0);
    u.is_zero = (u.exp == 0) && (u.mant48 == 0);

    // Decompose 48-bit significand into two 24-bit limbs
    u.limb_hi = static_cast<uint32_t>((u.mant48 >> 24) & 0x00FFFFFF);
    u.limb_lo = static_cast<uint32_t>(u.mant48 & 0x00FFFFFF);

    return u;
}

inline uint64_t fpx2_pack(uint8_t sign, int16_t exp, uint8_t meta, uint64_t mant48) {
    uint64_t raw = (static_cast<uint64_t>(sign & 0x1) << 63);
    raw |= (static_cast<uint64_t>(exp & 0x7FF) << 52);
    raw |= (static_cast<uint64_t>(meta & 0x0F) << 48);
    raw |= (mant48 & 0x0000FFFFFFFFFFFFULL);
    return raw;
}

constexpr uint64_t FPX2_POS_ZERO = 0x0000000000000000ULL;
constexpr uint64_t FPX2_NEG_ZERO = 0x8000000000000000ULL;
constexpr uint64_t FPX2_POS_INF  = 0x7FF0000000000000ULL;
constexpr uint64_t FPX2_NEG_INF  = 0xFFF0000000000000ULL;
constexpr uint64_t FPX2_QNAN     = 0x7FF8000000000000ULL;

struct Fpx2AluResult {
    uint64_t result;
    uint64_t residual;
    uint8_t  status;
};

/**
 * @brief Bit-Level Multi-Limb FPX2 ALU
 * 
 * Replays 24x24 FP32 significand partial products and performs 48-bit limb arithmetic.
 */
inline Fpx2AluResult fpx2_eval_alu(uint64_t raw_a, uint64_t raw_b, uint8_t opcode) {
    Fpx2AluResult out{0, 0, FpuStatus::OK};

    if (opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        out.result = raw_a & 0x7FFFFFFFFFFFFFFFULL;
        return out;
    }
    if (opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        out.result = raw_a ^ 0x8000000000000000ULL;
        return out;
    }

    Fpx2Unpacked a = fpx2_unpack(raw_a);
    Fpx2Unpacked b = fpx2_unpack(raw_b);

    if (opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    // MULTIPLICATION PATH: Four 24x24 partial products
    if (opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;

        if (a.is_nan || b.is_nan) {
            out.result = FPX2_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
            out.result = FPX2_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return out;
        }
        if (a.is_inf || b.is_inf) {
            out.result = res_sign ? FPX2_NEG_INF : FPX2_POS_INF;
            out.status = FpuStatus::IS_INF;
            return out;
        }
        if (a.is_zero || b.is_zero) {
            out.result = res_sign ? FPX2_NEG_ZERO : FPX2_POS_ZERO;
            out.status = FpuStatus::IS_ZERO;
            return out;
        }

        // Shared exponent addition
        int16_t exp_res = a.exp + b.exp - 1023;

        // Replay 24x24 partial products:
        // A = A_hi * 2^24 + A_lo
        // B = B_hi * 2^24 + B_lo
        // Product = (A_hi * B_hi) * 2^48 + (A_hi * B_lo + A_lo * B_hi) * 2^24 + (A_lo * B_lo)
        uint64_t p_hi_hi = static_cast<uint64_t>(a.limb_hi) * static_cast<uint64_t>(b.limb_hi);
        uint64_t p_hi_lo = static_cast<uint64_t>(a.limb_hi) * static_cast<uint64_t>(b.limb_lo);
        uint64_t p_lo_hi = static_cast<uint64_t>(a.limb_lo) * static_cast<uint64_t>(b.limb_hi);
        uint64_t p_lo_lo = static_cast<uint64_t>(a.limb_lo) * static_cast<uint64_t>(b.limb_lo);

        // Sum middle products
        uint64_t mid_sum = p_hi_lo + p_lo_hi + (p_lo_lo >> 24);
        uint64_t upper_prod = p_hi_hi + (mid_sum >> 24);
        uint64_t lower_prod = (mid_sum << 24) | (p_lo_lo & 0x00FFFFFF);

        // Normalization on upper_prod (up to 48 bits)
        uint8_t lz = clz64(upper_prod);
        // Desired leading 1 at bit 47 => lz = 16
        if (lz < 16) {
            uint8_t rshift = 16 - lz;
            upper_prod >>= rshift;
            exp_res += rshift;
        } else if (lz > 16) {
            uint8_t lshift = lz - 16;
            upper_prod = (upper_prod << lshift) | (lower_prod >> (64 - lshift));
            exp_res -= lshift;
        }

        if (exp_res >= 2047) {
            out.result = res_sign ? FPX2_NEG_INF : FPX2_POS_INF;
            out.status = FpuStatus::OVERFLOW | FpuStatus::IS_INF;
            return out;
        }
        if (exp_res <= 0) {
            out.result = res_sign ? FPX2_NEG_ZERO : FPX2_POS_ZERO;
            out.status = FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
            return out;
        }

        out.result   = fpx2_pack(res_sign, exp_res, 0, upper_prod);
        out.residual = fpx2_pack(res_sign, exp_res - 48, 0, lower_prod >> 16);
        return out;
    }

    // ADDITION / SUBTRACTION PATH (48-bit aligned limb arithmetic)
    if (a.is_nan || b.is_nan) {
        out.result = FPX2_QNAN;
        out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return out;
    }
    if (a.is_inf && b.is_inf) {
        if (a.sign != b.sign) {
            out.result = FPX2_QNAN;
            out.status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        } else {
            out.result = a.sign ? FPX2_NEG_INF : FPX2_POS_INF;
            out.status = FpuStatus::IS_INF;
        }
        return out;
    }
    if (a.is_inf) {
        out.result = a.sign ? FPX2_NEG_INF : FPX2_POS_INF;
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (b.is_inf) {
        out.result = b.sign ? FPX2_NEG_INF : FPX2_POS_INF;
        out.status = FpuStatus::IS_INF;
        return out;
    }
    if (a.is_zero) {
        out.result = raw_b ^ ((opcode == static_cast<uint8_t>(FpuOpcode::SUB)) ? 0x8000000000000000ULL : 0ULL);
        return out;
    }
    if (b.is_zero) {
        out.result = raw_a;
        return out;
    }

    bool a_ge_b = (a.exp > b.exp) || ((a.exp == b.exp) && (a.mant48 >= b.mant48));
    Fpx2Unpacked big = a_ge_b ? a : b;
    Fpx2Unpacked sml = a_ge_b ? b : a;

    int16_t exp_diff = big.exp - sml.exp;
    uint64_t big_m = big.mant48;
    uint64_t sml_m = (exp_diff >= 48) ? 0 : (sml.mant48 >> exp_diff);

    uint64_t res_m = 0;
    uint8_t  res_sign = big.sign;

    if (big.sign == sml.sign) {
        res_m = big_m + sml_m;
    } else {
        res_m = big_m - sml_m;
    }

    if (res_m == 0) {
        out.result = FPX2_POS_ZERO;
        out.status = FpuStatus::IS_ZERO;
        return out;
    }

    int16_t res_exp = big.exp;
    // Check overflow bit 48
    if (res_m & (1ULL << 48)) {
        res_m >>= 1;
        res_exp++;
    } else {
        uint8_t lz = clz64(res_m);
        if (lz > 16) {
            uint8_t shift = lz - 16;
            res_m <<= shift;
            res_exp -= shift;
        }
    }

    if (res_exp >= 2047) {
        out.result = res_sign ? FPX2_NEG_INF : FPX2_POS_INF;
        out.status = FpuStatus::OVERFLOW | FpuStatus::IS_INF;
        return out;
    }
    if (res_exp <= 0) {
        out.result = res_sign ? FPX2_NEG_ZERO : FPX2_POS_ZERO;
        out.status = FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
        return out;
    }

    out.result = fpx2_pack(res_sign, res_exp, 0, res_m);
    return out;
}

} // namespace vph
