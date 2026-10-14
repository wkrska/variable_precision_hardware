#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp32_ieee_core.hpp"
#include "pipeline_regs.hpp"

namespace vph {

/**
 * @brief Stage 1 Combinational Step: Unpack, Decode & Alignment/Multiplier Preparation
 */
inline Stage1Reg fp32_stage1(const FpRequest<4>& req, bool valid) {
    Stage1Reg s1{};
    s1.valid = valid;
    s1.eof   = req.eof;
    s1.tag   = req.tag;
    s1.opcode = req.opcode;
    s1.rounding_mode = req.rounding_mode;

    if (!valid) return s1;

    s1.raw_a = bytes_to_u32(req.op_a);
    s1.raw_b = bytes_to_u32(req.op_b);

    // Fast-path unary operations (ABS, NEG)
    if (req.opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        s1.is_special = true;
        s1.special_result = s1.raw_a & 0x7FFFFFFF;
        s1.special_status = FpuStatus::OK;
        return s1;
    }
    if (req.opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        s1.is_special = true;
        s1.special_result = s1.raw_a ^ 0x80000000;
        s1.special_status = FpuStatus::OK;
        return s1;
    }

    Fp32Unpacked a = fp32_unpack(s1.raw_a);
    Fp32Unpacked b = fp32_unpack(s1.raw_b);

    if (req.opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    // MULTIPLICATION PATH DECODE
    if (req.opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        req.opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;
        s1.mul_sign = res_sign;

        if (a.is_nan || b.is_nan) {
            s1.is_special = true;
            s1.special_result = FP32_QNAN;
            s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return s1;
        }
        if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
            s1.is_special = true;
            s1.special_result = FP32_QNAN;
            s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return s1;
        }
        if (a.is_inf || b.is_inf) {
            s1.is_special = true;
            s1.special_result = (res_sign ? FP32_NEG_INF : FP32_POS_INF);
            s1.special_status = FpuStatus::IS_INF;
            return s1;
        }
        if (a.is_zero || b.is_zero) {
            s1.is_special = true;
            s1.special_result = (res_sign ? FP32_NEG_ZERO : FP32_POS_ZERO);
            s1.special_status = FpuStatus::IS_ZERO;
            return s1;
        }

        s1.mul_exp = a.exp + b.exp - 127;
        s1.mul_mant_a = a.mant;
        s1.mul_mant_b = b.mant;
        return s1;
    }

    // ADDITION / SUBTRACTION PATH DECODE
    if (a.is_nan || b.is_nan) {
        s1.is_special = true;
        s1.special_result = FP32_QNAN;
        s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return s1;
    }
    if (a.is_inf && b.is_inf) {
        if (a.sign != b.sign) {
            s1.is_special = true;
            s1.special_result = FP32_QNAN;
            s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        } else {
            s1.is_special = true;
            s1.special_result = (a.sign ? FP32_NEG_INF : FP32_POS_INF);
            s1.special_status = FpuStatus::IS_INF;
        }
        return s1;
    }
    if (a.is_inf) {
        s1.is_special = true;
        s1.special_result = (a.sign ? FP32_NEG_INF : FP32_POS_INF);
        s1.special_status = FpuStatus::IS_INF;
        return s1;
    }
    if (b.is_inf) {
        s1.is_special = true;
        s1.special_result = (b.sign ? FP32_NEG_INF : FP32_POS_INF);
        s1.special_status = FpuStatus::IS_INF;
        return s1;
    }
    if (a.is_zero && b.is_zero) {
        uint8_t res_sign = (a.sign && b.sign) ? 1 : 0;
        s1.is_special = true;
        s1.special_result = (res_sign ? FP32_NEG_ZERO : FP32_POS_ZERO);
        s1.special_status = FpuStatus::IS_ZERO;
        return s1;
    }
    if (a.is_zero) {
        s1.is_special = true;
        s1.special_result = fp32_pack(b.sign, b.exp, b.mant & 0x7FFFFF);
        s1.special_status = FpuStatus::OK;
        return s1;
    }
    if (b.is_zero) {
        s1.is_special = true;
        s1.special_result = fp32_pack(a.sign, a.exp, a.mant & 0x7FFFFF);
        s1.special_status = FpuStatus::OK;
        return s1;
    }

    // Magnitude comparison to determine alignment direction
    bool a_larger = false;
    if (a.exp > b.exp) {
        a_larger = true;
    } else if (a.exp == b.exp) {
        a_larger = (a.mant >= b.mant);
    }

    Fp32Unpacked large = a_larger ? a : b;
    Fp32Unpacked small = a_larger ? b : a;

    s1.large_sign = large.sign;
    s1.large_exp  = large.exp;
    s1.large_mant = large.mant;

    s1.small_sign = small.sign;
    s1.small_exp  = small.exp;
    s1.small_mant = small.mant;

    s1.exp_diff   = large.exp - small.exp;
    s1.eff_sub    = (large.sign != small.sign);

    return s1;
}

/**
 * @brief Stage 2 Combinational Step: Alignment Shift & Multiplier Execution
 */
inline Stage2Reg fp32_stage2(const Stage1Reg& s1) {
    Stage2Reg s2{};
    s2.valid = s1.valid;
    s2.eof   = s1.eof;
    s2.tag   = s1.tag;
    s2.opcode = s1.opcode;
    s2.rounding_mode = s1.rounding_mode;
    s2.is_special = s1.is_special;
    s2.special_result = s1.special_result;
    s2.special_status = s1.special_status;

    if (!s1.valid || s1.is_special) return s2;

    // MULTIPLICATION: 24-bit x 24-bit multiplier -> 48 bits
    if (s1.opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        s1.opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        s2.mul_sign = s1.mul_sign;
        s2.mul_exp  = s1.mul_exp;
        s2.mul_prod = static_cast<uint64_t>(s1.mul_mant_a) * static_cast<uint64_t>(s1.mul_mant_b);
        return s2;
    }

    // ADD/SUB: Alignment shift with guard, round, and sticky bits
    s2.large_sign = s1.large_sign;
    s2.large_exp  = s1.large_exp;
    s2.eff_sub    = s1.eff_sub;

    s2.mant_large_ext = static_cast<uint64_t>(s1.large_mant) << 3;

    uint64_t small_ext = static_cast<uint64_t>(s1.small_mant) << 3;
    uint8_t sticky = 0;
    if (s1.exp_diff > 0) {
        if (s1.exp_diff >= 32) {
            sticky = (s1.small_mant != 0) ? 1 : 0;
            small_ext = 0;
        } else {
            uint32_t mask = (1U << s1.exp_diff) - 1;
            sticky = ((s1.small_mant & mask) != 0) ? 1 : 0;
            small_ext >>= s1.exp_diff;
        }
    }
    s2.mant_small_ext = small_ext | sticky;

    return s2;
}

/**
 * @brief Stage 3 Combinational Step: Significand ALU & Leading-Zero Detection
 */
inline Stage3Reg fp32_stage3(const Stage2Reg& s2) {
    Stage3Reg s3{};
    s3.valid = s2.valid;
    s3.eof   = s2.eof;
    s3.tag   = s2.tag;
    s3.opcode = s2.opcode;
    s3.rounding_mode = s2.rounding_mode;
    s3.is_special = s2.is_special;
    s3.special_result = s2.special_result;
    s3.special_status = s2.special_status;

    if (!s2.valid || s2.is_special) return s3;

    // MULTIPLICATION PATH
    if (s2.opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        s2.opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        s3.res_sign = s2.mul_sign;
        if (s2.mul_prod & (1ULL << 47)) {
            // Product in [2.0, 4.0): normal range
            s3.res_exp  = s2.mul_exp + 1;
            s3.res_mant = static_cast<uint32_t>(s2.mul_prod >> 24);
            s3.guard    = static_cast<uint8_t>((s2.mul_prod >> 23) & 1);
            s3.round    = static_cast<uint8_t>((s2.mul_prod >> 22) & 1);
            s3.sticky   = (s2.mul_prod & 0x00000000003FFFFFULL) ? 1 : 0;
            s3.low_residual = static_cast<uint32_t>(s2.mul_prod & 0xFFFFFF);
        } else {
            // Product in [1.0, 2.0)
            s3.res_exp  = s2.mul_exp;
            s3.res_mant = static_cast<uint32_t>(s2.mul_prod >> 23);
            s3.guard    = static_cast<uint8_t>((s2.mul_prod >> 22) & 1);
            s3.round    = static_cast<uint8_t>((s2.mul_prod >> 21) & 1);
            s3.sticky   = (s2.mul_prod & 0x00000000001FFFFFULL) ? 1 : 0;
            s3.low_residual = static_cast<uint32_t>(s2.mul_prod & 0x7FFFFF);
        }
        return s3;
    }

    // ADD/SUB PATH
    s3.res_sign = s2.large_sign;
    uint64_t mant_res;
    if (s2.eff_sub) {
        mant_res = s2.mant_large_ext - s2.mant_small_ext;
    } else {
        mant_res = s2.mant_large_ext + s2.mant_small_ext;
    }

    if (mant_res == 0) {
        s3.is_special = true;
        s3.special_result = FP32_POS_ZERO;
        s3.special_status = FpuStatus::IS_ZERO;
        return s3;
    }

    int16_t exp_res = s2.large_exp;

    if (!s2.eff_sub) {
        if (mant_res & (1ULL << 27)) {
            mant_res >>= 1;
            exp_res++;
        }
    } else {
        uint32_t upper = static_cast<uint32_t>(mant_res >> 3);
        if (upper != 0) {
            uint8_t lz = clz32(upper) - 8;
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

    s3.res_exp  = exp_res;
    s3.res_mant = static_cast<uint32_t>(mant_res >> 3);
    s3.guard    = static_cast<uint8_t>((mant_res >> 2) & 1);
    s3.round    = static_cast<uint8_t>((mant_res >> 1) & 1);
    s3.sticky   = static_cast<uint8_t>(mant_res & 1);
    s3.low_residual = static_cast<uint32_t>(mant_res & 0x7);

    return s3;
}

/**
 * @brief Stage 4 Combinational Step: Rounding, Range Clamp & Output Packet Generation
 */
inline Stage4Reg fp32_stage4(const Stage3Reg& s3) {
    Stage4Reg s4{};
    s4.valid = s3.valid;
    s4.resp.eof    = s3.eof;
    s4.resp.tag    = s3.tag;
    s4.resp.opcode = s3.opcode;
    s4.resp.status = FpuStatus::OK;
    u32_to_bytes(0, s4.resp.residual);

    if (!s3.valid) return s4;

    if (s3.is_special) {
        u32_to_bytes(s3.special_result, s4.resp.result);
        s4.resp.status = s3.special_status;
        return s4;
    }

    // IEEE 754 Round to Nearest Even (RNE)
    uint32_t final_mant = s3.res_mant;
    int16_t  final_exp  = s3.res_exp;

    bool lsb = (final_mant & 1);
    bool round_up = s3.guard && (s3.round || s3.sticky || lsb);
    if (round_up) {
        final_mant++;
        if (final_mant & (1U << 24)) {
            final_mant >>= 1;
            final_exp++;
        }
    }

    if (s3.guard || s3.round || s3.sticky) {
        s4.resp.status |= FpuStatus::INEXACT;
    }

    uint32_t result_u32;
    if (final_exp >= 255) {
        result_u32 = (s3.res_sign ? FP32_NEG_INF : FP32_POS_INF);
        s4.resp.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
    } else if (final_exp <= 0) {
        result_u32 = (s3.res_sign ? FP32_NEG_ZERO : FP32_POS_ZERO);
        s4.resp.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
    } else {
        result_u32 = fp32_pack(s3.res_sign, final_exp, final_mant & 0x7FFFFF);
    }

    u32_to_bytes(result_u32, s4.resp.result);

    // Format residual for augmented operations (MUL_HILO / ADD_HILO)
    if (s3.opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO) && s3.low_residual != 0) {
        uint8_t lz = clz32(s3.low_residual) - 8;
        int16_t res_exp = final_exp - 24 - lz;
        if (res_exp > 0 && res_exp < 255) {
            uint32_t norm_frac = (s3.low_residual << (lz + 1)) & 0x7FFFFF;
            uint32_t res_u32 = fp32_pack(s3.res_sign, res_exp, norm_frac);
            u32_to_bytes(res_u32, s4.resp.residual);
        }
    }

    return s4;
}

/**
 * @brief Standalone FP32 CPU IP Kernel
 * 
 * Parameterized by input pipe `InPipe` and output pipe `OutPipe`.
 * Executes an explicit 4-stage pipeline with Initiation Interval = 1.
 */
template <typename InPipe, typename OutPipe>
struct Fp32CpuIpKernel {
    void operator()() const {
        Stage1Reg r1{};
        Stage2Reg r2{};
        Stage3Reg r3{};
        Stage4Reg r4{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            // Stage 4: Write to output pipe if valid
            if (r4.valid) {
                OutPipe::write(r4.resp);
                if (r4.resp.eof) {
                    active = false;
                }
            }

            // Advance explicit pipeline registers (feed-forward, cycle-accurate)
            r4 = fp32_stage4(r3);
            r3 = fp32_stage3(r2);
            r2 = fp32_stage2(r1);

            // Stage 1: Read from input pipe
            bool has_data = false;
            FpRequest<4> req = InPipe::read(has_data);
            r1 = fp32_stage1(req, has_data);
        }
    }
};

} // namespace vph
