#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp64_ieee_core.hpp"

namespace vph {

struct Stage1Reg64 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode;

    uint64_t raw_a;
    uint64_t raw_b;

    bool     is_special;
    uint64_t special_result;
    uint8_t  special_status;

    // ADD/SUB prep
    bool     large_sign;
    int16_t  large_exp;
    uint64_t large_mant;
    bool     small_sign;
    int16_t  small_exp;
    uint64_t small_mant;
    int16_t  exp_diff;
    bool     eff_sub;

    // MUL prep
    uint8_t  mul_sign;
    int16_t  mul_exp;
    uint64_t mul_ah;
    uint64_t mul_al;
    uint64_t mul_bh;
    uint64_t mul_bl;
};

struct Stage2Reg64 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode;

    bool     is_special;
    uint64_t special_result;
    uint8_t  special_status;

    // ADD/SUB extended operands (56-bit)
    bool     large_sign;
    int16_t  large_exp;
    uint64_t mant_large_ext;
    uint64_t mant_small_ext;
    bool     eff_sub;

    // MUL partial product sums
    uint8_t  mul_sign;
    int16_t  mul_exp;
    uint64_t mul_high_sum;
    uint64_t mul_low_bits;
};

struct Stage3Reg64 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode;

    bool     is_special;
    uint64_t special_result;
    uint8_t  special_status;

    uint8_t  res_sign;
    int16_t  res_exp;
    uint64_t res_mant;
    uint8_t  guard;
    uint8_t  round;
    uint8_t  sticky;
    uint64_t low_residual;
};

struct Stage4Reg64 {
    bool          valid;
    FpResponse<8> resp;
};

inline Stage1Reg64 fp64_stage1(const FpRequest<8>& req, bool valid) {
    Stage1Reg64 s1{};
    s1.valid = valid;
    s1.eof   = req.eof;
    s1.tag   = req.tag;
    s1.opcode = req.opcode;
    s1.rounding_mode = req.rounding_mode;

    if (!valid) return s1;

    s1.raw_a = bytes_to_u64(req.op_a);
    s1.raw_b = bytes_to_u64(req.op_b);

    if (req.opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        s1.is_special = true;
        s1.special_result = s1.raw_a & 0x7FFFFFFFFFFFFFFFULL;
        s1.special_status = FpuStatus::OK;
        return s1;
    }
    if (req.opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        s1.is_special = true;
        s1.special_result = s1.raw_a ^ 0x8000000000000000ULL;
        s1.special_status = FpuStatus::OK;
        return s1;
    }

    Fp64Unpacked a = fp64_unpack(s1.raw_a);
    Fp64Unpacked b = fp64_unpack(s1.raw_b);

    if (req.opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    if (req.opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        req.opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;
        s1.mul_sign = res_sign;

        if (a.is_nan || b.is_nan) {
            s1.is_special = true;
            s1.special_result = FP64_QNAN;
            s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return s1;
        }
        if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
            s1.is_special = true;
            s1.special_result = FP64_QNAN;
            s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return s1;
        }
        if (a.is_inf || b.is_inf) {
            s1.is_special = true;
            s1.special_result = (res_sign ? FP64_NEG_INF : FP64_POS_INF);
            s1.special_status = FpuStatus::IS_INF;
            return s1;
        }
        if (a.is_zero || b.is_zero) {
            s1.is_special = true;
            s1.special_result = (res_sign ? FP64_NEG_ZERO : FP64_POS_ZERO);
            s1.special_status = FpuStatus::IS_ZERO;
            return s1;
        }

        s1.mul_exp = a.exp + b.exp - 1023;
        s1.mul_ah  = a.mant >> 26;
        s1.mul_al  = a.mant & 0x3FFFFFFULL;
        s1.mul_bh  = b.mant >> 26;
        s1.mul_bl  = b.mant & 0x3FFFFFFULL;
        return s1;
    }

    // ADD / SUB
    if (a.is_nan || b.is_nan) {
        s1.is_special = true;
        s1.special_result = FP64_QNAN;
        s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return s1;
    }
    if (a.is_inf && b.is_inf) {
        if (a.sign != b.sign) {
            s1.is_special = true;
            s1.special_result = FP64_QNAN;
            s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        } else {
            s1.is_special = true;
            s1.special_result = (a.sign ? FP64_NEG_INF : FP64_POS_INF);
            s1.special_status = FpuStatus::IS_INF;
        }
        return s1;
    }
    if (a.is_inf) {
        s1.is_special = true;
        s1.special_result = (a.sign ? FP64_NEG_INF : FP64_POS_INF);
        s1.special_status = FpuStatus::IS_INF;
        return s1;
    }
    if (b.is_inf) {
        s1.is_special = true;
        s1.special_result = (b.sign ? FP64_NEG_INF : FP64_POS_INF);
        s1.special_status = FpuStatus::IS_INF;
        return s1;
    }
    if (a.is_zero && b.is_zero) {
        uint8_t res_sign = (a.sign && b.sign) ? 1 : 0;
        s1.is_special = true;
        s1.special_result = (res_sign ? FP64_NEG_ZERO : FP64_POS_ZERO);
        s1.special_status = FpuStatus::IS_ZERO;
        return s1;
    }
    if (a.is_zero) {
        s1.is_special = true;
        s1.special_result = fp64_pack(b.sign, b.exp, b.mant & 0x000FFFFFFFFFFFFFULL);
        s1.special_status = FpuStatus::OK;
        return s1;
    }
    if (b.is_zero) {
        s1.is_special = true;
        s1.special_result = fp64_pack(a.sign, a.exp, a.mant & 0x000FFFFFFFFFFFFFULL);
        s1.special_status = FpuStatus::OK;
        return s1;
    }

    bool a_larger = false;
    if (a.exp > b.exp) {
        a_larger = true;
    } else if (a.exp == b.exp) {
        a_larger = (a.mant >= b.mant);
    }

    Fp64Unpacked large = a_larger ? a : b;
    Fp64Unpacked small = a_larger ? b : a;

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

inline Stage2Reg64 fp64_stage2(const Stage1Reg64& s1) {
    Stage2Reg64 s2{};
    s2.valid = s1.valid;
    s2.eof   = s1.eof;
    s2.tag   = s1.tag;
    s2.opcode = s1.opcode;
    s2.rounding_mode = s1.rounding_mode;
    s2.is_special = s1.is_special;
    s2.special_result = s1.special_result;
    s2.special_status = s1.special_status;

    if (!s1.valid || s1.is_special) return s2;

    if (s1.opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        s1.opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        s2.mul_sign = s1.mul_sign;
        s2.mul_exp  = s1.mul_exp;

        uint64_t pll = s1.mul_al * s1.mul_bl;
        uint64_t phl = s1.mul_ah * s1.mul_bl;
        uint64_t plh = s1.mul_al * s1.mul_bh;
        uint64_t phh = s1.mul_ah * s1.mul_bh;

        uint64_t mid_sum  = phl + plh + (pll >> 26);
        s2.mul_high_sum   = phh + (mid_sum >> 26);
        s2.mul_low_bits   = (pll & 0x3FFFFFFULL) | ((mid_sum & 0x3FFFFFFULL) << 26);
        return s2;
    }

    s2.large_sign = s1.large_sign;
    s2.large_exp  = s1.large_exp;
    s2.eff_sub    = s1.eff_sub;

    s2.mant_large_ext = s1.large_mant << 3;

    uint64_t small_ext = s1.small_mant << 3;
    uint8_t sticky = 0;
    if (s1.exp_diff > 0) {
        if (s1.exp_diff >= 60) {
            sticky = (s1.small_mant != 0) ? 1 : 0;
            small_ext = 0;
        } else {
            uint64_t mask = (1ULL << s1.exp_diff) - 1;
            sticky = ((s1.small_mant & mask) != 0) ? 1 : 0;
            small_ext >>= s1.exp_diff;
        }
    }
    s2.mant_small_ext = small_ext | sticky;

    return s2;
}

inline Stage3Reg64 fp64_stage3(const Stage2Reg64& s2) {
    Stage3Reg64 s3{};
    s3.valid = s2.valid;
    s3.eof   = s2.eof;
    s3.tag   = s2.tag;
    s3.opcode = s2.opcode;
    s3.rounding_mode = s2.rounding_mode;
    s3.is_special = s2.is_special;
    s3.special_result = s2.special_result;
    s3.special_status = s2.special_status;

    if (!s2.valid || s2.is_special) return s3;

    if (s2.opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        s2.opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        s3.res_sign = s2.mul_sign;

        if (s2.mul_high_sum & (1ULL << 53)) {
            s3.res_exp  = s2.mul_exp + 1;
            s3.res_mant = s2.mul_high_sum >> 1;
            s3.guard    = static_cast<uint8_t>(s2.mul_high_sum & 1);
            s3.round    = static_cast<uint8_t>((s2.mul_low_bits >> 51) & 1);
            s3.sticky   = (s2.mul_low_bits & 0x7FFFFFFFFFFFFULL) ? 1 : 0;
        } else {
            s3.res_exp  = s2.mul_exp;
            s3.res_mant = s2.mul_high_sum & 0x001FFFFFFFFFFFFFULL;
            s3.guard    = static_cast<uint8_t>((s2.mul_low_bits >> 51) & 1);
            s3.round    = static_cast<uint8_t>((s2.mul_low_bits >> 50) & 1);
            s3.sticky   = (s2.mul_low_bits & 0x3FFFFFFFFFFFFULL) ? 1 : 0;
        }
        s3.low_residual = s2.mul_low_bits & 0xFFFFFFFFULL;
        return s3;
    }

    s3.res_sign = s2.large_sign;
    uint64_t mant_res;
    if (s2.eff_sub) {
        mant_res = s2.mant_large_ext - s2.mant_small_ext;
    } else {
        mant_res = s2.mant_large_ext + s2.mant_small_ext;
    }

    if (mant_res == 0) {
        s3.is_special = true;
        s3.special_result = FP64_POS_ZERO;
        s3.special_status = FpuStatus::IS_ZERO;
        return s3;
    }

    int16_t exp_res = s2.large_exp;

    if (!s2.eff_sub) {
        if (mant_res & (1ULL << 56)) {
            mant_res >>= 1;
            exp_res++;
        }
    } else {
        uint64_t upper = mant_res >> 3;
        if (upper != 0) {
            uint8_t lz = clz64(upper) - 11;
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
    s3.res_mant = mant_res >> 3;
    s3.guard    = static_cast<uint8_t>((mant_res >> 2) & 1);
    s3.round    = static_cast<uint8_t>((mant_res >> 1) & 1);
    s3.sticky   = static_cast<uint8_t>(mant_res & 1);
    s3.low_residual = mant_res & 0x7;

    return s3;
}

inline Stage4Reg64 fp64_stage4(const Stage3Reg64& s3) {
    Stage4Reg64 s4{};
    s4.valid = s3.valid;
    s4.resp.eof    = s3.eof;
    s4.resp.tag    = s3.tag;
    s4.resp.opcode = s3.opcode;
    s4.resp.status = FpuStatus::OK;
    u64_to_bytes(0, s4.resp.residual);

    if (!s3.valid) return s4;

    if (s3.is_special) {
        u64_to_bytes(s3.special_result, s4.resp.result);
        s4.resp.status = s3.special_status;
        return s4;
    }

    uint64_t final_mant = s3.res_mant;
    int16_t  final_exp  = s3.res_exp;

    bool lsb = (final_mant & 1);
    bool round_up = s3.guard && (s3.round || s3.sticky || lsb);
    if (round_up) {
        final_mant++;
        if (final_mant & (1ULL << 53)) {
            final_mant >>= 1;
            final_exp++;
        }
    }

    if (s3.guard || s3.round || s3.sticky) {
        s4.resp.status |= FpuStatus::INEXACT;
    }

    uint64_t result_u64;
    if (final_exp >= 2047) {
        result_u64 = (s3.res_sign ? FP64_NEG_INF : FP64_POS_INF);
        s4.resp.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
    } else if (final_exp <= 0) {
        result_u64 = (s3.res_sign ? FP64_NEG_ZERO : FP64_POS_ZERO);
        s4.resp.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
    } else {
        result_u64 = fp64_pack(s3.res_sign, final_exp, final_mant & 0x000FFFFFFFFFFFFFULL);
    }

    u64_to_bytes(result_u64, s4.resp.result);
    return s4;
}

/**
 * @brief Standalone FP64 CPU IP Kernel with Initiation Interval = 1
 */
template <typename InPipe, typename OutPipe>
struct Fp64CpuIpKernel {
    void operator()() const {
        Stage1Reg64 r1{};
        Stage2Reg64 r2{};
        Stage3Reg64 r3{};
        Stage4Reg64 r4{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            if (r4.valid) {
                OutPipe::write(r4.resp);
                if (r4.resp.eof) {
                    active = false;
                }
            }

            r4 = fp64_stage4(r3);
            r3 = fp64_stage3(r2);
            r2 = fp64_stage2(r1);

            bool has_data = false;
            FpRequest<8> req = InPipe::read(has_data);
            r1 = fp64_stage1(req, has_data);
        }
    }
};

} // namespace vph
