#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "bf16_core.hpp"

namespace vph {

struct Stage1RegBf16 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode;

    uint16_t raw_a;
    uint16_t raw_b;

    bool     is_special;
    uint16_t special_result;
    uint8_t  special_status;

    bool     large_sign;
    int16_t  large_exp;
    uint16_t large_mant;
    bool     small_sign;
    int16_t  small_exp;
    uint16_t small_mant;
    int16_t  exp_diff;
    bool     eff_sub;

    uint8_t  mul_sign;
    int16_t  mul_exp;
    uint16_t mul_mant_a;
    uint16_t mul_mant_b;
};

struct Stage2RegBf16 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode;

    bool     is_special;
    uint16_t special_result;
    uint8_t  special_status;

    bool     large_sign;
    int16_t  large_exp;
    uint32_t mant_large_ext;
    uint32_t mant_small_ext;
    bool     eff_sub;

    uint8_t  mul_sign;
    int16_t  mul_exp;
    uint32_t mul_prod;
};

struct Stage3RegBf16 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode;

    bool     is_special;
    uint16_t special_result;
    uint8_t  special_status;

    uint8_t  res_sign;
    int16_t  res_exp;
    uint16_t res_mant;
    uint8_t  guard;
    uint8_t  round;
    uint8_t  sticky;
};

struct Stage4RegBf16 {
    bool          valid;
    FpResponse<2> resp;
};

inline Stage1RegBf16 bf16_stage1(const FpRequest<2>& req, bool valid) {
    Stage1RegBf16 s1{};
    s1.valid = valid;
    s1.eof   = req.eof;
    s1.tag   = req.tag;
    s1.opcode = req.opcode;
    s1.rounding_mode = req.rounding_mode;

    if (!valid) return s1;

    s1.raw_a = bytes_to_u16(req.op_a);
    s1.raw_b = bytes_to_u16(req.op_b);

    if (req.opcode == static_cast<uint8_t>(FpuOpcode::ABS)) {
        s1.is_special = true;
        s1.special_result = s1.raw_a & 0x7FFF;
        s1.special_status = FpuStatus::OK;
        return s1;
    }
    if (req.opcode == static_cast<uint8_t>(FpuOpcode::NEG)) {
        s1.is_special = true;
        s1.special_result = s1.raw_a ^ 0x8000;
        s1.special_status = FpuStatus::OK;
        return s1;
    }

    Bf16Unpacked a = bf16_unpack(s1.raw_a);
    Bf16Unpacked b = bf16_unpack(s1.raw_b);

    if (req.opcode == static_cast<uint8_t>(FpuOpcode::SUB)) {
        b.sign ^= 1;
    }

    if (req.opcode == static_cast<uint8_t>(FpuOpcode::MUL) ||
        req.opcode == static_cast<uint8_t>(FpuOpcode::MUL_HILO)) {
        
        uint8_t res_sign = a.sign ^ b.sign;
        s1.mul_sign = res_sign;

        if (a.is_nan || b.is_nan) {
            s1.is_special = true;
            s1.special_result = BF16_QNAN;
            s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return s1;
        }
        if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
            s1.is_special = true;
            s1.special_result = BF16_QNAN;
            s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
            return s1;
        }
        if (a.is_inf || b.is_inf) {
            s1.is_special = true;
            s1.special_result = (res_sign ? BF16_NEG_INF : BF16_POS_INF);
            s1.special_status = FpuStatus::IS_INF;
            return s1;
        }
        if (a.is_zero || b.is_zero) {
            s1.is_special = true;
            s1.special_result = (res_sign ? BF16_NEG_ZERO : BF16_POS_ZERO);
            s1.special_status = FpuStatus::IS_ZERO;
            return s1;
        }

        s1.mul_exp = a.exp + b.exp - 127;
        s1.mul_mant_a = a.mant;
        s1.mul_mant_b = b.mant;
        return s1;
    }

    // ADD / SUB
    if (a.is_nan || b.is_nan) {
        s1.is_special = true;
        s1.special_result = BF16_QNAN;
        s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        return s1;
    }
    if (a.is_inf && b.is_inf) {
        if (a.sign != b.sign) {
            s1.is_special = true;
            s1.special_result = BF16_QNAN;
            s1.special_status = FpuStatus::IS_NAN | FpuStatus::INVALID;
        } else {
            s1.is_special = true;
            s1.special_result = (a.sign ? BF16_NEG_INF : BF16_POS_INF);
            s1.special_status = FpuStatus::IS_INF;
        }
        return s1;
    }
    if (a.is_inf) {
        s1.is_special = true;
        s1.special_result = (a.sign ? BF16_NEG_INF : BF16_POS_INF);
        s1.special_status = FpuStatus::IS_INF;
        return s1;
    }
    if (b.is_inf) {
        s1.is_special = true;
        s1.special_result = (b.sign ? BF16_NEG_INF : BF16_POS_INF);
        s1.special_status = FpuStatus::IS_INF;
        return s1;
    }
    if (a.is_zero && b.is_zero) {
        uint8_t res_sign = (a.sign && b.sign) ? 1 : 0;
        s1.is_special = true;
        s1.special_result = (res_sign ? BF16_NEG_ZERO : BF16_POS_ZERO);
        s1.special_status = FpuStatus::IS_ZERO;
        return s1;
    }
    if (a.is_zero) {
        s1.is_special = true;
        s1.special_result = bf16_pack(b.sign, b.exp, b.mant & 0x7F);
        s1.special_status = FpuStatus::OK;
        return s1;
    }
    if (b.is_zero) {
        s1.is_special = true;
        s1.special_result = bf16_pack(a.sign, a.exp, a.mant & 0x7F);
        s1.special_status = FpuStatus::OK;
        return s1;
    }

    bool a_larger = false;
    if (a.exp > b.exp) {
        a_larger = true;
    } else if (a.exp == b.exp) {
        a_larger = (a.mant >= b.mant);
    }

    Bf16Unpacked large = a_larger ? a : b;
    Bf16Unpacked small = a_larger ? b : a;

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

inline Stage2RegBf16 bf16_stage2(const Stage1RegBf16& s1) {
    Stage2RegBf16 s2{};
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
        s2.mul_prod = static_cast<uint32_t>(s1.mul_mant_a) * static_cast<uint32_t>(s1.mul_mant_b);
        return s2;
    }

    s2.large_sign = s1.large_sign;
    s2.large_exp  = s1.large_exp;
    s2.eff_sub    = s1.eff_sub;

    s2.mant_large_ext = static_cast<uint32_t>(s1.large_mant) << 3;

    uint32_t small_ext = static_cast<uint32_t>(s1.small_mant) << 3;
    uint8_t sticky = 0;
    if (s1.exp_diff > 0) {
        if (s1.exp_diff >= 11) {
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

inline Stage3RegBf16 bf16_stage3(const Stage2RegBf16& s2) {
    Stage3RegBf16 s3{};
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

        if (s2.mul_prod & (1U << 15)) {
            s3.res_exp  = s2.mul_exp + 1;
            s3.res_mant = static_cast<uint16_t>(s2.mul_prod >> 8);
            s3.guard    = static_cast<uint8_t>((s2.mul_prod >> 7) & 1);
            s3.round    = static_cast<uint8_t>((s2.mul_prod >> 6) & 1);
            s3.sticky   = (s2.mul_prod & 0x3F) ? 1 : 0;
        } else {
            s3.res_exp  = s2.mul_exp;
            s3.res_mant = static_cast<uint16_t>(s2.mul_prod >> 7);
            s3.guard    = static_cast<uint8_t>((s2.mul_prod >> 6) & 1);
            s3.round    = static_cast<uint8_t>((s2.mul_prod >> 5) & 1);
            s3.sticky   = (s2.mul_prod & 0x1F) ? 1 : 0;
        }
        return s3;
    }

    s3.res_sign = s2.large_sign;
    uint32_t mant_res;
    if (s2.eff_sub) {
        mant_res = s2.mant_large_ext - s2.mant_small_ext;
    } else {
        mant_res = s2.mant_large_ext + s2.mant_small_ext;
    }

    if (mant_res == 0) {
        s3.is_special = true;
        s3.special_result = BF16_POS_ZERO;
        s3.special_status = FpuStatus::IS_ZERO;
        return s3;
    }

    int16_t exp_res = s2.large_exp;

    if (!s2.eff_sub) {
        if (mant_res & (1U << 11)) {
            mant_res >>= 1;
            exp_res++;
        }
    } else {
        uint32_t upper = mant_res >> 3;
        if (upper != 0) {
            uint8_t lz = clz32(upper) - 24;
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
    s3.res_mant = static_cast<uint16_t>(mant_res >> 3);
    s3.guard    = static_cast<uint8_t>((mant_res >> 2) & 1);
    s3.round    = static_cast<uint8_t>((mant_res >> 1) & 1);
    s3.sticky   = static_cast<uint8_t>(mant_res & 1);

    return s3;
}

inline Stage4RegBf16 bf16_stage4(const Stage3RegBf16& s3) {
    Stage4RegBf16 s4{};
    s4.valid = s3.valid;
    s4.resp.eof    = s3.eof;
    s4.resp.tag    = s3.tag;
    s4.resp.opcode = s3.opcode;
    s4.resp.status = FpuStatus::OK;
    u16_to_bytes(0, s4.resp.residual);

    if (!s3.valid) return s4;

    if (s3.is_special) {
        u16_to_bytes(s3.special_result, s4.resp.result);
        s4.resp.status = s3.special_status;
        return s4;
    }

    uint16_t final_mant = s3.res_mant;
    int16_t  final_exp  = s3.res_exp;

    bool lsb = (final_mant & 1);
    bool round_up = s3.guard && (s3.round || s3.sticky || lsb);
    if (round_up) {
        final_mant++;
        if (final_mant & (1U << 8)) {
            final_mant >>= 1;
            final_exp++;
        }
    }

    if (s3.guard || s3.round || s3.sticky) {
        s4.resp.status |= FpuStatus::INEXACT;
    }

    uint16_t result_u16;
    if (final_exp >= 255) {
        result_u16 = (s3.res_sign ? BF16_NEG_INF : BF16_POS_INF);
        s4.resp.status |= FpuStatus::OVERFLOW | FpuStatus::IS_INF;
    } else if (final_exp <= 0) {
        result_u16 = (s3.res_sign ? BF16_NEG_ZERO : BF16_POS_ZERO);
        s4.resp.status |= FpuStatus::UNDERFLOW | FpuStatus::IS_ZERO;
    } else {
        result_u16 = bf16_pack(s3.res_sign, final_exp, final_mant & 0x7F);
    }

    u16_to_bytes(result_u16, s4.resp.result);
    return s4;
}

template <typename InPipe, typename OutPipe>
struct Bf16CpuIpKernel {
    void operator()() const {
        Stage1RegBf16 r1{};
        Stage2RegBf16 r2{};
        Stage3RegBf16 r3{};
        Stage4RegBf16 r4{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            if (r4.valid) {
                OutPipe::write(r4.resp);
                if (r4.resp.eof) {
                    active = false;
                }
            }

            r4 = bf16_stage4(r3);
            r3 = bf16_stage3(r2);
            r2 = bf16_stage2(r1);

            bool has_data = false;
            FpRequest<2> req = InPipe::read(has_data);
            r1 = bf16_stage1(req, has_data);
        }
    }
};

} // namespace vph
