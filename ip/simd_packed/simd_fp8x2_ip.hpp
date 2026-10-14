#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp8_core.hpp"
#include "fp16_ieee_core.hpp"
#include "ip/fp8_cpu/fp8_cpu_ip.hpp"
#include "ip/fp16_cpu/fp16_cpu_ip.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"

namespace vph {

/**
 * @brief Explicit Pipeline Stages for Dual-Mode 16-bit SIMD/Scalar IP
 * 
 * In Packed SIMD mode: Executes 2 parallel FP8 lanes (Lane 0: byte 0, Lane 1: byte 1).
 * In Full Precision mode: Executes 1 scalar FP16 operation across the full 16-bit payload (bytes 0..1).
 * Mode is dynamically decoded per-cycle via is_scalar_mode(opcode, rounding_mode).
 */
struct SimdFp8x2Stage1Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage1Reg8  lane0;
    Stage1Reg8  lane1;
    Stage1Reg16 scalar_s1;
};

struct SimdFp8x2Stage2Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage2Reg8  lane0;
    Stage2Reg8  lane1;
    Stage2Reg16 scalar_s2;
};

struct SimdFp8x2Stage3Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage2Reg8  lane0;
    Stage2Reg8  lane1;
    Stage3Reg16 scalar_s3;
};

struct SimdFp8x2Stage4Reg {
    bool          valid;
    FpResponse<2> resp;
};

inline SimdFp8x2Stage1Reg simd_fp8x2_stage1(const FpRequest<2>& req, bool valid) {
    SimdFp8x2Stage1Reg s1{};
    s1.valid     = valid;
    s1.eof       = req.eof;
    s1.tag       = req.tag;
    s1.opcode    = req.opcode;
    s1.is_scalar = is_scalar_mode(req.opcode, req.rounding_mode);

    if (!valid) return s1;

    if (s1.is_scalar) {
        FpRequest<2> req_scalar = req;
        req_scalar.opcode = get_base_opcode(req.opcode);
        s1.scalar_s1 = fp16_stage1(req_scalar, true);
    } else {
        FpRequest<1> req_l0{};
        req_l0.eof           = req.eof;
        req_l0.tag           = req.tag;
        req_l0.opcode        = get_base_opcode(req.opcode);
        req_l0.rounding_mode = req.rounding_mode;
        req_l0.op_a[0]       = req.op_a[0];
        req_l0.op_b[0]       = req.op_b[0];

        FpRequest<1> req_l1{};
        req_l1.eof           = req.eof;
        req_l1.tag           = req.tag;
        req_l1.opcode        = get_base_opcode(req.opcode);
        req_l1.rounding_mode = req.rounding_mode;
        req_l1.op_a[0]       = req.op_a[1];
        req_l1.op_b[0]       = req.op_b[1];

        s1.lane0 = fp8_stage1(req_l0, true);
        s1.lane1 = fp8_stage1(req_l1, true);
    }

    return s1;
}

inline SimdFp8x2Stage2Reg simd_fp8x2_stage2(const SimdFp8x2Stage1Reg& s1) {
    SimdFp8x2Stage2Reg s2{};
    s2.valid     = s1.valid;
    s2.eof       = s1.eof;
    s2.tag       = s1.tag;
    s2.opcode    = s1.opcode;
    s2.is_scalar = s1.is_scalar;

    if (!s1.valid) return s2;

    if (s1.is_scalar) {
        s2.scalar_s2 = fp16_stage2(s1.scalar_s1);
    } else {
        s2.lane0 = fp8_stage2(s1.lane0);
        s2.lane1 = fp8_stage2(s1.lane1);
    }

    return s2;
}

inline SimdFp8x2Stage3Reg simd_fp8x2_stage3(const SimdFp8x2Stage2Reg& s2) {
    SimdFp8x2Stage3Reg s3{};
    s3.valid     = s2.valid;
    s3.eof       = s2.eof;
    s3.tag       = s2.tag;
    s3.opcode    = s2.opcode;
    s3.is_scalar = s2.is_scalar;

    if (!s2.valid) return s3;

    if (s2.is_scalar) {
        s3.scalar_s3 = fp16_stage3(s2.scalar_s2);
    } else {
        s3.lane0 = s2.lane0;
        s3.lane1 = s2.lane1;
    }

    return s3;
}

inline SimdFp8x2Stage4Reg simd_fp8x2_stage4(const SimdFp8x2Stage3Reg& s3) {
    SimdFp8x2Stage4Reg s4{};
    s4.valid       = s3.valid;
    s4.resp.eof    = s3.eof;
    s4.resp.tag    = s3.tag;
    s4.resp.opcode = s3.opcode;

    if (!s3.valid) return s4;

    if (s3.is_scalar) {
        Stage4Reg16 sc_out = fp16_stage4(s3.scalar_s3);
        s4.resp.result[0]   = sc_out.resp.result[0];
        s4.resp.result[1]   = sc_out.resp.result[1];
        s4.resp.residual[0] = sc_out.resp.residual[0];
        s4.resp.residual[1] = sc_out.resp.residual[1];
        s4.resp.status      = sc_out.resp.status;
    } else {
        Stage3Reg8 l0_out = fp8_stage3(s3.lane0);
        Stage3Reg8 l1_out = fp8_stage3(s3.lane1);

        s4.resp.result[0]   = l0_out.resp.result[0];
        s4.resp.residual[0] = l0_out.resp.residual[0];

        s4.resp.result[1]   = l1_out.resp.result[0];
        s4.resp.residual[1] = l1_out.resp.residual[0];

        s4.resp.status = l0_out.resp.status | l1_out.resp.status;
    }

    return s4;
}

/**
 * @brief 2x FP8 Packed SIMD IP Kernel (16-bit Container)
 */
template <typename InPipe, typename OutPipe>
struct SimdFp8x2IpKernel {
    void operator()() const {
        SimdFp8x2Stage1Reg r1{};
        SimdFp8x2Stage2Reg r2{};
        SimdFp8x2Stage3Reg r3{};
        SimdFp8x2Stage4Reg r4{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            if (r4.valid) {
                OutPipe::write(r4.resp);
                if (r4.resp.eof) {
                    active = false;
                }
            }

            r4 = simd_fp8x2_stage4(r3);
            r3 = simd_fp8x2_stage3(r2);
            r2 = simd_fp8x2_stage2(r1);

            bool has_data = false;
            FpRequest<2> req = InPipe::read(has_data);
            r1 = simd_fp8x2_stage1(req, has_data);
        }
    }
};


// Pipe types and kernel wrappers
class SimdFp8x2InPipeId;
class SimdFp8x2OutPipeId;

using SimdFp8x2InPipe  = FpuPipe<SimdFp8x2InPipeId,  FpRequest<2>,  16>;
using SimdFp8x2OutPipe = FpuPipe<SimdFp8x2OutPipeId, FpResponse<2>, 16>;

class SimdFp8x2ProducerTask;
class SimdFp8x2IpTask;
class SimdFp8x2ConsumerTask;

template <typename InAccessor>
struct SimdFp8x2ProducerKernel : FpProducerKernel<SimdFp8x2InPipe, InAccessor, 2> {
    using FpProducerKernel<SimdFp8x2InPipe, InAccessor, 2>::FpProducerKernel;
};
template <typename InAccessor>
SimdFp8x2ProducerKernel(InAccessor, uint32_t) -> SimdFp8x2ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct SimdFp8x2ConsumerKernel : FpConsumerKernel<SimdFp8x2OutPipe, OutAccessor, 2> {
    using FpConsumerKernel<SimdFp8x2OutPipe, OutAccessor, 2>::FpConsumerKernel;
};
template <typename OutAccessor>
SimdFp8x2ConsumerKernel(OutAccessor) -> SimdFp8x2ConsumerKernel<OutAccessor>;

} // namespace vph
