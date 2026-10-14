#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp16_ieee_core.hpp"
#include "fp32_ieee_core.hpp"
#include "pipeline_regs.hpp"
#include "ip/fp16_cpu/fp16_cpu_ip.hpp"
#include "ip/fp32_cpu/fp32_cpu_ip.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"

namespace vph {

/**
 * @brief Explicit Pipeline Stages for Dual-Mode 32-bit SIMD/Scalar IP
 * 
 * In Packed SIMD mode: Executes 2 parallel FP16 lanes (Lane 0: bytes 0..1, Lane 1: bytes 2..3).
 * In Full Precision mode: Executes 1 scalar FP32 operation across the full 32-bit payload (bytes 0..3).
 * Mode is dynamically decoded per-cycle via is_scalar_mode(opcode, rounding_mode).
 */
struct SimdFp16x2Stage1Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage1Reg16 lane0;
    Stage1Reg16 lane1;
    Stage1Reg   scalar_s1;
};

struct SimdFp16x2Stage2Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage2Reg16 lane0;
    Stage2Reg16 lane1;
    Stage2Reg   scalar_s2;
};

struct SimdFp16x2Stage3Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage3Reg16 lane0;
    Stage3Reg16 lane1;
    Stage3Reg   scalar_s3;
};

struct SimdFp16x2Stage4Reg {
    bool          valid;
    FpResponse<4> resp;
};

inline SimdFp16x2Stage1Reg simd_fp16x2_stage1(const FpRequest<4>& req, bool valid) {
    SimdFp16x2Stage1Reg s1{};
    s1.valid     = valid;
    s1.eof       = req.eof;
    s1.tag       = req.tag;
    s1.opcode    = req.opcode;
    s1.is_scalar = is_scalar_mode(req.opcode, req.rounding_mode);

    if (!valid) return s1;

    if (s1.is_scalar) {
        // Full precision scalar mode: Pass 4-byte container to FP32 datapath
        FpRequest<4> req_scalar = req;
        req_scalar.opcode = get_base_opcode(req.opcode);
        s1.scalar_s1 = fp32_stage1(req_scalar, true);
    } else {
        // Packed SIMD mode: Split Lane 0 (bytes 0..1) and Lane 1 (bytes 2..3)
        FpRequest<2> req_l0{};
        req_l0.eof           = req.eof;
        req_l0.tag           = req.tag;
        req_l0.opcode        = get_base_opcode(req.opcode);
        req_l0.rounding_mode = req.rounding_mode;
        req_l0.op_a[0]       = req.op_a[0];
        req_l0.op_a[1]       = req.op_a[1];
        req_l0.op_b[0]       = req.op_b[0];
        req_l0.op_b[1]       = req.op_b[1];

        FpRequest<2> req_l1{};
        req_l1.eof           = req.eof;
        req_l1.tag           = req.tag;
        req_l1.opcode        = get_base_opcode(req.opcode);
        req_l1.rounding_mode = req.rounding_mode;
        req_l1.op_a[0]       = req.op_a[2];
        req_l1.op_a[1]       = req.op_a[3];
        req_l1.op_b[0]       = req.op_b[2];
        req_l1.op_b[1]       = req.op_b[3];

        s1.lane0 = fp16_stage1(req_l0, true);
        s1.lane1 = fp16_stage1(req_l1, true);
    }

    return s1;
}

inline SimdFp16x2Stage2Reg simd_fp16x2_stage2(const SimdFp16x2Stage1Reg& s1) {
    SimdFp16x2Stage2Reg s2{};
    s2.valid     = s1.valid;
    s2.eof       = s1.eof;
    s2.tag       = s1.tag;
    s2.opcode    = s1.opcode;
    s2.is_scalar = s1.is_scalar;

    if (!s1.valid) return s2;

    if (s1.is_scalar) {
        s2.scalar_s2 = fp32_stage2(s1.scalar_s1);
    } else {
        s2.lane0 = fp16_stage2(s1.lane0);
        s2.lane1 = fp16_stage2(s1.lane1);
    }

    return s2;
}

inline SimdFp16x2Stage3Reg simd_fp16x2_stage3(const SimdFp16x2Stage2Reg& s2) {
    SimdFp16x2Stage3Reg s3{};
    s3.valid     = s2.valid;
    s3.eof       = s2.eof;
    s3.tag       = s2.tag;
    s3.opcode    = s2.opcode;
    s3.is_scalar = s2.is_scalar;

    if (!s2.valid) return s3;

    if (s2.is_scalar) {
        s3.scalar_s3 = fp32_stage3(s2.scalar_s2);
    } else {
        s3.lane0 = fp16_stage3(s2.lane0);
        s3.lane1 = fp16_stage3(s2.lane1);
    }

    return s3;
}

inline SimdFp16x2Stage4Reg simd_fp16x2_stage4(const SimdFp16x2Stage3Reg& s3) {
    SimdFp16x2Stage4Reg s4{};
    s4.valid       = s3.valid;
    s4.resp.eof    = s3.eof;
    s4.resp.tag    = s3.tag;
    s4.resp.opcode = s3.opcode;

    if (!s3.valid) return s4;

    if (s3.is_scalar) {
        Stage4Reg sc_out = fp32_stage4(s3.scalar_s3);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            s4.resp.result[i]   = sc_out.resp.result[i];
            s4.resp.residual[i] = sc_out.resp.residual[i];
        }
        s4.resp.status = sc_out.resp.status;
    } else {
        Stage4Reg16 l0_out = fp16_stage4(s3.lane0);
        Stage4Reg16 l1_out = fp16_stage4(s3.lane1);

        // Pack Lane 0 into bytes 0..1
        s4.resp.result[0]   = l0_out.resp.result[0];
        s4.resp.result[1]   = l0_out.resp.result[1];
        s4.resp.residual[0] = l0_out.resp.residual[0];
        s4.resp.residual[1] = l0_out.resp.residual[1];

        // Pack Lane 1 into bytes 2..3
        s4.resp.result[2]   = l1_out.resp.result[0];
        s4.resp.result[3]   = l1_out.resp.result[1];
        s4.resp.residual[2] = l1_out.resp.residual[0];
        s4.resp.residual[3] = l1_out.resp.residual[1];

        // Aggregate status flags across both lanes
        s4.resp.status = l0_out.resp.status | l1_out.resp.status;
    }

    return s4;
}


/**
 * @brief 2x FP16 Packed SIMD IP Kernel (32-bit Container)
 */
template <typename InPipe, typename OutPipe>
struct SimdFp16x2IpKernel {
    void operator()() const {
        SimdFp16x2Stage1Reg r1{};
        SimdFp16x2Stage2Reg r2{};
        SimdFp16x2Stage3Reg r3{};
        SimdFp16x2Stage4Reg r4{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            if (r4.valid) {
                OutPipe::write(r4.resp);
                if (r4.resp.eof) {
                    active = false;
                }
            }

            r4 = simd_fp16x2_stage4(r3);
            r3 = simd_fp16x2_stage3(r2);
            r2 = simd_fp16x2_stage2(r1);

            bool has_data = false;
            FpRequest<4> req = InPipe::read(has_data);
            r1 = simd_fp16x2_stage1(req, has_data);
        }
    }
};

// Pipe types and kernel wrappers
class SimdFp16x2InPipeId;
class SimdFp16x2OutPipeId;

using SimdFp16x2InPipe  = FpuPipe<SimdFp16x2InPipeId,  FpRequest<4>,  16>;
using SimdFp16x2OutPipe = FpuPipe<SimdFp16x2OutPipeId, FpResponse<4>, 16>;

class SimdFp16x2ProducerTask;
class SimdFp16x2IpTask;
class SimdFp16x2ConsumerTask;

template <typename InAccessor>
struct SimdFp16x2ProducerKernel : FpProducerKernel<SimdFp16x2InPipe, InAccessor, 4> {
    using FpProducerKernel<SimdFp16x2InPipe, InAccessor, 4>::FpProducerKernel;
};
template <typename InAccessor>
SimdFp16x2ProducerKernel(InAccessor, uint32_t) -> SimdFp16x2ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct SimdFp16x2ConsumerKernel : FpConsumerKernel<SimdFp16x2OutPipe, OutAccessor, 4> {
    using FpConsumerKernel<SimdFp16x2OutPipe, OutAccessor, 4>::FpConsumerKernel;
};
template <typename OutAccessor>
SimdFp16x2ConsumerKernel(OutAccessor) -> SimdFp16x2ConsumerKernel<OutAccessor>;

} // namespace vph
