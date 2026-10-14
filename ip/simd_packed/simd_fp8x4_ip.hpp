#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp8_core.hpp"
#include "fp32_ieee_core.hpp"
#include "pipeline_regs.hpp"
#include "ip/fp8_cpu/fp8_cpu_ip.hpp"
#include "ip/fp32_cpu/fp32_cpu_ip.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"

namespace vph {

/**
 * @brief Explicit Pipeline Stages for Dual-Mode 32-bit SIMD/Scalar IP (4x FP8 / 1x FP32)
 * 
 * In Packed SIMD mode: Executes 4 parallel FP8 lanes (Bytes 0, 1, 2, 3).
 * In Full Precision mode: Executes 1 scalar FP32 operation across the full 32-bit payload (Bytes 0..3).
 * Mode is dynamically decoded per-cycle via is_scalar_mode(opcode, rounding_mode).
 */
struct SimdFp8x4Stage1Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage1Reg8  lane0;
    Stage1Reg8  lane1;
    Stage1Reg8  lane2;
    Stage1Reg8  lane3;
    Stage1Reg   scalar_s1;
};

struct SimdFp8x4Stage2Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage2Reg8  lane0;
    Stage2Reg8  lane1;
    Stage2Reg8  lane2;
    Stage2Reg8  lane3;
    Stage2Reg   scalar_s2;
};

struct SimdFp8x4Stage3Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage2Reg8  lane0;
    Stage2Reg8  lane1;
    Stage2Reg8  lane2;
    Stage2Reg8  lane3;
    Stage3Reg   scalar_s3;
};

struct SimdFp8x4Stage4Reg {
    bool          valid;
    FpResponse<4> resp;
};

inline SimdFp8x4Stage1Reg simd_fp8x4_stage1(const FpRequest<4>& req, bool valid) {
    SimdFp8x4Stage1Reg s1{};
    s1.valid     = valid;
    s1.eof       = req.eof;
    s1.tag       = req.tag;
    s1.opcode    = req.opcode;
    s1.is_scalar = is_scalar_mode(req.opcode, req.rounding_mode);

    if (!valid) return s1;

    if (s1.is_scalar) {
        FpRequest<4> req_scalar = req;
        req_scalar.opcode = get_base_opcode(req.opcode);
        s1.scalar_s1 = fp32_stage1(req_scalar, true);
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

        FpRequest<1> req_l2{};
        req_l2.eof           = req.eof;
        req_l2.tag           = req.tag;
        req_l2.opcode        = get_base_opcode(req.opcode);
        req_l2.rounding_mode = req.rounding_mode;
        req_l2.op_a[0]       = req.op_a[2];
        req_l2.op_b[0]       = req.op_b[2];

        FpRequest<1> req_l3{};
        req_l3.eof           = req.eof;
        req_l3.tag           = req.tag;
        req_l3.opcode        = get_base_opcode(req.opcode);
        req_l3.rounding_mode = req.rounding_mode;
        req_l3.op_a[0]       = req.op_a[3];
        req_l3.op_b[0]       = req.op_b[3];

        s1.lane0 = fp8_stage1(req_l0, true);
        s1.lane1 = fp8_stage1(req_l1, true);
        s1.lane2 = fp8_stage1(req_l2, true);
        s1.lane3 = fp8_stage1(req_l3, true);
    }

    return s1;
}

inline SimdFp8x4Stage2Reg simd_fp8x4_stage2(const SimdFp8x4Stage1Reg& s1) {
    SimdFp8x4Stage2Reg s2{};
    s2.valid     = s1.valid;
    s2.eof       = s1.eof;
    s2.tag       = s1.tag;
    s2.opcode    = s1.opcode;
    s2.is_scalar = s1.is_scalar;

    if (!s1.valid) return s2;

    if (s1.is_scalar) {
        s2.scalar_s2 = fp32_stage2(s1.scalar_s1);
    } else {
        s2.lane0 = fp8_stage2(s1.lane0);
        s2.lane1 = fp8_stage2(s1.lane1);
        s2.lane2 = fp8_stage2(s1.lane2);
        s2.lane3 = fp8_stage2(s1.lane3);
    }

    return s2;
}

inline SimdFp8x4Stage3Reg simd_fp8x4_stage3(const SimdFp8x4Stage2Reg& s2) {
    SimdFp8x4Stage3Reg s3{};
    s3.valid     = s2.valid;
    s3.eof       = s2.eof;
    s3.tag       = s2.tag;
    s3.opcode    = s2.opcode;
    s3.is_scalar = s2.is_scalar;

    if (!s2.valid) return s3;

    if (s2.is_scalar) {
        s3.scalar_s3 = fp32_stage3(s2.scalar_s2);
    } else {
        s3.lane0 = s2.lane0;
        s3.lane1 = s2.lane1;
        s3.lane2 = s2.lane2;
        s3.lane3 = s2.lane3;
    }

    return s3;
}

inline SimdFp8x4Stage4Reg simd_fp8x4_stage4(const SimdFp8x4Stage3Reg& s3) {
    SimdFp8x4Stage4Reg s4{};
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
        Stage3Reg8 l0_out = fp8_stage3(s3.lane0);
        Stage3Reg8 l1_out = fp8_stage3(s3.lane1);
        Stage3Reg8 l2_out = fp8_stage3(s3.lane2);
        Stage3Reg8 l3_out = fp8_stage3(s3.lane3);

        s4.resp.result[0]   = l0_out.resp.result[0];
        s4.resp.residual[0] = l0_out.resp.residual[0];

        s4.resp.result[1]   = l1_out.resp.result[0];
        s4.resp.residual[1] = l1_out.resp.residual[0];

        s4.resp.result[2]   = l2_out.resp.result[0];
        s4.resp.residual[2] = l2_out.resp.residual[0];

        s4.resp.result[3]   = l3_out.resp.result[0];
        s4.resp.residual[3] = l3_out.resp.residual[0];

        s4.resp.status = l0_out.resp.status | l1_out.resp.status |
                         l2_out.resp.status | l3_out.resp.status;
    }

    return s4;
}

/**
 * @brief 4x FP8 Packed SIMD IP Kernel (32-bit Container)
 */
template <typename InPipe, typename OutPipe>
struct SimdFp8x4IpKernel {
    void operator()() const {
        SimdFp8x4Stage1Reg r1{};
        SimdFp8x4Stage2Reg r2{};
        SimdFp8x4Stage3Reg r3{};
        SimdFp8x4Stage4Reg r4{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            if (r4.valid) {
                OutPipe::write(r4.resp);
                if (r4.resp.eof) {
                    active = false;
                }
            }

            r4 = simd_fp8x4_stage4(r3);
            r3 = simd_fp8x4_stage3(r2);
            r2 = simd_fp8x4_stage2(r1);

            bool has_data = false;
            FpRequest<4> req = InPipe::read(has_data);
            r1 = simd_fp8x4_stage1(req, has_data);
        }
    }
};


// Pipe types and kernel wrappers
class SimdFp8x4InPipeId;
class SimdFp8x4OutPipeId;

using SimdFp8x4InPipe  = FpuPipe<SimdFp8x4InPipeId,  FpRequest<4>,  16>;
using SimdFp8x4OutPipe = FpuPipe<SimdFp8x4OutPipeId, FpResponse<4>, 16>;

class SimdFp8x4ProducerTask;
class SimdFp8x4IpTask;
class SimdFp8x4ConsumerTask;

template <typename InAccessor>
struct SimdFp8x4ProducerKernel : FpProducerKernel<SimdFp8x4InPipe, InAccessor, 4> {
    using FpProducerKernel<SimdFp8x4InPipe, InAccessor, 4>::FpProducerKernel;
};
template <typename InAccessor>
SimdFp8x4ProducerKernel(InAccessor, uint32_t) -> SimdFp8x4ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct SimdFp8x4ConsumerKernel : FpConsumerKernel<SimdFp8x4OutPipe, OutAccessor, 4> {
    using FpConsumerKernel<SimdFp8x4OutPipe, OutAccessor, 4>::FpConsumerKernel;
};
template <typename OutAccessor>
SimdFp8x4ConsumerKernel(OutAccessor) -> SimdFp8x4ConsumerKernel<OutAccessor>;

} // namespace vph
