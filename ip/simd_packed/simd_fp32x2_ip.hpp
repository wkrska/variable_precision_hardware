#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp32_ieee_core.hpp"
#include "fp64_ieee_core.hpp"
#include "pipeline_regs.hpp"
#include "ip/fp32_cpu/fp32_cpu_ip.hpp"
#include "ip/fp64_cpu/fp64_cpu_ip.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"

namespace vph {

/**
 * @brief Explicit Pipeline Stages for Dual-Mode 64-bit SIMD/Scalar IP
 * 
 * In Packed SIMD mode: Executes 2 parallel FP32 lanes (Lane 0: bytes 0..3, Lane 1: bytes 4..7).
 * In Full Precision mode: Executes 1 scalar FP64 operation across the full 64-bit payload (bytes 0..7).
 * Mode is dynamically decoded per-cycle via is_scalar_mode(opcode, rounding_mode).
 */
struct SimdFp32x2Stage1Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage1Reg   lane0;
    Stage1Reg   lane1;
    Stage1Reg64 scalar_s1;
};

struct SimdFp32x2Stage2Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage2Reg   lane0;
    Stage2Reg   lane1;
    Stage2Reg64 scalar_s2;
};

struct SimdFp32x2Stage3Reg {
    bool        valid;
    bool        eof;
    uint16_t    tag;
    uint8_t     opcode;
    bool        is_scalar;
    Stage3Reg   lane0;
    Stage3Reg   lane1;
    Stage3Reg64 scalar_s3;
};

struct SimdFp32x2Stage4Reg {
    bool          valid;
    FpResponse<8> resp;
};

inline SimdFp32x2Stage1Reg simd_fp32x2_stage1(const FpRequest<8>& req, bool valid) {
    SimdFp32x2Stage1Reg s1{};
    s1.valid     = valid;
    s1.eof       = req.eof;
    s1.tag       = req.tag;
    s1.opcode    = req.opcode;
    s1.is_scalar = is_scalar_mode(req.opcode, req.rounding_mode);

    if (!valid) return s1;

    if (s1.is_scalar) {
        // Full precision scalar mode: Pass 8-byte container to FP64 datapath
        FpRequest<8> req_scalar = req;
        req_scalar.opcode = get_base_opcode(req.opcode);
        s1.scalar_s1 = fp64_stage1(req_scalar, true);
    } else {
        // Packed SIMD mode: Split into Lane 0 (bytes 0..3) and Lane 1 (bytes 4..7)
        FpRequest<4> req_l0{};
        req_l0.eof           = req.eof;
        req_l0.tag           = req.tag;
        req_l0.opcode        = get_base_opcode(req.opcode);
        req_l0.rounding_mode = req.rounding_mode;
        req_l0.op_a[0]       = req.op_a[0];
        req_l0.op_a[1]       = req.op_a[1];
        req_l0.op_a[2]       = req.op_a[2];
        req_l0.op_a[3]       = req.op_a[3];
        req_l0.op_b[0]       = req.op_b[0];
        req_l0.op_b[1]       = req.op_b[1];
        req_l0.op_b[2]       = req.op_b[2];
        req_l0.op_b[3]       = req.op_b[3];

        FpRequest<4> req_l1{};
        req_l1.eof           = req.eof;
        req_l1.tag           = req.tag;
        req_l1.opcode        = get_base_opcode(req.opcode);
        req_l1.rounding_mode = req.rounding_mode;
        req_l1.op_a[0]       = req.op_a[4];
        req_l1.op_a[1]       = req.op_a[5];
        req_l1.op_a[2]       = req.op_a[6];
        req_l1.op_a[3]       = req.op_a[7];
        req_l1.op_b[0]       = req.op_b[4];
        req_l1.op_b[1]       = req.op_b[5];
        req_l1.op_b[2]       = req.op_b[6];
        req_l1.op_b[3]       = req.op_b[7];

        s1.lane0 = fp32_stage1(req_l0, true);
        s1.lane1 = fp32_stage1(req_l1, true);
    }

    return s1;
}

inline SimdFp32x2Stage2Reg simd_fp32x2_stage2(const SimdFp32x2Stage1Reg& s1) {
    SimdFp32x2Stage2Reg s2{};
    s2.valid     = s1.valid;
    s2.eof       = s1.eof;
    s2.tag       = s1.tag;
    s2.opcode    = s1.opcode;
    s2.is_scalar = s1.is_scalar;

    if (!s1.valid) return s2;

    if (s1.is_scalar) {
        s2.scalar_s2 = fp64_stage2(s1.scalar_s1);
    } else {
        s2.lane0 = fp32_stage2(s1.lane0);
        s2.lane1 = fp32_stage2(s1.lane1);
    }

    return s2;
}

inline SimdFp32x2Stage3Reg simd_fp32x2_stage3(const SimdFp32x2Stage2Reg& s2) {
    SimdFp32x2Stage3Reg s3{};
    s3.valid     = s2.valid;
    s3.eof       = s2.eof;
    s3.tag       = s2.tag;
    s3.opcode    = s2.opcode;
    s3.is_scalar = s2.is_scalar;

    if (!s2.valid) return s3;

    if (s2.is_scalar) {
        s3.scalar_s3 = fp64_stage3(s2.scalar_s2);
    } else {
        s3.lane0 = fp32_stage3(s2.lane0);
        s3.lane1 = fp32_stage3(s2.lane1);
    }

    return s3;
}

inline SimdFp32x2Stage4Reg simd_fp32x2_stage4(const SimdFp32x2Stage3Reg& s3) {
    SimdFp32x2Stage4Reg s4{};
    s4.valid       = s3.valid;
    s4.resp.eof    = s3.eof;
    s4.resp.tag    = s3.tag;
    s4.resp.opcode = s3.opcode;

    if (!s3.valid) return s4;

    if (s3.is_scalar) {
        Stage4Reg64 sc_out = fp64_stage4(s3.scalar_s3);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            s4.resp.result[i]   = sc_out.resp.result[i];
            s4.resp.residual[i] = sc_out.resp.residual[i];
        }
        s4.resp.status = sc_out.resp.status;
    } else {
        Stage4Reg l0_out = fp32_stage4(s3.lane0);
        Stage4Reg l1_out = fp32_stage4(s3.lane1);

        // Pack Lane 0 into bytes 0..3
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            s4.resp.result[i]   = l0_out.resp.result[i];
            s4.resp.residual[i] = l0_out.resp.residual[i];
        }

        // Pack Lane 1 into bytes 4..7
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            s4.resp.result[i + 4]   = l1_out.resp.result[i];
            s4.resp.residual[i + 4] = l1_out.resp.residual[i];
        }

        s4.resp.status = l0_out.resp.status | l1_out.resp.status;
    }

    return s4;
}


/**
 * @brief 2x FP32 Packed SIMD IP Kernel (64-bit Container)
 */
template <typename InPipe, typename OutPipe>
struct SimdFp32x2IpKernel {
    void operator()() const {
        SimdFp32x2Stage1Reg r1{};
        SimdFp32x2Stage2Reg r2{};
        SimdFp32x2Stage3Reg r3{};
        SimdFp32x2Stage4Reg r4{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            if (r4.valid) {
                OutPipe::write(r4.resp);
                if (r4.resp.eof) {
                    active = false;
                }
            }

            r4 = simd_fp32x2_stage4(r3);
            r3 = simd_fp32x2_stage3(r2);
            r2 = simd_fp32x2_stage2(r1);

            bool has_data = false;
            FpRequest<8> req = InPipe::read(has_data);
            r1 = simd_fp32x2_stage1(req, has_data);
        }
    }
};

// Pipe types and kernel wrappers
class SimdFp32x2InPipeId;
class SimdFp32x2OutPipeId;

using SimdFp32x2InPipe  = FpuPipe<SimdFp32x2InPipeId,  FpRequest<8>,  16>;
using SimdFp32x2OutPipe = FpuPipe<SimdFp32x2OutPipeId, FpResponse<8>, 16>;

class SimdFp32x2ProducerTask;
class SimdFp32x2IpTask;
class SimdFp32x2ConsumerTask;

template <typename InAccessor>
struct SimdFp32x2ProducerKernel : FpProducerKernel<SimdFp32x2InPipe, InAccessor, 8> {
    using FpProducerKernel<SimdFp32x2InPipe, InAccessor, 8>::FpProducerKernel;
};
template <typename InAccessor>
SimdFp32x2ProducerKernel(InAccessor, uint32_t) -> SimdFp32x2ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct SimdFp32x2ConsumerKernel : FpConsumerKernel<SimdFp32x2OutPipe, OutAccessor, 8> {
    using FpConsumerKernel<SimdFp32x2OutPipe, OutAccessor, 8>::FpConsumerKernel;
};
template <typename OutAccessor>
SimdFp32x2ConsumerKernel(OutAccessor) -> SimdFp32x2ConsumerKernel<OutAccessor>;

} // namespace vph
