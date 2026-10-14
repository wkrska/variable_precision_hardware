#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp32_ieee_core.hpp"
#include "pipeline_regs.hpp"
#include "ip/fp32_cpu/fp32_cpu_ip.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"

namespace vph {

/**
 * @brief 4-Lane Streaming Vector Accelerator Pipeline Stages (128-bit container)
 * 
 * Specifically optimized for accelerator streaming architectures with deep elastic FIFO
 * buffering and multi-lane initiation interval II=1.
 */
struct StreamingAccelStage1Reg {
    bool      valid;
    bool      eof;
    uint16_t  tag;
    uint8_t   opcode;
    Stage1Reg lane0;
    Stage1Reg lane1;
    Stage1Reg lane2;
    Stage1Reg lane3;
};

struct StreamingAccelStage2Reg {
    bool      valid;
    bool      eof;
    uint16_t  tag;
    uint8_t   opcode;
    Stage2Reg lane0;
    Stage2Reg lane1;
    Stage2Reg lane2;
    Stage2Reg lane3;
};

struct StreamingAccelStage3Reg {
    bool      valid;
    bool      eof;
    uint16_t  tag;
    uint8_t   opcode;
    Stage3Reg lane0;
    Stage3Reg lane1;
    Stage3Reg lane2;
    Stage3Reg lane3;
};

struct StreamingAccelStage4Reg {
    bool           valid;
    FpResponse<16> resp;
};

inline StreamingAccelStage1Reg streaming_accel_stage1(const FpRequest<16>& req, bool valid) {
    StreamingAccelStage1Reg s1{};
    s1.valid = valid;
    s1.eof   = req.eof;
    s1.tag   = req.tag;
    s1.opcode = req.opcode;

    if (!valid) return s1;

    // Helper lambda to extract 4-byte slice
    auto make_sub_req = [&](size_t offset) {
        FpRequest<4> r{};
        r.eof           = req.eof;
        r.tag           = req.tag;
        r.opcode        = req.opcode;
        r.rounding_mode = req.rounding_mode;
        r.op_a[0] = req.op_a[offset + 0];
        r.op_a[1] = req.op_a[offset + 1];
        r.op_a[2] = req.op_a[offset + 2];
        r.op_a[3] = req.op_a[offset + 3];
        r.op_b[0] = req.op_b[offset + 0];
        r.op_b[1] = req.op_b[offset + 1];
        r.op_b[2] = req.op_b[offset + 2];
        r.op_b[3] = req.op_b[offset + 3];
        return r;
    };

    s1.lane0 = fp32_stage1(make_sub_req(0),  true);
    s1.lane1 = fp32_stage1(make_sub_req(4),  true);
    s1.lane2 = fp32_stage1(make_sub_req(8),  true);
    s1.lane3 = fp32_stage1(make_sub_req(12), true);

    return s1;
}

inline StreamingAccelStage2Reg streaming_accel_stage2(const StreamingAccelStage1Reg& s1) {
    StreamingAccelStage2Reg s2{};
    s2.valid = s1.valid;
    s2.eof   = s1.eof;
    s2.tag   = s1.tag;
    s2.opcode = s1.opcode;

    if (!s1.valid) return s2;

    s2.lane0 = fp32_stage2(s1.lane0);
    s2.lane1 = fp32_stage2(s1.lane1);
    s2.lane2 = fp32_stage2(s1.lane2);
    s2.lane3 = fp32_stage2(s1.lane3);

    return s2;
}

inline StreamingAccelStage3Reg streaming_accel_stage3(const StreamingAccelStage2Reg& s2) {
    StreamingAccelStage3Reg s3{};
    s3.valid = s2.valid;
    s3.eof   = s2.eof;
    s3.tag   = s2.tag;
    s3.opcode = s2.opcode;

    if (!s2.valid) return s3;

    s3.lane0 = fp32_stage3(s2.lane0);
    s3.lane1 = fp32_stage3(s2.lane1);
    s3.lane2 = fp32_stage3(s2.lane2);
    s3.lane3 = fp32_stage3(s2.lane3);

    return s3;
}

inline StreamingAccelStage4Reg streaming_accel_stage4(const StreamingAccelStage3Reg& s3) {
    StreamingAccelStage4Reg s4{};
    s4.valid = s3.valid;
    s4.resp.eof    = s3.eof;
    s4.resp.tag    = s3.tag;
    s4.resp.opcode = s3.opcode;

    if (!s3.valid) return s4;

    Stage4Reg l0_out = fp32_stage4(s3.lane0);
    Stage4Reg l1_out = fp32_stage4(s3.lane1);
    Stage4Reg l2_out = fp32_stage4(s3.lane2);
    Stage4Reg l3_out = fp32_stage4(s3.lane3);

    auto copy_lane = [&](size_t offset, const Stage4Reg& l) {
        s4.resp.result[offset + 0]   = l.resp.result[0];
        s4.resp.result[offset + 1]   = l.resp.result[1];
        s4.resp.result[offset + 2]   = l.resp.result[2];
        s4.resp.result[offset + 3]   = l.resp.result[3];
        s4.resp.residual[offset + 0] = l.resp.residual[0];
        s4.resp.residual[offset + 1] = l.resp.residual[1];
        s4.resp.residual[offset + 2] = l.resp.residual[2];
        s4.resp.residual[offset + 3] = l.resp.residual[3];
    };

    copy_lane(0,  l0_out);
    copy_lane(4,  l1_out);
    copy_lane(8,  l2_out);
    copy_lane(12, l3_out);

    s4.resp.status = l0_out.resp.status | l1_out.resp.status |
                     l2_out.resp.status | l3_out.resp.status;

    return s4;
}

/**
 * @brief 4-Lane Streaming Vector Accelerator FPU Kernel (128-bit container)
 */
template <typename InPipe, typename OutPipe>
struct StreamingAccelIpKernel {
    void operator()() const {
        StreamingAccelStage1Reg r1{};
        StreamingAccelStage2Reg r2{};
        StreamingAccelStage3Reg r3{};
        StreamingAccelStage4Reg r4{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            if (r4.valid) {
                OutPipe::write(r4.resp);
                if (r4.resp.eof) {
                    active = false;
                }
            }

            r4 = streaming_accel_stage4(r3);
            r3 = streaming_accel_stage3(r2);
            r2 = streaming_accel_stage2(r1);

            bool has_data = false;
            FpRequest<16> req = InPipe::read(has_data);
            r1 = streaming_accel_stage1(req, has_data);
        }
    }
};

// Pipe types and kernel wrappers
class StreamingAccelInPipeId;
class StreamingAccelOutPipeId;

// Deep pipe FIFO buffers (depth 64) for streaming accelerator decouplers
using StreamingAccelInPipe  = FpuPipe<StreamingAccelInPipeId,  FpRequest<16>,  64>;
using StreamingAccelOutPipe = FpuPipe<StreamingAccelOutPipeId, FpResponse<16>, 64>;

class StreamingAccelProducerTask;
class StreamingAccelIpTask;
class StreamingAccelConsumerTask;

template <typename InAccessor>
struct StreamingAccelProducerKernel : FpProducerKernel<StreamingAccelInPipe, InAccessor, 16> {
    using FpProducerKernel<StreamingAccelInPipe, InAccessor, 16>::FpProducerKernel;
};
template <typename InAccessor>
StreamingAccelProducerKernel(InAccessor, uint32_t) -> StreamingAccelProducerKernel<InAccessor>;

template <typename OutAccessor>
struct StreamingAccelConsumerKernel : FpConsumerKernel<StreamingAccelOutPipe, OutAccessor, 16> {
    using FpConsumerKernel<StreamingAccelOutPipe, OutAccessor, 16>::FpConsumerKernel;
};
template <typename OutAccessor>
StreamingAccelConsumerKernel(OutAccessor) -> StreamingAccelConsumerKernel<OutAccessor>;

} // namespace vph
