#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fpx2_core.hpp"
#include "fp64_ieee_core.hpp" // for bytes_to_u64 and u64_to_bytes
#include "pipe_interface.hpp"
#include "common_kernels.hpp"

namespace vph {

struct Stage1RegFpx2 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint64_t raw_a;
    uint64_t raw_b;
};

struct Stage2RegFpx2 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint64_t result;
    uint64_t residual;
    uint8_t  status;
};

struct Stage3RegFpx2 {
    bool          valid;
    FpResponse<8> resp;
};

inline Stage1RegFpx2 fpx2_stage1(const FpRequest<8>& req, bool valid) {
    Stage1RegFpx2 s1{};
    s1.valid = valid;
    s1.eof   = req.eof;
    s1.tag   = req.tag;
    s1.opcode = req.opcode;
    s1.raw_a = bytes_to_u64(req.op_a);
    s1.raw_b = bytes_to_u64(req.op_b);
    return s1;
}

inline Stage2RegFpx2 fpx2_stage2(const Stage1RegFpx2& s1) {
    Stage2RegFpx2 s2{};
    s2.valid = s1.valid;
    s2.eof   = s1.eof;
    s2.tag   = s1.tag;
    s2.opcode = s1.opcode;

    if (!s1.valid) return s2;

    Fpx2AluResult res = fpx2_eval_alu(s1.raw_a, s1.raw_b, s1.opcode);
    s2.result   = res.result;
    s2.residual = res.residual;
    s2.status   = res.status;
    return s2;
}

inline Stage3RegFpx2 fpx2_stage3(const Stage2RegFpx2& s2) {
    Stage3RegFpx2 s3{};
    s3.valid = s2.valid;
    s3.resp.eof    = s2.eof;
    s3.resp.tag    = s2.tag;
    s3.resp.opcode = s2.opcode;
    s3.resp.status = s2.status;
    u64_to_bytes(s2.result,   s3.resp.result);
    u64_to_bytes(s2.residual, s3.resp.residual);
    return s3;
}

/**
 * @brief Custom FPX2 Multi-Limb Shared-Exponent IP Kernel
 * 
 * Replays 24x24 FP32 significand multipliers for high-precision limb arithmetic.
 */
template <typename InPipe, typename OutPipe>
struct Fpx2CustomIpKernel {
    void operator()() const {
        Stage1RegFpx2 r1{};
        Stage2RegFpx2 r2{};
        Stage3RegFpx2 r3{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            if (r3.valid) {
                OutPipe::write(r3.resp);
                if (r3.resp.eof) {
                    active = false;
                }
            }

            r3 = fpx2_stage3(r2);
            r2 = fpx2_stage2(r1);

            bool has_data = false;
            FpRequest<8> req = InPipe::read(has_data);
            r1 = fpx2_stage1(req, has_data);
        }
    }
};

// Pipe types and kernel wrappers
class Fpx2InPipeId;
class Fpx2OutPipeId;

using Fpx2InPipe  = FpuPipe<Fpx2InPipeId,  FpRequest<8>,  16>;
using Fpx2OutPipe = FpuPipe<Fpx2OutPipeId, FpResponse<8>, 16>;

class Fpx2ProducerTask;
class Fpx2IpTask;
class Fpx2ConsumerTask;

template <typename InAccessor>
struct Fpx2ProducerKernel : FpProducerKernel<Fpx2InPipe, InAccessor, 8> {
    using FpProducerKernel<Fpx2InPipe, InAccessor, 8>::FpProducerKernel;
};
template <typename InAccessor>
Fpx2ProducerKernel(InAccessor, uint32_t) -> Fpx2ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct Fpx2ConsumerKernel : FpConsumerKernel<Fpx2OutPipe, OutAccessor, 8> {
    using FpConsumerKernel<Fpx2OutPipe, OutAccessor, 8>::FpConsumerKernel;
};
template <typename OutAccessor>
Fpx2ConsumerKernel(OutAccessor) -> Fpx2ConsumerKernel<OutAccessor>;

} // namespace vph
