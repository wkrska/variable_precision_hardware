#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp8_core.hpp"

namespace vph {

/**
 * @brief Stage 1 Pipeline Register for FP8 CPU IP
 */
struct Stage1Reg8 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode; // 0 = E4M3, 1 = E5M2
    uint8_t  raw_a;
    uint8_t  raw_b;
};

/**
 * @brief Stage 2 Pipeline Register for FP8 CPU IP
 */
struct Stage2Reg8 {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  result;
    uint8_t  residual;
    uint8_t  status;
};

/**
 * @brief Stage 3 Pipeline Register for FP8 CPU IP
 */
struct Stage3Reg8 {
    bool          valid;
    FpResponse<1> resp;
};

inline Stage1Reg8 fp8_stage1(const FpRequest<1>& req, bool valid) {
    Stage1Reg8 s1{};
    s1.valid = valid;
    s1.eof   = req.eof;
    s1.tag   = req.tag;
    s1.opcode = req.opcode;
    s1.rounding_mode = req.rounding_mode;
    s1.raw_a = req.op_a[0];
    s1.raw_b = req.op_b[0];
    return s1;
}

inline Stage2Reg8 fp8_stage2(const Stage1Reg8& s1) {
    Stage2Reg8 s2{};
    s2.valid = s1.valid;
    s2.eof   = s1.eof;
    s2.tag   = s1.tag;
    s2.opcode = s1.opcode;

    if (!s1.valid) return s2;

    Fp8AluResult res;
    if (s1.rounding_mode == static_cast<uint8_t>(Fp8Type::E5M2)) {
        res = fp8_e5m2_eval_alu(s1.raw_a, s1.raw_b, s1.opcode);
    } else {
        res = fp8_e4m3_eval_alu(s1.raw_a, s1.raw_b, s1.opcode);
    }

    s2.result   = res.result;
    s2.residual = res.residual;
    s2.status   = res.status;
    return s2;
}

inline Stage3Reg8 fp8_stage3(const Stage2Reg8& s2) {
    Stage3Reg8 s3{};
    s3.valid = s2.valid;
    s3.resp.eof    = s2.eof;
    s3.resp.tag    = s2.tag;
    s3.resp.opcode = s2.opcode;
    s3.resp.status = s2.status;
    s3.resp.result[0]   = s2.result;
    s3.resp.residual[0] = s2.residual;
    return s3;
}

/**
 * @brief Bit-Level FP8 CPU Standalone IP Kernel
 * 
 * Executes FP8 operations (E4M3 and E5M2) with II=1 and explicit pipelining.
 */
template <typename InPipe, typename OutPipe>
struct Fp8CpuIpKernel {
    void operator()() const {
        Stage1Reg8 r1{};
        Stage2Reg8 r2{};
        Stage3Reg8 r3{};

        bool active = true;

        [[intel::initiation_interval(1)]]
        while (active) {
            if (r3.valid) {
                OutPipe::write(r3.resp);
                if (r3.resp.eof) {
                    active = false;
                }
            }

            r3 = fp8_stage3(r2);
            r2 = fp8_stage2(r1);

            bool has_data = false;
            FpRequest<1> req = InPipe::read(has_data);
            r1 = fp8_stage1(req, has_data);
        }
    }
};

} // namespace vph
