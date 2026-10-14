#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"
#include "fp32_cpu_ip.hpp"

namespace vph {

// Pipe Identifiers for FP32 CPU IP
class Fp32InPipeId;
class Fp32OutPipeId;

// Concrete Pipe Types
using Fp32InPipe  = FpuPipe<Fp32InPipeId,  FpRequest<4>,  16>;
using Fp32OutPipe = FpuPipe<Fp32OutPipeId, FpResponse<4>, 16>;

// Kernel Task Type Identifiers
class Fp32ProducerTask;
class Fp32IpTask;
class Fp32ConsumerTask;

// Generic Producer & Consumer specialized for FP32 with CTAD guides
template <typename InAccessor>
struct Fp32ProducerKernel : FpProducerKernel<Fp32InPipe, InAccessor, 4> {
    using FpProducerKernel<Fp32InPipe, InAccessor, 4>::FpProducerKernel;
};
template <typename InAccessor>
Fp32ProducerKernel(InAccessor, uint32_t) -> Fp32ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct Fp32ConsumerKernel : FpConsumerKernel<Fp32OutPipe, OutAccessor, 4> {
    using FpConsumerKernel<Fp32OutPipe, OutAccessor, 4>::FpConsumerKernel;
};
template <typename OutAccessor>
Fp32ConsumerKernel(OutAccessor) -> Fp32ConsumerKernel<OutAccessor>;

} // namespace vph
