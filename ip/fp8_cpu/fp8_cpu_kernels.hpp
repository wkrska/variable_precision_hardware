#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"
#include "fp8_cpu_ip.hpp"

namespace vph {

class Fp8InPipeId;
class Fp8OutPipeId;

using Fp8InPipe  = FpuPipe<Fp8InPipeId,  FpRequest<1>,  16>;
using Fp8OutPipe = FpuPipe<Fp8OutPipeId, FpResponse<1>, 16>;

class Fp8ProducerTask;
class Fp8IpTask;
class Fp8ConsumerTask;

template <typename InAccessor>
struct Fp8ProducerKernel : FpProducerKernel<Fp8InPipe, InAccessor, 1> {
    using FpProducerKernel<Fp8InPipe, InAccessor, 1>::FpProducerKernel;
};
template <typename InAccessor>
Fp8ProducerKernel(InAccessor, uint32_t) -> Fp8ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct Fp8ConsumerKernel : FpConsumerKernel<Fp8OutPipe, OutAccessor, 1> {
    using FpConsumerKernel<Fp8OutPipe, OutAccessor, 1>::FpConsumerKernel;
};
template <typename OutAccessor>
Fp8ConsumerKernel(OutAccessor) -> Fp8ConsumerKernel<OutAccessor>;

} // namespace vph
