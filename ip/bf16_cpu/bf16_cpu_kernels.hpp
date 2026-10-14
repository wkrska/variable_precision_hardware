#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"
#include "bf16_cpu_ip.hpp"

namespace vph {

class Bf16InPipeId;
class Bf16OutPipeId;

using Bf16InPipe  = FpuPipe<Bf16InPipeId,  FpRequest<2>,  16>;
using Bf16OutPipe = FpuPipe<Bf16OutPipeId, FpResponse<2>, 16>;

class Bf16ProducerTask;
class Bf16IpTask;
class Bf16ConsumerTask;

template <typename InAccessor>
struct Bf16ProducerKernel : FpProducerKernel<Bf16InPipe, InAccessor, 2> {
    using FpProducerKernel<Bf16InPipe, InAccessor, 2>::FpProducerKernel;
};
template <typename InAccessor>
Bf16ProducerKernel(InAccessor, uint32_t) -> Bf16ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct Bf16ConsumerKernel : FpConsumerKernel<Bf16OutPipe, OutAccessor, 2> {
    using FpConsumerKernel<Bf16OutPipe, OutAccessor, 2>::FpConsumerKernel;
};
template <typename OutAccessor>
Bf16ConsumerKernel(OutAccessor) -> Bf16ConsumerKernel<OutAccessor>;

} // namespace vph
