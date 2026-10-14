#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"
#include "fp16_cpu_ip.hpp"

namespace vph {

class Fp16InPipeId;
class Fp16OutPipeId;

using Fp16InPipe  = FpuPipe<Fp16InPipeId,  FpRequest<2>,  16>;
using Fp16OutPipe = FpuPipe<Fp16OutPipeId, FpResponse<2>, 16>;

class Fp16ProducerTask;
class Fp16IpTask;
class Fp16ConsumerTask;

template <typename InAccessor>
struct Fp16ProducerKernel : FpProducerKernel<Fp16InPipe, InAccessor, 2> {
    using FpProducerKernel<Fp16InPipe, InAccessor, 2>::FpProducerKernel;
};
template <typename InAccessor>
Fp16ProducerKernel(InAccessor, uint32_t) -> Fp16ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct Fp16ConsumerKernel : FpConsumerKernel<Fp16OutPipe, OutAccessor, 2> {
    using FpConsumerKernel<Fp16OutPipe, OutAccessor, 2>::FpConsumerKernel;
};
template <typename OutAccessor>
Fp16ConsumerKernel(OutAccessor) -> Fp16ConsumerKernel<OutAccessor>;

} // namespace vph
