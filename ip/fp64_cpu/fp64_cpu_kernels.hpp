#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "pipe_interface.hpp"
#include "common_kernels.hpp"
#include "fp64_cpu_ip.hpp"

namespace vph {

class Fp64InPipeId;
class Fp64OutPipeId;

using Fp64InPipe  = FpuPipe<Fp64InPipeId,  FpRequest<8>,  16>;
using Fp64OutPipe = FpuPipe<Fp64OutPipeId, FpResponse<8>, 16>;

class Fp64ProducerTask;
class Fp64IpTask;
class Fp64ConsumerTask;

template <typename InAccessor>
struct Fp64ProducerKernel : FpProducerKernel<Fp64InPipe, InAccessor, 8> {
    using FpProducerKernel<Fp64InPipe, InAccessor, 8>::FpProducerKernel;
};
template <typename InAccessor>
Fp64ProducerKernel(InAccessor, uint32_t) -> Fp64ProducerKernel<InAccessor>;

template <typename OutAccessor>
struct Fp64ConsumerKernel : FpConsumerKernel<Fp64OutPipe, OutAccessor, 8> {
    using FpConsumerKernel<Fp64OutPipe, OutAccessor, 8>::FpConsumerKernel;
};
template <typename OutAccessor>
Fp64ConsumerKernel(OutAccessor) -> Fp64ConsumerKernel<OutAccessor>;

} // namespace vph
