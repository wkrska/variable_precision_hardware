#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "fpu_types.hpp"

namespace vph {

/**
 * @brief FPGA SYCL Pipe Wrapper
 * 
 * Maps to Intel FPGA FIFO primitives with specified minimum depth.
 */
template <class Id, typename T, size_t Depth = 16>
using FpuPipe = sycl::ext::intel::pipe<Id, T, Depth>;

} // namespace vph
