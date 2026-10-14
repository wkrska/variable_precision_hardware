#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"

namespace vph {

/**
 * @brief Templatized Universal Producer Kernel
 * 
 * Streams an array of FpRequest<NBytes> from host/device memory into the specified InPipe
 * with Initiation Interval 1.
 * 
 * @tparam InPipe     The typed sycl::ext::intel::pipe to write into
 * @tparam InAccessor SYCL accessor type for reading requests
 * @tparam NBytes     Data byte width (e.g. 1 for FP8, 2 for FP16/BF16, 4 for FP32, 8 for FP64)
 */
template <typename InPipe, typename InAccessor, size_t NBytes>
struct FpProducerKernel {
    InAccessor in_acc;
    uint32_t   num_items;

    FpProducerKernel(InAccessor acc, uint32_t count)
        : in_acc(acc), num_items(count) {}

    void operator()() const {
        [[intel::initiation_interval(1)]]
        for (uint32_t i = 0; i < num_items; i++) {
            FpRequest<NBytes> req = in_acc[i];
            if (i == num_items - 1) {
                req.eof = true;
            }
            InPipe::write(req);
        }
    }
};

/**
 * @brief Templatized Universal Consumer Kernel
 * 
 * Reads FpResponse<NBytes> from the specified OutPipe and stores them sequentially
 * into the output buffer until EOF is encountered with Initiation Interval 1.
 * 
 * @tparam OutPipe     The typed sycl::ext::intel::pipe to read from
 * @tparam OutAccessor SYCL accessor type for writing responses
 * @tparam NBytes      Data byte width (e.g. 1 for FP8, 2 for FP16/BF16, 4 for FP32, 8 for FP64)
 */
template <typename OutPipe, typename OutAccessor, size_t NBytes>
struct FpConsumerKernel {
    OutAccessor out_acc;

    FpConsumerKernel(OutAccessor acc) : out_acc(acc) {}

    void operator()() const {
        bool active = true;
        uint32_t idx = 0;

        [[intel::initiation_interval(1)]]
        while (active) {
            bool has_data = false;
            FpResponse<NBytes> resp = OutPipe::read(has_data);
            if (has_data) {
                out_acc[idx++] = resp;
                if (resp.eof) {
                    active = false;
                }
            }
        }
    }
};

/**
 * @brief Factory helper functions to instantiate producer and consumer without explicit accessor types.
 */
template <typename InPipe, size_t NBytes, typename InAccessor>
inline auto make_producer(InAccessor acc, uint32_t count) {
    return FpProducerKernel<InPipe, InAccessor, NBytes>(acc, count);
}

template <typename OutPipe, size_t NBytes, typename OutAccessor>
inline auto make_consumer(OutAccessor acc) {
    return FpConsumerKernel<OutPipe, OutAccessor, NBytes>(acc);
}

} // namespace vph
