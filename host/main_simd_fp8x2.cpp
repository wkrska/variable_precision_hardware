#include <iostream>
#include <vector>
#include <chrono>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "pipe_interface.hpp"
#include "ip/simd_packed/simd_fp8x2_ip.hpp"
#include "host/host_verification.hpp"

using namespace sycl;
using namespace vph;

int main() {
    try {
        #if defined(FPGA_EMULATOR)
            ext::intel::fpga_emulator_selector selector;
        #elif defined(FPGA_SIMULATOR)
            ext::intel::fpga_simulator_selector selector;
        #else
            ext::intel::fpga_selector selector;
        #endif

        queue q(selector, [](exception_list el) {
            for (auto& ep : el) {
                try {
                    std::rethrow_exception(ep);
                } catch (sycl::exception const& e) {
                    std::cerr << "SYCL Exception: " << e.what() << std::endl;
                }
            }
        });

        std::cout << "[INFO] Running 2x FP8 Packed SIMD IP on device: "
                  << q.get_device().get_info<info::device::name>() << std::endl;

        std::vector<FpRequest<2>> test_requests = HostVerification::generate_simd_fp8x2_test_vectors();
        uint32_t num_items = static_cast<uint32_t>(test_requests.size());
        std::vector<FpResponse<2>> test_responses(num_items);

        std::cout << "[INFO] Enqueuing " << num_items << " test vectors to 2x FP8 Packed SIMD pipeline..." << std::endl;

        {
            buffer<FpRequest<2>, 1> in_buf(test_requests.data(), range<1>(num_items));
            buffer<FpResponse<2>, 1> out_buf(test_responses.data(), range<1>(num_items));

            auto start_time = std::chrono::high_resolution_clock::now();

            q.submit([&](handler& h) {
                accessor out_acc(out_buf, h, write_only, no_init);
                h.single_task<SimdFp8x2ConsumerTask>(SimdFp8x2ConsumerKernel(out_acc));
            });

            q.submit([&](handler& h) {
                h.single_task<SimdFp8x2IpTask>(
                    SimdFp8x2IpKernel<SimdFp8x2InPipe, SimdFp8x2OutPipe>()
                );
            });

            q.submit([&](handler& h) {
                accessor in_acc(in_buf, h, read_only);
                h.single_task<SimdFp8x2ProducerTask>(SimdFp8x2ProducerKernel(in_acc, num_items));
            });

            q.wait();

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end_time - start_time;
            std::cout << "[INFO] Execution completed in " << duration.count() << " ms." << std::endl;
        }

        bool success = HostVerification::verify_simd_fp8x2_results(test_requests, test_responses);
        if (!success) {
            std::cerr << "[FAILED] 2x FP8 Packed SIMD Verification did not pass." << std::endl;
            return 1;
        }

        std::cout << "[SUCCESS] 2x FP8 Packed SIMD IP fully validated!" << std::endl;
        return 0;

    } catch (sycl::exception const& e) {
        std::cerr << "[ERROR] Caught SYCL exception: " << e.what() << std::endl;
        return 2;
    }
}
