#include <iostream>
#include <vector>
#include <chrono>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "fpu_types.hpp"
#include "pipe_interface.hpp"
#include "ip/fp32_cpu/fp32_cpu_ip.hpp"
#include "ip/fp32_cpu/fp32_cpu_kernels.hpp"
#include "host/host_verification.hpp"

using namespace sycl;
using namespace vph;

int main() {
    try {
        // Device Selector: select FPGA emulator or hardware device
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

        std::cout << "[INFO] Running on device: "
                  << q.get_device().get_info<info::device::name>() << std::endl;

        // 1. Generate test vectors using reusable host harness
        std::vector<FpRequest<4>> test_requests = HostVerification::generate_fp32_test_vectors();
        uint32_t num_items = static_cast<uint32_t>(test_requests.size());
        std::vector<FpResponse<4>> test_responses(num_items);

        std::cout << "[INFO] Enqueuing " << num_items << " test vectors to FP32 CPU IP pipeline..." << std::endl;

        // 2. Allocate SYCL device buffers
        {
            buffer<FpRequest<4>, 1> in_buf(test_requests.data(), range<1>(num_items));
            buffer<FpResponse<4>, 1> out_buf(test_responses.data(), range<1>(num_items));

            auto start_time = std::chrono::high_resolution_clock::now();

            // 3. Launch Consumer Kernel
            q.submit([&](handler& h) {
                accessor out_acc(out_buf, h, write_only, no_init);
                h.single_task<Fp32ConsumerTask>(Fp32ConsumerKernel(out_acc));
            });

            // 4. Launch Standalone FP32 CPU IP Kernel
            q.submit([&](handler& h) {
                h.single_task<Fp32IpTask>(
                    Fp32CpuIpKernel<Fp32InPipe, Fp32OutPipe>()
                );
            });

            // 5. Launch Producer Kernel
            q.submit([&](handler& h) {
                accessor in_acc(in_buf, h, read_only);
                h.single_task<Fp32ProducerTask>(Fp32ProducerKernel(in_acc, num_items));
            });

            // Wait for all 3 kernels to finish execution
            q.wait();

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end_time - start_time;
            std::cout << "[INFO] Execution completed in " << duration.count() << " ms." << std::endl;
        }

        // 6. Verify bit-exact correctness against software reference
        bool success = HostVerification::verify_fp32_results(test_requests, test_responses);
        if (!success) {
            std::cerr << "[FAILED] Verification did not pass." << std::endl;
            return 1;
        }

        std::cout << "[SUCCESS] FP32 CPU IP fully validated!" << std::endl;
        return 0;

    } catch (sycl::exception const& e) {
        std::cerr << "[ERROR] Caught SYCL exception: " << e.what() << std::endl;
        return 2;
    }
}
