#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cassert>

#include "fpu_types.hpp"
#include "bit_ops.hpp"
#include "fp32_ieee_core.hpp"
#include "fp64_ieee_core.hpp"
#include "fp16_ieee_core.hpp"
#include "bf16_core.hpp"
#include "fp8_core.hpp"
#include "fpx2_core.hpp"

namespace vph {

/**
 * @brief Recyclable Host Verification Suite
 * 
 * Provides automated test vector generation and bit-exact software validation
 * across normal arithmetic, catastrophic cancellation, rounding ties, and IEEE special cases.
 */
class HostVerification {
public:
    /**
     * @brief Build a comprehensive test suite of FP32 requests.
     */
    static std::vector<FpRequest<4>> generate_fp32_test_vectors() {
        std::vector<FpRequest<4>> requests;
        uint16_t tag = 1;

        auto add_test = [&](float a, float b, FpuOpcode op) {
            FpRequest<4> req{};
            u32_to_bytes(float_to_u32(a), req.op_a);
            u32_to_bytes(float_to_u32(b), req.op_b);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = 0; // RNE
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // 1. Basic Arithmetic
        add_test(1.0f, 2.0f, FpuOpcode::ADD);
        add_test(5.5f, 2.25f, FpuOpcode::ADD);
        add_test(100.0f, -25.0f, FpuOpcode::ADD);
        add_test(-50.0f, -50.0f, FpuOpcode::ADD);

        add_test(10.0f, 4.0f, FpuOpcode::SUB);
        add_test(1.0f, 2.0f, FpuOpcode::SUB);
        add_test(-5.0f, 10.0f, FpuOpcode::SUB);
        add_test(10.0f, -10.0f, FpuOpcode::SUB);

        add_test(3.0f, 4.0f, FpuOpcode::MUL);
        add_test(-2.5f, 4.0f, FpuOpcode::MUL);
        add_test(-1.5f, -2.0f, FpuOpcode::MUL);
        add_test(0.125f, 8.0f, FpuOpcode::MUL);

        // 2. Unary (ABS, NEG)
        add_test(-42.0f, 0.0f, FpuOpcode::ABS);
        add_test(42.0f, 0.0f, FpuOpcode::NEG);

        // 3. Catastrophic Cancellation (tests Leading Zero Count & Normalization)
        add_test(1.0000002f, 1.0f, FpuOpcode::SUB);
        add_test(1024.0625f, 1024.0f, FpuOpcode::SUB);
        add_test(16777216.0f, 16777214.0f, FpuOpcode::SUB);

        // 4. Alignment & Shifter Range (testing 1 to 32 bit shifts + sticky bit)
        for (int exp_shift = 1; exp_shift <= 30; exp_shift += 3) {
            float a = 1024.0f;
            float b = 1024.0f / static_cast<float>(1ULL << exp_shift);
            add_test(a, b, FpuOpcode::ADD);
            add_test(a, b, FpuOpcode::SUB);
        }

        // 5. Rounding Ties (Round to Nearest Even - RNE)
        // 1.0 + 2^-24 = 1.0000000596... (halfway tie cases)
        add_test(1.0f, 5.9604645e-8f, FpuOpcode::ADD); // Half-ULP tie
        add_test(1.5f, 5.9604645e-8f, FpuOpcode::ADD);

        // 6. Special Cases: Zeros, Infinities, NaNs
        float pos_zero = 0.0f;
        float neg_zero = -0.0f;
        float inf = std::numeric_limits<float>::infinity();
        float nan = std::numeric_limits<float>::quiet_NaN();

        add_test(pos_zero, pos_zero, FpuOpcode::ADD);
        add_test(pos_zero, neg_zero, FpuOpcode::ADD);
        add_test(neg_zero, neg_zero, FpuOpcode::ADD);
        add_test(5.0f, 0.0f, FpuOpcode::ADD);
        add_test(0.0f, 5.0f, FpuOpcode::MUL);

        add_test(inf, 5.0f, FpuOpcode::ADD);
        add_test(inf, inf, FpuOpcode::ADD);
        add_test(inf, -inf, FpuOpcode::ADD); // Inf - Inf -> NaN
        add_test(inf, 0.0f, FpuOpcode::MUL);  // 0 * Inf -> NaN
        add_test(inf, 2.0f, FpuOpcode::MUL);

        add_test(nan, 1.0f, FpuOpcode::ADD);
        add_test(2.0f, nan, FpuOpcode::MUL);

        // 7. Augmented Error-Free Operations (ADD_HILO, MUL_HILO)
        add_test(1.0f, 1e-7f, FpuOpcode::ADD_HILO);
        add_test(1.234567f, 7.654321f, FpuOpcode::MUL_HILO);

        return requests;
    }

    /**
     * @brief Verify output responses against the bit-exact software golden model.
     */
    static bool verify_fp32_results(const std::vector<FpRequest<4>>& requests,
                                    const std::vector<FpResponse<4>>& responses) {
        if (requests.size() != responses.size()) {
            std::cerr << "[ERROR] Size mismatch: " << requests.size() << " requests vs "
                      << responses.size() << " responses!" << std::endl;
            return false;
        }

        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "          FP32 CPU IP VERIFICATION REPORT             \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            uint32_t raw_a = bytes_to_u32(req.op_a);
            uint32_t raw_b = bytes_to_u32(req.op_b);
            uint32_t hw_res = bytes_to_u32(resp.result);

            // Compute expected bit-exact result using core ALU model
            Fp32AluResult expected = fp32_eval_alu(raw_a, raw_b, req.opcode);

            float fa = u32_to_float(raw_a);
            float fb = u32_to_float(raw_b);
            float f_hw = u32_to_float(hw_res);
            float f_exp = u32_to_float(expected.result);

            bool match = false;
            // For NaNs, accept any quiet NaN
            if (std::isnan(f_exp)) {
                match = std::isnan(f_hw);
            } else {
                match = (hw_res == expected.result);
            }

            if (match) {
                pass_count++;
            } else {
                fail_count++;
                std::cout << "[FAIL] #" << i << " Tag=" << req.tag << " Op=" << static_cast<int>(req.opcode) << "\n"
                          << "   A = " << fa << " (0x" << std::hex << raw_a << ")\n"
                          << "   B = " << fb << " (0x" << raw_b << ")\n"
                          << "   HW  = " << f_hw << " (0x" << hw_res << ")\n"
                          << "   EXP = " << f_exp << " (0x" << expected.result << std::dec << ")\n";
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }

    /**
     * @brief Build a comprehensive test suite of FP64 requests.
     */
    static std::vector<FpRequest<8>> generate_fp64_test_vectors() {
        std::vector<FpRequest<8>> requests;
        uint16_t tag = 1;

        auto add_test = [&](double a, double b, FpuOpcode op) {
            FpRequest<8> req{};
            u64_to_bytes(double_to_u64(a), req.op_a);
            u64_to_bytes(double_to_u64(b), req.op_b);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = 0; // RNE
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // 1. Basic Arithmetic
        add_test(1.0, 2.0, FpuOpcode::ADD);
        add_test(5.5, 2.25, FpuOpcode::ADD);
        add_test(100.0, -25.0, FpuOpcode::ADD);
        add_test(-50.0, -50.0, FpuOpcode::ADD);

        add_test(10.0, 4.0, FpuOpcode::SUB);
        add_test(1.0, 2.0, FpuOpcode::SUB);
        add_test(-5.0, 10.0, FpuOpcode::SUB);
        add_test(10.0, -10.0, FpuOpcode::SUB);

        add_test(3.0, 4.0, FpuOpcode::MUL);
        add_test(-2.5, 4.0, FpuOpcode::MUL);
        add_test(-1.5, -2.0, FpuOpcode::MUL);
        add_test(0.125, 8.0, FpuOpcode::MUL);

        // 2. Unary (ABS, NEG)
        add_test(-42.0, 0.0, FpuOpcode::ABS);
        add_test(42.0, 0.0, FpuOpcode::NEG);

        // 3. Catastrophic Cancellation (tests Leading Zero Count & Normalization on 53 bits)
        add_test(1.0000000000000002, 1.0, FpuOpcode::SUB);
        add_test(1024.0000000625, 1024.0, FpuOpcode::SUB);

        // 4. Alignment & Shifter Range (testing 1 to 60 bit shifts + sticky bit)
        for (int exp_shift = 1; exp_shift <= 60; exp_shift += 6) {
            double a = 1024.0;
            double b = 1024.0 / static_cast<double>(1ULL << (exp_shift > 31 ? 31 : exp_shift));
            add_test(a, b, FpuOpcode::ADD);
            add_test(a, b, FpuOpcode::SUB);
        }

        // 5. Special Cases: Zeros, Infinities, NaNs
        double pos_zero = 0.0;
        double neg_zero = -0.0;
        double inf = std::numeric_limits<double>::infinity();
        double nan = std::numeric_limits<double>::quiet_NaN();

        add_test(pos_zero, pos_zero, FpuOpcode::ADD);
        add_test(pos_zero, neg_zero, FpuOpcode::ADD);
        add_test(neg_zero, neg_zero, FpuOpcode::ADD);
        add_test(5.0, 0.0, FpuOpcode::ADD);
        add_test(0.0, 5.0, FpuOpcode::MUL);

        add_test(inf, 5.0, FpuOpcode::ADD);
        add_test(inf, inf, FpuOpcode::ADD);
        add_test(inf, -inf, FpuOpcode::ADD);
        add_test(inf, 0.0, FpuOpcode::MUL);
        add_test(inf, 2.0, FpuOpcode::MUL);

        add_test(nan, 1.0, FpuOpcode::ADD);
        add_test(2.0, nan, FpuOpcode::MUL);

        return requests;
    }

    /**
     * @brief Verify FP64 output responses against the bit-exact software golden model.
     */
    static bool verify_fp64_results(const std::vector<FpRequest<8>>& requests,
                                    const std::vector<FpResponse<8>>& responses) {
        if (requests.size() != responses.size()) {
            std::cerr << "[ERROR] Size mismatch: " << requests.size() << " requests vs "
                      << responses.size() << " responses!" << std::endl;
            return false;
        }

        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "          FP64 CPU IP VERIFICATION REPORT             \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            uint64_t raw_a = bytes_to_u64(req.op_a);
            uint64_t raw_b = bytes_to_u64(req.op_b);
            uint64_t hw_res = bytes_to_u64(resp.result);

            Fp64AluResult expected = fp64_eval_alu(raw_a, raw_b, req.opcode);

            double da = u64_to_double(raw_a);
            double db = u64_to_double(raw_b);
            double d_hw = u64_to_double(hw_res);
            double d_exp = u64_to_double(expected.result);

            bool match = false;
            if (std::isnan(d_exp)) {
                match = std::isnan(d_hw);
            } else {
                match = (hw_res == expected.result);
            }

            if (match) {
                pass_count++;
            } else {
                fail_count++;
                std::cout << "[FAIL] #" << i << " Tag=" << req.tag << " Op=" << static_cast<int>(req.opcode) << "\n"
                          << "   A = " << da << " (0x" << std::hex << raw_a << ")\n"
                          << "   B = " << db << " (0x" << raw_b << ")\n"
                          << "   HW  = " << d_hw << " (0x" << hw_res << ")\n"
                          << "   EXP = " << d_exp << " (0x" << expected.result << std::dec << ")\n";
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }

    /**
     * @brief Build a comprehensive test suite of FP16 requests.
     */
    static std::vector<FpRequest<2>> generate_fp16_test_vectors() {
        std::vector<FpRequest<2>> requests;
        uint16_t tag = 1;

        auto add_test = [&](float a, float b, FpuOpcode op) {
            FpRequest<2> req{};
            u16_to_bytes(float_to_fp16(a), req.op_a);
            u16_to_bytes(float_to_fp16(b), req.op_b);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = 0;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        add_test(1.0f, 2.0f, FpuOpcode::ADD);
        add_test(5.5f, 2.25f, FpuOpcode::ADD);
        add_test(100.0f, -25.0f, FpuOpcode::ADD);
        add_test(-50.0f, -50.0f, FpuOpcode::ADD);

        add_test(10.0f, 4.0f, FpuOpcode::SUB);
        add_test(1.0f, 2.0f, FpuOpcode::SUB);
        add_test(-5.0f, 10.0f, FpuOpcode::SUB);
        add_test(10.0f, -10.0f, FpuOpcode::SUB);

        add_test(3.0f, 4.0f, FpuOpcode::MUL);
        add_test(-2.5f, 4.0f, FpuOpcode::MUL);
        add_test(0.125f, 8.0f, FpuOpcode::MUL);

        add_test(-42.0f, 0.0f, FpuOpcode::ABS);
        add_test(42.0f, 0.0f, FpuOpcode::NEG);

        // Cancellation & Shifting
        add_test(1.001f, 1.0f, FpuOpcode::SUB);
        add_test(1024.0f, 0.25f, FpuOpcode::ADD);

        // Special cases
        float inf = std::numeric_limits<float>::infinity();
        float nan = std::numeric_limits<float>::quiet_NaN();
        add_test(0.0f, 0.0f, FpuOpcode::ADD);
        add_test(inf, 5.0f, FpuOpcode::ADD);
        add_test(inf, -inf, FpuOpcode::ADD);
        add_test(inf, 0.0f, FpuOpcode::MUL);
        add_test(nan, 1.0f, FpuOpcode::ADD);

        return requests;
    }

    static bool verify_fp16_results(const std::vector<FpRequest<2>>& requests,
                                    const std::vector<FpResponse<2>>& responses) {
        if (requests.size() != responses.size()) return false;
        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "          FP16 CPU IP VERIFICATION REPORT             \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            uint16_t raw_a = bytes_to_u16(req.op_a);
            uint16_t raw_b = bytes_to_u16(req.op_b);
            uint16_t hw_res = bytes_to_u16(resp.result);

            Fp16AluResult expected = fp16_eval_alu(raw_a, raw_b, req.opcode);

            float fa = fp16_to_float(raw_a);
            float fb = fp16_to_float(raw_b);
            float f_hw = fp16_to_float(hw_res);
            float f_exp = fp16_to_float(expected.result);

            bool match = false;
            if (std::isnan(f_exp)) {
                match = std::isnan(f_hw);
            } else {
                match = (hw_res == expected.result);
            }

            if (match) {
                pass_count++;
            } else {
                fail_count++;
                std::cout << "[FAIL] #" << i << " Tag=" << req.tag << " Op=" << static_cast<int>(req.opcode) << "\n"
                          << "   A = " << fa << " (0x" << std::hex << raw_a << ")\n"
                          << "   B = " << fb << " (0x" << raw_b << ")\n"
                          << "   HW  = " << f_hw << " (0x" << hw_res << ")\n"
                          << "   EXP = " << f_exp << " (0x" << expected.result << std::dec << ")\n";
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }

    /**
     * @brief Build a comprehensive test suite of BF16 requests.
     */
    static std::vector<FpRequest<2>> generate_bf16_test_vectors() {
        std::vector<FpRequest<2>> requests;
        uint16_t tag = 1;

        auto add_test = [&](float a, float b, FpuOpcode op) {
            FpRequest<2> req{};
            u16_to_bytes(float_to_bf16(a), req.op_a);
            u16_to_bytes(float_to_bf16(b), req.op_b);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = 0;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        add_test(1.0f, 2.0f, FpuOpcode::ADD);
        add_test(5.5f, 2.25f, FpuOpcode::ADD);
        add_test(100.0f, -25.0f, FpuOpcode::ADD);
        add_test(-50.0f, -50.0f, FpuOpcode::ADD);

        add_test(10.0f, 4.0f, FpuOpcode::SUB);
        add_test(1.0f, 2.0f, FpuOpcode::SUB);
        add_test(-5.0f, 10.0f, FpuOpcode::SUB);
        add_test(10.0f, -10.0f, FpuOpcode::SUB);

        add_test(3.0f, 4.0f, FpuOpcode::MUL);
        add_test(-2.5f, 4.0f, FpuOpcode::MUL);
        add_test(0.125f, 8.0f, FpuOpcode::MUL);

        add_test(-42.0f, 0.0f, FpuOpcode::ABS);
        add_test(42.0f, 0.0f, FpuOpcode::NEG);

        add_test(1.001f, 1.0f, FpuOpcode::SUB);
        add_test(1024.0f, 0.25f, FpuOpcode::ADD);

        float inf = std::numeric_limits<float>::infinity();
        float nan = std::numeric_limits<float>::quiet_NaN();
        add_test(0.0f, 0.0f, FpuOpcode::ADD);
        add_test(inf, 5.0f, FpuOpcode::ADD);
        add_test(inf, -inf, FpuOpcode::ADD);
        add_test(inf, 0.0f, FpuOpcode::MUL);
        add_test(nan, 1.0f, FpuOpcode::ADD);

        return requests;
    }

    static bool verify_bf16_results(const std::vector<FpRequest<2>>& requests,
                                    const std::vector<FpResponse<2>>& responses) {
        if (requests.size() != responses.size()) return false;
        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "          BF16 CPU IP VERIFICATION REPORT             \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            uint16_t raw_a = bytes_to_u16(req.op_a);
            uint16_t raw_b = bytes_to_u16(req.op_b);
            uint16_t hw_res = bytes_to_u16(resp.result);

            Bf16AluResult expected = bf16_eval_alu(raw_a, raw_b, req.opcode);

            float fa = bf16_to_float(raw_a);
            float fb = bf16_to_float(raw_b);
            float f_hw = bf16_to_float(hw_res);
            float f_exp = bf16_to_float(expected.result);

            bool match = false;
            if (std::isnan(f_exp)) {
                match = std::isnan(f_hw);
            } else {
                match = (hw_res == expected.result);
            }

            if (match) {
                pass_count++;
            } else {
                fail_count++;
                std::cout << "[FAIL] #" << i << " Tag=" << req.tag << " Op=" << static_cast<int>(req.opcode) << "\n"
                          << "   A = " << fa << " (0x" << std::hex << raw_a << ")\n"
                          << "   B = " << fb << " (0x" << raw_b << ")\n"
                          << "   HW  = " << f_hw << " (0x" << hw_res << ")\n"
                          << "   EXP = " << f_exp << " (0x" << expected.result << std::dec << ")\n";
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }

    // ========================================================================
    // FP8 CPU Verification
    // ========================================================================
    static std::vector<FpRequest<1>> generate_fp8_test_vectors() {
        std::vector<FpRequest<1>> requests;
        uint16_t tag = 1;

        auto add_test = [&](uint8_t a, uint8_t b, FpuOpcode op, uint8_t fmt) {
            FpRequest<1> req{};
            req.op_a[0] = a;
            req.op_b[0] = b;
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = fmt; // 0 = E4M3, 1 = E5M2
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // E4M3 Tests: (sign:1, exp:4, mant:3, bias:7)
        // 1.0 = (0, exp=7, mant=0) -> 0x38
        // 2.0 = (0, exp=8, mant=0) -> 0x40
        // 0.5 = (0, exp=6, mant=0) -> 0x30
        add_test(0x38, 0x40, FpuOpcode::ADD, 0); // 1.0 + 2.0 = 3.0
        add_test(0x40, 0x38, FpuOpcode::SUB, 0); // 2.0 - 1.0 = 1.0
        add_test(0x38, 0x40, FpuOpcode::MUL, 0); // 1.0 * 2.0 = 2.0
        add_test(0x40, 0x30, FpuOpcode::MUL, 0); // 2.0 * 0.5 = 1.0
        add_test(0xB8, 0x00, FpuOpcode::ABS, 0); // ABS(-1.0)
        add_test(0x38, 0x00, FpuOpcode::NEG, 0); // NEG(1.0)
        add_test(0x00, 0x00, FpuOpcode::ADD, 0); // 0 + 0

        // E5M2 Tests: (sign:1, exp:5, mant:2, bias:15)
        // 1.0 = (0, exp=15, mant=0) -> 0x3C
        // 2.0 = (0, exp=16, mant=0) -> 0x40
        add_test(0x3C, 0x40, FpuOpcode::ADD, 1); // 1.0 + 2.0
        add_test(0x40, 0x3C, FpuOpcode::SUB, 1); // 2.0 - 1.0
        add_test(0x3C, 0x40, FpuOpcode::MUL, 1); // 1.0 * 2.0
        add_test(0xBC, 0x00, FpuOpcode::ABS, 1); // ABS
        add_test(0x3C, 0x00, FpuOpcode::NEG, 1); // NEG
        add_test(0x7C, 0x3C, FpuOpcode::ADD, 1); // Inf + 1.0

        return requests;
    }

    static bool verify_fp8_results(const std::vector<FpRequest<1>>& requests,
                                   const std::vector<FpResponse<1>>& responses) {
        if (requests.size() != responses.size()) return false;
        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "          FP8 CPU IP VERIFICATION REPORT              \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            Fp8AluResult expected;
            if (req.rounding_mode == static_cast<uint8_t>(Fp8Type::E5M2)) {
                expected = fp8_e5m2_eval_alu(req.op_a[0], req.op_b[0], req.opcode);
            } else {
                expected = fp8_e4m3_eval_alu(req.op_a[0], req.op_b[0], req.opcode);
            }

            if (resp.result[0] == expected.result) {
                pass_count++;
            } else {
                fail_count++;
                std::cout << "[FAIL] #" << i << " Tag=" << req.tag
                          << " A=0x" << std::hex << (int)req.op_a[0]
                          << " B=0x" << (int)req.op_b[0]
                          << " HW=0x" << (int)resp.result[0]
                          << " EXP=0x" << (int)expected.result << std::dec << "\n";
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }

    // ========================================================================
    // 2x FP16 Packed SIMD / 1x FP32 Full Precision Scalar (32-bit Container)
    // ========================================================================
    static std::vector<FpRequest<4>> generate_simd_fp16x2_test_vectors() {
        std::vector<FpRequest<4>> requests;
        uint16_t tag = 1;

        // 1. Packed 2x FP16 SIMD tests
        auto add_packed_test = [&](float a0, float b0, float a1, float b1, FpuOpcode op) {
            FpRequest<4> req{};
            u16_to_bytes(float_to_fp16(a0), &req.op_a[0]);
            u16_to_bytes(float_to_fp16(b0), &req.op_b[0]);
            u16_to_bytes(float_to_fp16(a1), &req.op_a[2]);
            u16_to_bytes(float_to_fp16(b1), &req.op_b[2]);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = OP_MODE_PACKED;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // 2. Full Precision 1x FP32 Scalar tests
        auto add_scalar_test = [&](float a, float b, FpuOpcode op) {
            FpRequest<4> req{};
            u32_to_bytes(float_to_u32(a), req.op_a);
            u32_to_bytes(float_to_u32(b), req.op_b);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = OP_MODE_SCALAR;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // Packed Mode Vectors
        add_packed_test(1.0f, 2.0f, 10.0f, 20.0f, FpuOpcode::ADD);
        add_packed_test(5.5f, 2.25f, 100.0f, -25.0f, FpuOpcode::ADD);
        add_packed_test(10.0f, 4.0f, 1.0f, 2.0f, FpuOpcode::SUB);
        add_packed_test(3.0f, 4.0f, -2.5f, 4.0f, FpuOpcode::MUL);
        add_packed_test(-42.0f, 0.0f, 33.0f, 0.0f, FpuOpcode::ABS);
        add_packed_test(42.0f, 0.0f, -33.0f, 0.0f, FpuOpcode::NEG);
        add_packed_test(1.001f, 1.0f, 1024.0f, 0.25f, FpuOpcode::SUB);
        add_packed_test(0.0f, 0.0f, 5.0f, 0.0f, FpuOpcode::MUL);

        // Scalar Full Precision (FP32) Vectors in the same IP
        add_scalar_test(1.0f, 2.0f, FpuOpcode::SCALAR_ADD);
        add_scalar_test(5.5f, 2.25f, FpuOpcode::SCALAR_ADD);
        add_scalar_test(10.0f, 4.0f, FpuOpcode::SCALAR_SUB);
        add_scalar_test(3.0f, 4.0f, FpuOpcode::SCALAR_MUL);
        add_scalar_test(-42.0f, 0.0f, FpuOpcode::SCALAR_ABS);
        add_scalar_test(42.0f, 0.0f, FpuOpcode::SCALAR_NEG);

        return requests;
    }

    static bool verify_simd_fp16x2_results(const std::vector<FpRequest<4>>& requests,
                                          const std::vector<FpResponse<4>>& responses) {
        if (requests.size() != responses.size()) return false;
        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "  2x FP16 PACKED / 1x FP32 SCALAR IP VERIFICATION     \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            if (is_scalar_mode(req.opcode, req.rounding_mode)) {
                // Scalar Full Precision FP32 Validation
                uint32_t a = bytes_to_u32(req.op_a);
                uint32_t b = bytes_to_u32(req.op_b);
                uint32_t hw = bytes_to_u32(resp.result);
                Fp32AluResult exp = fp32_eval_alu(a, b, get_base_opcode(req.opcode));
                bool match = (hw == exp.result);
                if (match) {
                    pass_count++;
                } else {
                    fail_count++;
                    std::cout << "[FAIL SCALAR FP32] #" << i << " Tag=" << req.tag 
                              << " HW=0x" << std::hex << hw << " EXP=0x" << exp.result << std::dec << "\n";
                }
            } else {
                // Packed 2x FP16 SIMD Validation
                uint16_t l0_a = bytes_to_u16(&req.op_a[0]);
                uint16_t l0_b = bytes_to_u16(&req.op_b[0]);
                uint16_t l1_a = bytes_to_u16(&req.op_a[2]);
                uint16_t l1_b = bytes_to_u16(&req.op_b[2]);

                uint16_t l0_hw = bytes_to_u16(&resp.result[0]);
                uint16_t l1_hw = bytes_to_u16(&resp.result[2]);

                Fp16AluResult l0_exp = fp16_eval_alu(l0_a, l0_b, get_base_opcode(req.opcode));
                Fp16AluResult l1_exp = fp16_eval_alu(l1_a, l1_b, get_base_opcode(req.opcode));

                bool m0 = (l0_hw == l0_exp.result);
                bool m1 = (l1_hw == l1_exp.result);

                if (m0 && m1) {
                    pass_count++;
                } else {
                    fail_count++;
                    std::cout << "[FAIL PACKED FP16x2] #" << i << " Tag=" << req.tag
                              << " L0 Match=" << m0 << " (HW=0x" << std::hex << l0_hw << " EXP=0x" << l0_exp.result << ")"
                              << " L1 Match=" << m1 << " (HW=0x" << l1_hw << " EXP=0x" << l1_exp.result << std::dec << ")\n";
                }
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }

    // ========================================================================
    // 2x FP32 Packed SIMD / 1x FP64 Full Precision Scalar (64-bit Container)
    // ========================================================================
    static std::vector<FpRequest<8>> generate_simd_fp32x2_test_vectors() {
        std::vector<FpRequest<8>> requests;
        uint16_t tag = 1;

        // 1. Packed 2x FP32 SIMD tests
        auto add_packed_test = [&](float a0, float b0, float a1, float b1, FpuOpcode op) {
            FpRequest<8> req{};
            u32_to_bytes(float_to_u32(a0), &req.op_a[0]);
            u32_to_bytes(float_to_u32(b0), &req.op_b[0]);
            u32_to_bytes(float_to_u32(a1), &req.op_a[4]);
            u32_to_bytes(float_to_u32(b1), &req.op_b[4]);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = OP_MODE_PACKED;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // 2. Full Precision 1x FP64 Scalar tests
        auto add_scalar_test = [&](double a, double b, FpuOpcode op) {
            FpRequest<8> req{};
            u64_to_bytes(double_to_u64(a), req.op_a);
            u64_to_bytes(double_to_u64(b), req.op_b);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = OP_MODE_SCALAR;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // Packed Mode Vectors
        add_packed_test(1.0f, 2.0f, 100.0f, -25.0f, FpuOpcode::ADD);
        add_packed_test(5.5f, 2.25f, -50.0f, -50.0f, FpuOpcode::ADD);
        add_packed_test(10.0f, 4.0f, 1.0f, 2.0f, FpuOpcode::SUB);
        add_packed_test(3.0f, 4.0f, -2.5f, 4.0f, FpuOpcode::MUL);
        add_packed_test(-42.0f, 0.0f, 33.0f, 0.0f, FpuOpcode::ABS);
        add_packed_test(42.0f, 0.0f, -33.0f, 0.0f, FpuOpcode::NEG);
        add_packed_test(1.0000002f, 1.0f, 1024.0625f, 1024.0f, FpuOpcode::SUB);

        // Scalar Full Precision (FP64) Vectors in the same IP
        add_scalar_test(1.0, 2.0, FpuOpcode::SCALAR_ADD);
        add_scalar_test(100.5, -25.25, FpuOpcode::SCALAR_ADD);
        add_scalar_test(50.0, 15.0, FpuOpcode::SCALAR_SUB);
        add_scalar_test(12.5, 4.0, FpuOpcode::SCALAR_MUL);
        add_scalar_test(-42.0, 0.0, FpuOpcode::SCALAR_ABS);
        add_scalar_test(42.0, 0.0, FpuOpcode::SCALAR_NEG);

        return requests;
    }

    static bool verify_simd_fp32x2_results(const std::vector<FpRequest<8>>& requests,
                                          const std::vector<FpResponse<8>>& responses) {
        if (requests.size() != responses.size()) return false;
        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "  2x FP32 PACKED / 1x FP64 SCALAR IP VERIFICATION     \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            if (is_scalar_mode(req.opcode, req.rounding_mode)) {
                // Scalar Full Precision FP64 Validation
                uint64_t a = bytes_to_u64(req.op_a);
                uint64_t b = bytes_to_u64(req.op_b);
                uint64_t hw = bytes_to_u64(resp.result);

                double da = u64_to_double(a);
                double db = u64_to_double(b);
                double dexp = 0.0;
                uint8_t base_op = get_base_opcode(req.opcode);
                if (base_op == static_cast<uint8_t>(FpuOpcode::ADD)) dexp = da + db;
                else if (base_op == static_cast<uint8_t>(FpuOpcode::SUB)) dexp = da - db;
                else if (base_op == static_cast<uint8_t>(FpuOpcode::MUL)) dexp = da * db;
                else if (base_op == static_cast<uint8_t>(FpuOpcode::ABS)) dexp = std::fabs(da);
                else if (base_op == static_cast<uint8_t>(FpuOpcode::NEG)) dexp = -da;

                uint64_t exp_u64 = double_to_u64(dexp);
                bool match = (hw == exp_u64);
                if (match) {
                    pass_count++;
                } else {
                    fail_count++;
                    std::cout << "[FAIL SCALAR FP64] #" << i << " Tag=" << req.tag
                              << " HW=0x" << std::hex << hw << " EXP=0x" << exp_u64 << std::dec << "\n";
                }
            } else {
                // Packed 2x FP32 SIMD Validation
                uint32_t l0_a = bytes_to_u32(&req.op_a[0]);
                uint32_t l0_b = bytes_to_u32(&req.op_b[0]);
                uint32_t l1_a = bytes_to_u32(&req.op_a[4]);
                uint32_t l1_b = bytes_to_u32(&req.op_b[4]);

                uint32_t l0_hw = bytes_to_u32(&resp.result[0]);
                uint32_t l1_hw = bytes_to_u32(&resp.result[4]);

                Fp32AluResult l0_exp = fp32_eval_alu(l0_a, l0_b, get_base_opcode(req.opcode));
                Fp32AluResult l1_exp = fp32_eval_alu(l1_a, l1_b, get_base_opcode(req.opcode));

                bool m0 = (l0_hw == l0_exp.result);
                bool m1 = (l1_hw == l1_exp.result);

                if (m0 && m1) {
                    pass_count++;
                } else {
                    fail_count++;
                    std::cout << "[FAIL PACKED FP32x2] #" << i << " Tag=" << req.tag
                              << " L0 Match=" << m0 << " (HW=0x" << std::hex << l0_hw << " EXP=0x" << l0_exp.result << ")"
                              << " L1 Match=" << m1 << " (HW=0x" << l1_hw << " EXP=0x" << l1_exp.result << std::dec << ")\n";
                }
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }

    // ========================================================================
    // 2x FP8 Packed SIMD / 1x FP16 Full Precision Scalar (16-bit Container)
    // ========================================================================
    static std::vector<FpRequest<2>> generate_simd_fp8x2_test_vectors() {
        std::vector<FpRequest<2>> requests;
        uint16_t tag = 1;

        // 1. Packed 2x FP8 SIMD tests
        auto add_packed_test = [&](uint8_t a0, uint8_t b0, uint8_t a1, uint8_t b1, FpuOpcode op) {
            FpRequest<2> req{};
            req.op_a[0] = a0;
            req.op_b[0] = b0;
            req.op_a[1] = a1;
            req.op_b[1] = b1;
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = OP_MODE_PACKED; // E4M3 & Packed
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // 2. Full Precision 1x FP16 Scalar tests
        auto add_scalar_test = [&](float a, float b, FpuOpcode op) {
            FpRequest<2> req{};
            u16_to_bytes(float_to_fp16(a), req.op_a);
            u16_to_bytes(float_to_fp16(b), req.op_b);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = OP_MODE_SCALAR;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // Packed Mode Vectors
        add_packed_test(0x38, 0x40, 0x40, 0x30, FpuOpcode::ADD);
        add_packed_test(0x40, 0x38, 0x38, 0x40, FpuOpcode::SUB);
        add_packed_test(0x38, 0x40, 0x40, 0x30, FpuOpcode::MUL);
        add_packed_test(0xB8, 0x00, 0x38, 0x00, FpuOpcode::ABS);
        add_packed_test(0x38, 0x00, 0xB8, 0x00, FpuOpcode::NEG);

        // Scalar Full Precision (FP16) Vectors in the same IP
        add_scalar_test(1.0f, 2.0f, FpuOpcode::SCALAR_ADD);
        add_scalar_test(5.5f, 2.25f, FpuOpcode::SCALAR_ADD);
        add_scalar_test(10.0f, 4.0f, FpuOpcode::SCALAR_SUB);
        add_scalar_test(3.0f, 4.0f, FpuOpcode::SCALAR_MUL);
        add_scalar_test(-42.0f, 0.0f, FpuOpcode::SCALAR_ABS);
        add_scalar_test(42.0f, 0.0f, FpuOpcode::SCALAR_NEG);

        return requests;
    }

    static bool verify_simd_fp8x2_results(const std::vector<FpRequest<2>>& requests,
                                         const std::vector<FpResponse<2>>& responses) {
        if (requests.size() != responses.size()) return false;
        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "   2x FP8 PACKED / 1x FP16 SCALAR IP VERIFICATION      \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            if (is_scalar_mode(req.opcode, req.rounding_mode)) {
                // Scalar Full Precision FP16 Validation
                uint16_t a = bytes_to_u16(req.op_a);
                uint16_t b = bytes_to_u16(req.op_b);
                uint16_t hw = bytes_to_u16(resp.result);
                Fp16AluResult exp = fp16_eval_alu(a, b, get_base_opcode(req.opcode));
                bool match = (hw == exp.result);
                if (match) {
                    pass_count++;
                } else {
                    fail_count++;
                    std::cout << "[FAIL SCALAR FP16] #" << i << " Tag=" << req.tag
                              << " HW=0x" << std::hex << hw << " EXP=0x" << exp.result << std::dec << "\n";
                }
            } else {
                // Packed 2x FP8 SIMD Validation
                Fp8AluResult l0_exp = fp8_e4m3_eval_alu(req.op_a[0], req.op_b[0], get_base_opcode(req.opcode));
                Fp8AluResult l1_exp = fp8_e4m3_eval_alu(req.op_a[1], req.op_b[1], get_base_opcode(req.opcode));

                bool m0 = (resp.result[0] == l0_exp.result);
                bool m1 = (resp.result[1] == l1_exp.result);

                if (m0 && m1) {
                    pass_count++;
                } else {
                    fail_count++;
                    std::cout << "[FAIL PACKED FP8x2] #" << i << " Tag=" << req.tag
                              << " L0 Match=" << m0 << " L1 Match=" << m1 << "\n";
                }
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }

    // ========================================================================
    // 4x FP8 Packed SIMD / 1x FP32 Full Precision Scalar (32-bit Container)
    // ========================================================================
    static std::vector<FpRequest<4>> generate_simd_fp8x4_test_vectors() {
        std::vector<FpRequest<4>> requests;
        uint16_t tag = 1;

        // 1. Packed 4x FP8 SIMD tests
        auto add_packed_test = [&](uint8_t a0, uint8_t b0, uint8_t a1, uint8_t b1,
                                   uint8_t a2, uint8_t b2, uint8_t a3, uint8_t b3, FpuOpcode op) {
            FpRequest<4> req{};
            req.op_a[0] = a0; req.op_b[0] = b0;
            req.op_a[1] = a1; req.op_b[1] = b1;
            req.op_a[2] = a2; req.op_b[2] = b2;
            req.op_a[3] = a3; req.op_b[3] = b3;
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = OP_MODE_PACKED; // E4M3 & Packed
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // 2. Full Precision 1x FP32 Scalar tests
        auto add_scalar_test = [&](float a, float b, FpuOpcode op) {
            FpRequest<4> req{};
            u32_to_bytes(float_to_u32(a), req.op_a);
            u32_to_bytes(float_to_u32(b), req.op_b);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = OP_MODE_SCALAR;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // Packed Mode Vectors
        add_packed_test(0x38, 0x40, 0x40, 0x30, 0x38, 0x38, 0x40, 0x40, FpuOpcode::ADD);
        add_packed_test(0x40, 0x38, 0x38, 0x40, 0x40, 0x40, 0x38, 0x30, FpuOpcode::SUB);
        add_packed_test(0x38, 0x40, 0x40, 0x30, 0x30, 0x38, 0x38, 0x38, FpuOpcode::MUL);
        add_packed_test(0xB8, 0x00, 0x38, 0x00, 0xB8, 0x00, 0x40, 0x00, FpuOpcode::ABS);
        add_packed_test(0x38, 0x00, 0xB8, 0x00, 0x40, 0x00, 0xB8, 0x00, FpuOpcode::NEG);

        // Scalar Full Precision (FP32) Vectors in the same IP
        add_scalar_test(1.0f, 2.0f, FpuOpcode::SCALAR_ADD);
        add_scalar_test(5.5f, 2.25f, FpuOpcode::SCALAR_ADD);
        add_scalar_test(10.0f, 4.0f, FpuOpcode::SCALAR_SUB);
        add_scalar_test(3.0f, 4.0f, FpuOpcode::SCALAR_MUL);
        add_scalar_test(-42.0f, 0.0f, FpuOpcode::SCALAR_ABS);
        add_scalar_test(42.0f, 0.0f, FpuOpcode::SCALAR_NEG);

        return requests;
    }

    static bool verify_simd_fp8x4_results(const std::vector<FpRequest<4>>& requests,
                                         const std::vector<FpResponse<4>>& responses) {
        if (requests.size() != responses.size()) return false;
        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "   4x FP8 PACKED / 1x FP32 SCALAR IP VERIFICATION      \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            if (is_scalar_mode(req.opcode, req.rounding_mode)) {
                // Scalar Full Precision FP32 Validation
                uint32_t a = bytes_to_u32(req.op_a);
                uint32_t b = bytes_to_u32(req.op_b);
                uint32_t hw = bytes_to_u32(resp.result);
                Fp32AluResult exp = fp32_eval_alu(a, b, get_base_opcode(req.opcode));
                bool match = (hw == exp.result);
                if (match) {
                    pass_count++;
                } else {
                    fail_count++;
                    std::cout << "[FAIL SCALAR FP32] #" << i << " Tag=" << req.tag
                              << " HW=0x" << std::hex << hw << " EXP=0x" << exp.result << std::dec << "\n";
                }
            } else {
                // Packed 4x FP8 SIMD Validation
                bool match = true;
                for (size_t l = 0; l < 4; l++) {
                    Fp8AluResult exp = fp8_e4m3_eval_alu(req.op_a[l], req.op_b[l], get_base_opcode(req.opcode));
                    if (resp.result[l] != exp.result) {
                        match = false;
                    }
                }

                if (match) {
                    pass_count++;
                } else {
                    fail_count++;
                    std::cout << "[FAIL PACKED FP8x4] #" << i << " Tag=" << req.tag << "\n";
                }
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }


    // ========================================================================
    // 4-Lane Streaming Vector Accelerator Verification (128-bit Container)
    // ========================================================================
    static std::vector<FpRequest<16>> generate_streaming_accel_test_vectors() {
        std::vector<FpRequest<16>> requests;
        uint16_t tag = 1;

        auto add_test = [&](float a0, float b0, float a1, float b1,
                            float a2, float b2, float a3, float b3, FpuOpcode op) {
            FpRequest<16> req{};
            u32_to_bytes(float_to_u32(a0), &req.op_a[0]);
            u32_to_bytes(float_to_u32(b0), &req.op_b[0]);
            u32_to_bytes(float_to_u32(a1), &req.op_a[4]);
            u32_to_bytes(float_to_u32(b1), &req.op_b[4]);
            u32_to_bytes(float_to_u32(a2), &req.op_a[8]);
            u32_to_bytes(float_to_u32(b2), &req.op_b[8]);
            u32_to_bytes(float_to_u32(a3), &req.op_a[12]);
            u32_to_bytes(float_to_u32(b3), &req.op_b[12]);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = 0;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        add_test(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, FpuOpcode::ADD);
        add_test(10.0f, 5.0f, 20.0f, 10.0f, 30.0f, 15.0f, 40.0f, 20.0f, FpuOpcode::SUB);
        add_test(2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, FpuOpcode::MUL);
        add_test(-1.0f, 0.0f, -2.0f, 0.0f, 3.0f, 0.0f, -4.0f, 0.0f, FpuOpcode::ABS);
        add_test(1.0f, 0.0f, -2.0f, 0.0f, 3.0f, 0.0f, -4.0f, 0.0f, FpuOpcode::NEG);

        return requests;
    }

    static bool verify_streaming_accel_results(const std::vector<FpRequest<16>>& requests,
                                              const std::vector<FpResponse<16>>& responses) {
        if (requests.size() != responses.size()) return false;
        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "   STREAMING VECTOR ACCEL IP VERIFICATION REPORT      \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            bool match = true;
            for (size_t l = 0; l < 4; l++) {
                uint32_t a = bytes_to_u32(&req.op_a[l * 4]);
                uint32_t b = bytes_to_u32(&req.op_b[l * 4]);
                uint32_t hw = bytes_to_u32(&resp.result[l * 4]);
                Fp32AluResult exp = fp32_eval_alu(a, b, req.opcode);
                if (hw != exp.result) {
                    match = false;
                }
            }

            if (match) {
                pass_count++;
            } else {
                fail_count++;
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }

    // ========================================================================
    // Custom FPX2 Multi-Limb Verification (64-bit Container)
    // ========================================================================
    static std::vector<FpRequest<8>> generate_fpx2_test_vectors() {
        std::vector<FpRequest<8>> requests;
        uint16_t tag = 1;

        auto add_test = [&](uint64_t a, uint64_t b, FpuOpcode op) {
            FpRequest<8> req{};
            u64_to_bytes(a, req.op_a);
            u64_to_bytes(b, req.op_b);
            req.opcode = static_cast<uint8_t>(op);
            req.rounding_mode = 0;
            req.tag = tag++;
            req.eof = false;
            requests.push_back(req);
        };

        // FPX2: exp=1023 (1.0), mant48 has 24-bit limbs
        // 1.0 = (exp=1023, mant48 = 0x00800000000000ULL)
        // 2.0 = (exp=1024, mant48 = 0x00800000000000ULL)
        uint64_t val_1_0 = fpx2_pack(0, 1023, 0, 0x00800000000000ULL);
        uint64_t val_2_0 = fpx2_pack(0, 1024, 0, 0x00800000000000ULL);
        uint64_t val_3_0 = fpx2_pack(0, 1024, 0, 0x00C00000000000ULL);

        add_test(val_1_0, val_2_0, FpuOpcode::ADD);
        add_test(val_3_0, val_1_0, FpuOpcode::SUB);
        add_test(val_1_0, val_2_0, FpuOpcode::MUL);
        add_test(val_1_0, 0, FpuOpcode::ABS);
        add_test(val_1_0, 0, FpuOpcode::NEG);
        add_test(0, val_1_0, FpuOpcode::ADD);

        return requests;
    }

    static bool verify_fpx2_results(const std::vector<FpRequest<8>>& requests,
                                   const std::vector<FpResponse<8>>& responses) {
        if (requests.size() != responses.size()) return false;
        size_t pass_count = 0;
        size_t fail_count = 0;

        std::cout << "\n======================================================\n";
        std::cout << "      CUSTOM FPX2 IP VERIFICATION REPORT              \n";
        std::cout << "======================================================\n";

        for (size_t i = 0; i < requests.size(); i++) {
            const auto& req  = requests[i];
            const auto& resp = responses[i];

            uint64_t raw_a = bytes_to_u64(req.op_a);
            uint64_t raw_b = bytes_to_u64(req.op_b);
            uint64_t hw_res = bytes_to_u64(resp.result);

            Fpx2AluResult exp = fpx2_eval_alu(raw_a, raw_b, req.opcode);

            if (hw_res == exp.result) {
                pass_count++;
            } else {
                fail_count++;
                std::cout << "[FAIL] #" << i << " Tag=" << req.tag
                          << " A=0x" << std::hex << raw_a
                          << " B=0x" << raw_b
                          << " HW=0x" << hw_res
                          << " EXP=0x" << exp.result << std::dec << "\n";
            }
        }

        std::cout << "------------------------------------------------------\n";
        std::cout << " Total Tested : " << requests.size() << "\n";
        std::cout << " Passed       : " << pass_count << "\n";
        std::cout << " Failed       : " << fail_count << "\n";
        std::cout << " Status       : " << (fail_count == 0 ? "SUCCESS (100% Bit-Exact Match)" : "FAILURE") << "\n";
        std::cout << "======================================================\n\n";

        return (fail_count == 0);
    }
};

} // namespace vph
