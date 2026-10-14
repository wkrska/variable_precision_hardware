#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace vph {

/**
 * @brief Universal Floating-Point Operation Codes
 * 
 * Includes standard arithmetic (ADD, SUB, MUL) as well as augmented
 * dual-result instructions (ADD_HILO, MUL_HILO) for error-free transformations
 * (EFT / float-float expansions) as outlined in the research architecture.
 */
enum class FpuOpcode : uint8_t {
    NOP      = 0x00, // No-operation / passthrough
    ADD      = 0x01, // Single-result standard addition: round(a + b)
    SUB      = 0x02, // Single-result standard subtraction: round(a - b)
    MUL      = 0x03, // Single-result standard multiplication: round(a * b)
    ABS      = 0x04, // Unary absolute value: |a|
    NEG      = 0x05, // Unary negation: -a
    
    // Augmented operations for float-float / multi-limb expansions (IEEE 754-2019 / FPADDRE / FPMULRE)
    ADD_HILO = 0x10, // Returns {result = fl(a+b), residual = exact addition error}
    MUL_HILO = 0x11, // Returns {result = fl(a*b), residual = exact multiplication error}

    // Scalar full-precision mode opcodes (bit 7 set) for dual-mode packed SIMD IPs
    SCALAR_ADD      = 0x81, // Scalar full precision addition
    SCALAR_SUB      = 0x82, // Scalar full precision subtraction
    SCALAR_MUL      = 0x83, // Scalar full precision multiplication
    SCALAR_ABS      = 0x84, // Scalar full precision absolute value
    SCALAR_NEG      = 0x85, // Scalar full precision negation
    SCALAR_ADD_HILO = 0x90, // Scalar full precision augmented addition
    SCALAR_MUL_HILO = 0x91, // Scalar full precision augmented multiplication
};

// Mode flags for dynamic switching between Packed SIMD and Full-Precision Scalar in the same IP
constexpr uint8_t OP_MODE_PACKED = 0x00; ///< Operate on sub-precision lanes in parallel (e.g. 2x FP32, 2x FP16, etc.)
constexpr uint8_t OP_MODE_SCALAR = 0x80; ///< Operate on container at full precision (e.g. 1x FP64, 1x FP32, 1x FP16)
constexpr uint8_t OP_CODE_MASK   = 0x7F; ///< Mask to extract base operation

inline constexpr bool is_scalar_mode(uint8_t opcode, uint8_t rounding_mode = 0) {
    return ((opcode & OP_MODE_SCALAR) != 0) || ((rounding_mode & OP_MODE_SCALAR) != 0);
}

inline constexpr uint8_t get_base_opcode(uint8_t opcode) {
    return opcode & OP_CODE_MASK;
}

/**
 * @brief Floating-Point Status & Exception Flags (compatible with IEEE 754 / XRSTAT)
 */
struct FpuStatus {
    static constexpr uint8_t OK        = 0x00;
    static constexpr uint8_t INVALID   = 1 << 0; // Bit 0: Invalid operation (e.g., NaN generation, Inf - Inf)
    static constexpr uint8_t DIV_ZERO  = 1 << 1; // Bit 1: Division by zero
    static constexpr uint8_t OVERFLOW  = 1 << 2; // Bit 2: Result exceeded format range -> turned to Inf
    static constexpr uint8_t UNDERFLOW = 1 << 3; // Bit 3: Tiny result flushed to zero or subnormal
    static constexpr uint8_t INEXACT   = 1 << 4; // Bit 4: Result required rounding (lost precision)
    static constexpr uint8_t IS_NAN    = 1 << 5; // Bit 5: Result is NaN
    static constexpr uint8_t IS_INF    = 1 << 6; // Bit 6: Result is Infinity (+/- Inf)
    static constexpr uint8_t IS_ZERO   = 1 << 7; // Bit 7: Result is Zero (+/- 0)
};

/**
 * @brief Universal Request Packet for FPU IPs
 * 
 * Parameterized by byte-width `NBytes` (e.g. 4 for FP32, 8 for FP64, 2 for FP16/BF16, 1 for FP8).
 * Inputs are represented as explicit byte arrays to ensure strict memory interface neutrality.
 */
template <size_t NBytes>
struct alignas(4) FpRequest {
    uint8_t op_a[NBytes];   ///< First operand byte array (little-endian)
    uint8_t op_b[NBytes];   ///< Second operand byte array (little-endian)
    uint8_t opcode;         ///< FpuOpcode value
    uint8_t rounding_mode;  ///< Rounding mode (0 = Round-to-Nearest-Even RNE)
    uint16_t tag;           ///< Transaction tag / sequence ID
    bool eof;               ///< End-of-stream indicator to terminate consumer
};

/**
 * @brief Universal Response Packet from FPU IPs
 * 
 * Returns the primary rounded result, residual term (for augmented EFT operations),
 * status flags, and preserves transaction tags.
 */
template <size_t NBytes>
struct alignas(4) FpResponse {
    uint8_t result[NBytes];   ///< Primary rounded result byte array
    uint8_t residual[NBytes]; ///< Residual / error term (for ADD_HILO / MUL_HILO)
    uint8_t status;           ///< Bitmask of FpuStatus flags
    uint8_t opcode;           ///< Echoed opcode
    uint16_t tag;             ///< Echoed transaction tag
    bool eof;                 ///< End-of-stream indicator
};

} // namespace vph
