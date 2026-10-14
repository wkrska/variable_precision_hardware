#pragma once

#include <cstdint>
#include "fpu_types.hpp"

namespace vph {

/**
 * @brief Stage 1 Pipeline Register: Unpack, Decode & Alignment Prep
 */
struct Stage1Reg {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode;

    // Unpacked fields
    uint32_t raw_a;
    uint32_t raw_b;

    // Fast-path bypass for special values / unary
    bool     is_special;
    uint32_t special_result;
    uint8_t  special_status;

    // ADD/SUB prep
    bool     large_sign;
    int16_t  large_exp;
    uint32_t large_mant;
    bool     small_sign;
    int16_t  small_exp;
    uint32_t small_mant;
    int16_t  exp_diff;
    bool     eff_sub;

    // MUL prep
    uint8_t  mul_sign;
    int16_t  mul_exp;
    uint32_t mul_mant_a;
    uint32_t mul_mant_b;
};

/**
 * @brief Stage 2 Pipeline Register: Alignment Shift & Multiplier Execution
 */
struct Stage2Reg {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode;

    bool     is_special;
    uint32_t special_result;
    uint8_t  special_status;

    // ADD/SUB: aligned operands extended with guard/round/sticky
    bool     large_sign;
    int16_t  large_exp;
    uint64_t mant_large_ext;
    uint64_t mant_small_ext;
    bool     eff_sub;

    // MUL: 48-bit product
    uint8_t  mul_sign;
    int16_t  mul_exp;
    uint64_t mul_prod;
};

/**
 * @brief Stage 3 Pipeline Register: Significand ALU & Leading-Zero Detection
 */
struct Stage3Reg {
    bool     valid;
    bool     eof;
    uint16_t tag;
    uint8_t  opcode;
    uint8_t  rounding_mode;

    bool     is_special;
    uint32_t special_result;
    uint8_t  special_status;

    // Normalized significand, exponent, and round/sticky bits
    uint8_t  res_sign;
    int16_t  res_exp;
    uint32_t res_mant;      ///< 24-bit significand
    uint8_t  guard;
    uint8_t  round;
    uint8_t  sticky;
    uint32_t low_residual;  ///< Discarded lower bits for EFT (ADD_HILO / MUL_HILO)
};

/**
 * @brief Stage 4 Pipeline Register: Final Output Packet
 */
struct Stage4Reg {
    bool          valid;
    FpResponse<4> resp;
};

} // namespace vph
