# Variable Precision Hardware (VPH)

A modular, bit-level floating-point hardware library implemented in **Intel oneAPI SYCL HLS**.

This project implements parameterizable, bit-level floating-point processing IPs across multiple precisions (FP64, FP32, FP16, BF16, FP8, and custom multi-limb formats). Rather than relying on black-box high-level synthesis compiler intrinsics or standard C++ floating-point operations, each IP is explicitly modeled at the bit level:
* **Explicit Bit-Level Datapaths**: Unpacking, magnitude comparison, alignment barrel shifting with sticky-bit reduction, significand arithmetic, leading-zero detection (CLZ priority trees), IEEE 754 Round to Nearest Even (RNE) rounding, and field repacking.
* **Explicit Feed-Forward Pipelining**: Stage-by-stage pipeline register structures (`Stage1Reg`, `Stage2Reg`, `Stage3Reg`, `Stage4Reg`) updated in cycle-accurate sequence inside an `[[intel::initiation_interval(1)]]` loop to guarantee **II = 1** and eliminate timing bottlenecks.
* **Neutral Byte-Array Pipe Interfaces**: Standardized request (`FpRequest<NBytes>`) and response (`FpResponse<NBytes>`) structs conveying little-endian byte arrays over Intel SYCL FPGA pipes (`sycl::ext::intel::pipe`).
* **Hardware Support for Error-Free Transformations (EFT)**: Native support for augmented dual-result operations (`ADD_HILO`, `MUL_HILO`) as explored in *Hardware Support for Float-Float & Shared-Exponent Floating-Point Architecture*, extracting exact residuals directly from internal datapath stages.

---

## Repository Structure

```
variable_precision_hardware/
├── external/
│   └── oneAPI-samples/              # Intel oneAPI FPGA utility headers submodule
├── include/
│   ├── fpu_types.hpp                # Opcodes (FpuOpcode), status flags (FpuStatus), packet structs
│   ├── bit_ops.hpp                  # Synthesizable CLZ priority trees, byte-to-uint packing/unpacking
│   ├── pipe_interface.hpp           # FPGA pipe template wrapper
│   ├── pipeline_regs.hpp            # Explicit pipeline stage register definitions (Stage1Reg .. Stage4Reg)
│   ├── common_kernels.hpp           # Templatized universal producer & consumer kernels
│   ├── fp64_ieee_core.hpp           # Bit-level IEEE-754 binary64 ALU (DSP partial products)
│   ├── fp32_ieee_core.hpp           # Bit-level IEEE-754 binary32 ALU
│   ├── fp16_ieee_core.hpp           # Bit-level IEEE-754 binary16 ALU
│   ├── bf16_core.hpp                # Bit-level bfloat16 ALU
│   ├── fp8_core.hpp                 # Bit-level FP8 (E4M3 and E5M2) ALU
│   └── fpx2_core.hpp                # Custom composite exponent & multi-limb format (from paper)
├── ip/
│   ├── fp32_cpu/                    # Latency-optimized IEEE FP32 CPU IP (4-stage, II=1)
│   ├── fp64_cpu/                    # Latency-optimized IEEE FP64 CPU IP (4-stage, II=1)
│   ├── fp16_cpu/                    # Latency-optimized IEEE FP16 CPU IP (4-stage, II=1)
│   ├── bf16_cpu/                    # Latency-optimized BF16 CPU IP (4-stage, II=1)
│   ├── fp8_cpu/                     # Latency-optimized FP8 CPU IP (4-stage, II=1)
│   ├── simd_packed/                 # Packed SIMD IP variants:
│   │   ├── simd_fp16x2_ip.hpp       # 2× FP16 SIMD in 32-bit container
│   │   ├── simd_fp32x2_ip.hpp       # 2× FP32 SIMD in 64-bit container
│   │   ├── simd_fp8x2_ip.hpp        # 2× FP8 SIMD in 16-bit container
│   │   └── simd_fp8x4_ip.hpp        # 4× FP8 SIMD in 32-bit container
│   ├── streaming_accel/             # 4-Lane Throughput/Streaming Accelerator IP
│   └── custom_fpx/                  # Custom Multi-Limb Composite FPX2 IP
├── host/
│   ├── host_verification.hpp        # Shared bit-exact verification test suite generator
│   ├── main_fp32_cpu.cpp            # Host runner for FP32 CPU IP
│   ├── main_fp64_cpu.cpp            # Host runner for FP64 CPU IP
│   ├── main_fp16_cpu.cpp            # Host runner for FP16 CPU IP
│   ├── main_bf16_cpu.cpp            # Host runner for BF16 CPU IP
│   ├── main_fp8_cpu.cpp             # Host runner for FP8 CPU IP
│   ├── main_simd_fp16x2.cpp         # Host runner for 2× FP16 SIMD IP
│   ├── main_simd_fp32x2.cpp         # Host runner for 2× FP32 SIMD IP
│   ├── main_simd_fp8x2.cpp          # Host runner for 2× FP8 SIMD IP
│   ├── main_simd_fp8x4.cpp          # Host runner for 4× FP8 SIMD IP
│   ├── main_streaming_accel.cpp     # Host runner for 4-lane streaming vector accelerator IP
│   └── main_custom_fpx2.cpp         # Host runner for Custom FPX2 IP
├── scripts/
│   └── setup_env.sh                 # Environment loader script
├── Mixed Float Hardware Support.md   # Architectural research document on float-float & shared-exponent FP
├── Makefile                         # Unified build system
└── README.md                        # Documentation and architecture specification
```

---

## Universal IP Interface Specification

Each IP connects via two typed FIFO pipes: an input pipe and an output pipe.

### Request Packet: `FpRequest<NBytes>`
```cpp
template <size_t NBytes>
struct alignas(4) FpRequest {
    uint8_t  op_a[NBytes];   // First operand (little-endian byte array)
    uint8_t  op_b[NBytes];   // Second operand (little-endian byte array)
    uint8_t  opcode;         // Operation code (FpuOpcode)
    uint8_t  rounding_mode;  // Rounding mode (0 = RNE)
    uint16_t tag;            // Transaction tag / sequence identifier
    bool     eof;            // End-of-stream delimiter
};
```

### Response Packet: `FpResponse<NBytes>`
```cpp
template <size_t NBytes>
struct alignas(4) FpResponse {
    uint8_t  result[NBytes];   // Primary rounded result byte array
    uint8_t  residual[NBytes]; // Exact residual / error term (for ADD_HILO / MUL_HILO)
    uint8_t  status;           // IEEE exception bitmask (FpuStatus)
    uint8_t  opcode;           // Echoed opcode
    uint16_t tag;              // Echoed transaction tag
    bool     eof;              // End-of-stream delimiter
};
```

### Supported Opcodes (`FpuOpcode`)
| Opcode | Mnemonic | Description |
| :--- | :--- | :--- |
| `0x00` | `NOP` | No-operation / passthrough |
| `0x01` | `ADD` | Standard addition: $fl(a + b)$ |
| `0x02` | `SUB` | Standard subtraction: $fl(a - b)$ |
| `0x03` | `MUL` | Standard multiplication: $fl(a \times b)$ |
| `0x04` | `ABS` | Unary absolute value: $\|a\|$ |
| `0x05` | `NEG` | Unary negation: $-a$ |
| `0x10` | `ADD_HILO` | Augmented addition returning rounded sum and exact error residual |
| `0x11` | `MUL_HILO` | Augmented multiplication returning rounded product and exact error residual |

---

## FP32 CPU Latency-Optimized Architecture

The FP32 CPU IP implements a 4-stage latency-optimized pipeline:

```
[InPipe]
   │
   ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Unpack, Decode & Alignment/Multiplier Prep    │
│  - Unpack sign, exponent, and restore hidden bit       │
│  - Classify special cases (Zero, Inf, NaN) & Unary Ops │
│  - Magnitude comparison (large vs small)               │
│  - Compute exp_diff = exp_large - exp_small            │
│  - MUL prep: exp_sum = exp_a + exp_b - 127             │
└────────────────────────────────────────────────────────┘
   │ Stage1Reg
   ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Alignment Shift & Multiplier Execution        │
│  - Barrel rightshift mant_small with guard/round/sticky│
│  - 24-bit × 24-bit unsigned multiplier → 48-bit product│
└────────────────────────────────────────────────────────┘
   │ Stage2Reg
   ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Significand ALU & Leading-Zero Detection      │
│  - Significand Add/Sub: mant_large ± mant_small_ext    │
│  - Branchless CLZ priority tree on subtraction diff    │
│  - Mantissa normalization (shift left/right)           │
│  - Extract 24-bit significand + guard, round, sticky   │
└────────────────────────────────────────────────────────┘
   │ Stage3Reg
   ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Rounding, Range Clamp & Repack                │
│  - IEEE 754 RNE (Round to Nearest Even) rounding       │
│  - Exponent overflow (→ Inf) and underflow (→ Zero)    │
│  - Format EFT residual term (for MUL_HILO / ADD_HILO)  │
│  - Pack uint32 → 4-byte array                          │
└────────────────────────────────────────────────────────┘
   │ Stage4Reg
   ▼
[OutPipe]
```

### Pipelining Loop
In `ip/fp32_cpu/fp32_cpu_ip.hpp`, the main execution loop is expressed as:
```cpp
[[intel::initiation_interval(1)]]
while (active) {
    if (r4.valid) {
        OutPipe::write(r4.resp);
        if (r4.resp.eof) active = false;
    }
    r4 = fp32_stage4(r3);
    r3 = fp32_stage3(r2);
    r2 = fp32_stage2(r1);

    bool has_data = false;
    FpRequest<4> req = InPipe::read(has_data);
    r1 = fp32_stage1(req, has_data);
}
```
This guarantees an **Initiation Interval of 1 (II = 1)** with zero loop-carried dependency stalls.

---

## Verification & Synthesis Results

### 1. Functional Verification (`make sim`)
* Tested across 10 standalone test suites covering:
  * Standard arithmetic (positive, negative, cross-sign).
  * Catastrophic cancellation testing leading-zero counts and multi-bit normalization shifts.
  * Shifter alignment sweeps (exponent differences across entire range).
  * Halfway rounding ties (Round-to-Nearest-Even).
  * Special values ($\pm 0.0$, $\pm \infty$, $\text{NaN}$, $0 \times \infty$, $\infty - \infty$).
  * Augmented error-free operations (`ADD_HILO`, `MUL_HILO`).
* **Summary Table**:
  | IP Variant | Target | Tested | Status |
  | :--- | :--- | :--- | :--- |
  | `fp32_cpu` | IEEE FP32 CPU IP | 53 / 53 passed | **100% Bit-Exact Match** |
  | `fp64_cpu` | IEEE FP64 CPU IP | 48 / 48 passed | **100% Bit-Exact Match** |
  | `fp16_cpu` | IEEE FP16 CPU IP | 35 / 35 passed | **100% Bit-Exact Match** |
  | `bf16_cpu` | BF16 CPU IP | 35 / 35 passed | **100% Bit-Exact Match** |
  | `fp8_cpu`  | FP8 CPU IP (E4M3 / E5M2) | 13 / 13 passed | **100% Bit-Exact Match** |
  | `simd_fp16x2` | Dual-Mode 2× FP16 Packed / 1× FP32 Scalar | 14 / 14 passed | **100% Bit-Exact Match** (Synthesized II=1) |
  | `simd_fp32x2` | Dual-Mode 2× FP32 Packed / 1× FP64 Scalar | 13 / 13 passed | **100% Bit-Exact Match** (Synthesized II=1) |
  | `simd_fp8x2`  | Dual-Mode 2× FP8 Packed / 1× FP16 Scalar | 11 / 11 passed | **100% Bit-Exact Match** |
  | `simd_fp8x4`  | Dual-Mode 4× FP8 Packed / 1× FP32 Scalar | 11 / 11 passed | **100% Bit-Exact Match** |
  | `streaming_accel` | 4-Lane Streaming Vector Accel | 5 / 5 passed | **100% Bit-Exact Match** |
  | `custom_fpx2` | Multi-Limb Composite FPX2 | 6 / 6 passed | **100% Bit-Exact Match** |

### 2. Early Hardware Synthesis (`make link`)
Target: **Intel Agilex OFS N6001** (`ofs_n6001`), Clock: **650 MHz**:
* **Loop Initiation Interval (II)**: **~1** across all kernels (`IpTask`, `ProducerTask`, `ConsumerTask`).
* **$f_{\text{MAX}}$ Bottlenecks**: None.
* **Compiler Warnings**: 0.

---

## Build Instructions

Ensure the Intel oneAPI environment is loaded (or the Makefile will automatically source `/ofs_install/scripts/env_n6001_latest.sh`):

```bash
# 1. Run all FPGA Emulator testbench verifications
make sim

# 2. Run specific emulator testbench
make sim_fp32
make sim_fp64
make sim_fp16
make sim_bf16
make sim_fp8
make sim_simd_fp16x2
make sim_simd_fp32x2
make sim_simd_fp8x2
make sim_simd_fp8x4
make sim_streaming_accel
make sim_fpx2

# 3. Run Early Synthesis Link (checks II, timing, and resource utilization)
make link_fp32
make link_fp64
make link_fp16
make link_bf16
make link_fp8
make link_simd_fp16x2

# 4. Clean build outputs
make clean
```
