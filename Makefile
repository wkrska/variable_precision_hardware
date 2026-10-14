SHELL := /bin/bash

# Target Board and Clock Settings
TARGET        ?= ofs_n6001
BOARD_PACKAGE ?= $(if $(OFS_BSP_BASE),$(OFS_BSP_BASE),/ofs_install/ofs/n6001/2023.3/oneapi-asp/n6001)
MHZ           ?= 650MHz
SEED          ?= 0

# Include Paths
ONEAPI_SRC    = ./external/oneAPI-samples/DirectProgramming/C++SYCL_FPGA/include/
INCLUDES      = -I ./include -I ./ip -I ./host -I $(ONEAPI_SRC) -I .

# Compiler Flags
CXX           = icpx
SIM_FLAGS     = -DFPGA_EMULATOR=1 -fsycl -fintelfpga -Wall -Wextra -O2 -w $(INCLUDES)
LINK_FLAGS    = -DFPGA_HARDWARE=1 -fsycl -fintelfpga -Xshardware -Xsv -Xsclock=$(MHZ) -Xsseed=$(SEED) \
                -Xsboard-package=$(BOARD_PACKAGE) -Xstarget=$(TARGET) -DIS_BSP -fsycl-link=early -O3 -w $(INCLUDES)
HW_FLAGS      = -DFPGA_HARDWARE=1 -fsycl -fintelfpga -Xshardware -Xsv -Xsclock=$(MHZ) -Xsseed=$(SEED) \
                -Xsboard-package=$(BOARD_PACKAGE) -Xstarget=$(TARGET) -DIS_BSP -O3 -w $(INCLUDES)

# Environment check / wrapper
ENV_LOAD      = if ! command -v icpx &> /dev/null; then source /ofs_install/scripts/env_n6001_latest.sh; fi;

# Build Output Directories
BUILD_DIR     = build

.PHONY: all sim sim_fp32 sim_fp64 sim_fp16 sim_bf16 sim_fp8 \
        sim_simd_fp16x2 sim_simd_fp32x2 sim_simd_fp8x2 sim_simd_fp8x4 \
        sim_streaming_accel sim_fpx2 \
        link link_fp32 link_fp64 link_fp16 link_bf16 link_fp8 \
        link_simd_fp16x2 link_simd_fp32x2 link_simd_fp8x2 link_simd_fp8x4 \
        link_streaming_accel link_fpx2 \
        fpga fpga_fp32 fpga_fp64 fpga_fp16 fpga_bf16 fpga_fp8 \
        fpga_simd_fp16x2 fpga_simd_fp32x2 fpga_simd_fp8x2 fpga_simd_fp8x4 \
        fpga_streaming_accel fpga_fpx2 clean help

all: sim

help:
	@echo "Variable Precision Hardware - FPGA oneAPI HLS Build System"
	@echo "Targets:"
	@echo "  make sim                 : Build and run all FPGA emulator verifications"
	@echo "  make link                : Run early FPGA synthesis link on all core IPs (II & resources check)"
	@echo "  make fpga                : Run full FPGA hardware bitstream synthesis (Quartus place & route)"
	@echo "  make fpga_<target>       : Run full bitstream compile for a single IP (e.g. make fpga_fp32)"
	@echo "  make clean               : Remove build artifacts"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)


# -------------------------------------------------------------
# Emulation Targets: Compile and Execute Correctness Tests
# -------------------------------------------------------------
sim: sim_fp32 sim_fp64 sim_fp16 sim_bf16 sim_fp8 \
     sim_simd_fp16x2 sim_simd_fp32x2 sim_simd_fp8x2 sim_simd_fp8x4 \
     sim_streaming_accel sim_fpx2

sim_fp32: $(BUILD_DIR)
	@echo "=== Compiling FP32 CPU IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_fp32_cpu.cpp -o $(BUILD_DIR)/fp32_cpu_sim
	@echo "=== Running FP32 CPU IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/fp32_cpu_sim

sim_fp64: $(BUILD_DIR)
	@echo "=== Compiling FP64 CPU IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_fp64_cpu.cpp -o $(BUILD_DIR)/fp64_cpu_sim
	@echo "=== Running FP64 CPU IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/fp64_cpu_sim

sim_fp16: $(BUILD_DIR)
	@echo "=== Compiling FP16 CPU IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_fp16_cpu.cpp -o $(BUILD_DIR)/fp16_cpu_sim
	@echo "=== Running FP16 CPU IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/fp16_cpu_sim

sim_bf16: $(BUILD_DIR)
	@echo "=== Compiling BF16 CPU IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_bf16_cpu.cpp -o $(BUILD_DIR)/bf16_cpu_sim
	@echo "=== Running BF16 CPU IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/bf16_cpu_sim

sim_fp8: $(BUILD_DIR)
	@echo "=== Compiling FP8 CPU IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_fp8_cpu.cpp -o $(BUILD_DIR)/fp8_cpu_sim
	@echo "=== Running FP8 CPU IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/fp8_cpu_sim

sim_simd_fp16x2: $(BUILD_DIR)
	@echo "=== Compiling 2x FP16 Packed SIMD IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_simd_fp16x2.cpp -o $(BUILD_DIR)/simd_fp16x2_sim
	@echo "=== Running 2x FP16 Packed SIMD IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/simd_fp16x2_sim

sim_simd_fp32x2: $(BUILD_DIR)
	@echo "=== Compiling 2x FP32 Packed SIMD IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_simd_fp32x2.cpp -o $(BUILD_DIR)/simd_fp32x2_sim
	@echo "=== Running 2x FP32 Packed SIMD IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/simd_fp32x2_sim

sim_simd_fp8x2: $(BUILD_DIR)
	@echo "=== Compiling 2x FP8 Packed SIMD IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_simd_fp8x2.cpp -o $(BUILD_DIR)/simd_fp8x2_sim
	@echo "=== Running 2x FP8 Packed SIMD IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/simd_fp8x2_sim

sim_simd_fp8x4: $(BUILD_DIR)
	@echo "=== Compiling 4x FP8 Packed SIMD IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_simd_fp8x4.cpp -o $(BUILD_DIR)/simd_fp8x4_sim
	@echo "=== Running 4x FP8 Packed SIMD IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/simd_fp8x4_sim

sim_streaming_accel: $(BUILD_DIR)
	@echo "=== Compiling 4-Lane Streaming Vector Accel IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_streaming_accel.cpp -o $(BUILD_DIR)/streaming_accel_sim
	@echo "=== Running 4-Lane Streaming Vector Accel IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/streaming_accel_sim

sim_fpx2: $(BUILD_DIR)
	@echo "=== Compiling Custom FPX2 Multi-Limb IP for FPGA Emulator ==="
	$(ENV_LOAD) $(CXX) $(SIM_FLAGS) host/main_custom_fpx2.cpp -o $(BUILD_DIR)/fpx2_sim
	@echo "=== Running Custom FPX2 Multi-Limb IP Emulator Verification ==="
	env DPCPP_CPU_NUM_CUS=255 $(BUILD_DIR)/fpx2_sim

# -------------------------------------------------------------
# Early Synthesis Link Targets: Verify HLS Pipeline & II
# -------------------------------------------------------------
link: link_fp32 link_fp64 link_fp16 link_bf16 link_fp8 link_simd_fp16x2 link_simd_fp32x2 link_streaming_accel link_fpx2

link_fp32: $(BUILD_DIR)
	@echo "=== Synthesizing FP32 CPU IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_fp32_cpu.cpp -o $(BUILD_DIR)/fp32_cpu_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/fp32_cpu_link.prj/ ==="

link_fp64: $(BUILD_DIR)
	@echo "=== Synthesizing FP64 CPU IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_fp64_cpu.cpp -o $(BUILD_DIR)/fp64_cpu_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/fp64_cpu_link.prj/ ==="

link_fp16: $(BUILD_DIR)
	@echo "=== Synthesizing FP16 CPU IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_fp16_cpu.cpp -o $(BUILD_DIR)/fp16_cpu_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/fp16_cpu_link.prj/ ==="

link_bf16: $(BUILD_DIR)
	@echo "=== Synthesizing BF16 CPU IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_bf16_cpu.cpp -o $(BUILD_DIR)/bf16_cpu_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/bf16_cpu_link.prj/ ==="

link_fp8: $(BUILD_DIR)
	@echo "=== Synthesizing FP8 CPU IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_fp8_cpu.cpp -o $(BUILD_DIR)/fp8_cpu_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/fp8_cpu_link.prj/ ==="

link_simd_fp16x2: $(BUILD_DIR)
	@echo "=== Synthesizing 2x FP16 Packed SIMD IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_simd_fp16x2.cpp -o $(BUILD_DIR)/simd_fp16x2_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/simd_fp16x2_link.prj/ ==="

link_simd_fp32x2: $(BUILD_DIR)
	@echo "=== Synthesizing 2x FP32 Packed SIMD IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_simd_fp32x2.cpp -o $(BUILD_DIR)/simd_fp32x2_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/simd_fp32x2_link.prj/ ==="

link_simd_fp8x2: $(BUILD_DIR)
	@echo "=== Synthesizing 2x FP8 Packed SIMD IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_simd_fp8x2.cpp -o $(BUILD_DIR)/simd_fp8x2_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/simd_fp8x2_link.prj/ ==="

link_simd_fp8x4: $(BUILD_DIR)
	@echo "=== Synthesizing 4x FP8 Packed SIMD IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_simd_fp8x4.cpp -o $(BUILD_DIR)/simd_fp8x4_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/simd_fp8x4_link.prj/ ==="

link_streaming_accel: $(BUILD_DIR)
	@echo "=== Synthesizing 4-Lane Streaming Vector Accel IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_streaming_accel.cpp -o $(BUILD_DIR)/streaming_accel_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/streaming_accel_link.prj/ ==="

link_fpx2: $(BUILD_DIR)
	@echo "=== Synthesizing Custom FPX2 Multi-Limb IP (Early Link for II & Resource Check) ==="
	$(ENV_LOAD) $(CXX) $(LINK_FLAGS) host/main_custom_fpx2.cpp -o $(BUILD_DIR)/fpx2_link.a
	@echo "=== Early Link Complete. Synthesis Reports in $(BUILD_DIR)/fpx2_link.prj/ ==="

# -------------------------------------------------------------
# Full Hardware Synthesis Targets (Quartus Place & Route -> .fpga Bitstream)
# -------------------------------------------------------------
fpga: fpga_fp32 fpga_fp64 fpga_fp16 fpga_bf16 fpga_fp8 \
      fpga_simd_fp16x2 fpga_simd_fp32x2 fpga_simd_fp8x2 fpga_simd_fp8x4 \
      fpga_streaming_accel fpga_fpx2

fpga_fp32: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: FP32 CPU IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_fp32_cpu.cpp -o $(BUILD_DIR)/fp32_cpu.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/fp32_cpu.fpga ==="

fpga_fp64: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: FP64 CPU IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_fp64_cpu.cpp -o $(BUILD_DIR)/fp64_cpu.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/fp64_cpu.fpga ==="

fpga_fp16: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: FP16 CPU IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_fp16_cpu.cpp -o $(BUILD_DIR)/fp16_cpu.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/fp16_cpu.fpga ==="

fpga_bf16: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: BF16 CPU IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_bf16_cpu.cpp -o $(BUILD_DIR)/bf16_cpu.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/bf16_cpu.fpga ==="

fpga_fp8: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: FP8 CPU IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_fp8_cpu.cpp -o $(BUILD_DIR)/fp8_cpu.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/fp8_cpu.fpga ==="

fpga_simd_fp16x2: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: 2x FP16 Packed / 1x FP32 Scalar SIMD IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_simd_fp16x2.cpp -o $(BUILD_DIR)/simd_fp16x2.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/simd_fp16x2.fpga ==="

fpga_simd_fp32x2: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: 2x FP32 Packed / 1x FP64 Scalar SIMD IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_simd_fp32x2.cpp -o $(BUILD_DIR)/simd_fp32x2.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/simd_fp32x2.fpga ==="

fpga_simd_fp8x2: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: 2x FP8 Packed / 1x FP16 Scalar SIMD IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_simd_fp8x2.cpp -o $(BUILD_DIR)/simd_fp8x2.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/simd_fp8x2.fpga ==="

fpga_simd_fp8x4: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: 4x FP8 Packed / 1x FP32 Scalar SIMD IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_simd_fp8x4.cpp -o $(BUILD_DIR)/simd_fp8x4.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/simd_fp8x4.fpga ==="

fpga_streaming_accel: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: 4-Lane Streaming Vector Accel IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_streaming_accel.cpp -o $(BUILD_DIR)/streaming_accel.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/streaming_accel.fpga ==="

fpga_fpx2: $(BUILD_DIR)
	@echo "=== Full Hardware Synthesis: Custom FPX2 Multi-Limb IP ==="
	$(ENV_LOAD) $(CXX) $(HW_FLAGS) host/main_custom_fpx2.cpp -o $(BUILD_DIR)/fpx2.fpga
	@echo "=== Full Bitstream Generated: $(BUILD_DIR)/fpx2.fpga ==="

clean:
	rm -rf $(BUILD_DIR)

