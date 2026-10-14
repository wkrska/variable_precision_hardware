#!/usr/bin/env bash
# ==============================================================================
# Script: build_all_hw.sh
# Purpose: Build all or specific Variable Precision Hardware IPs for FPGA target
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Setup OFS and oneAPI environment if not already loaded
if ! command -v icpx &> /dev/null; then
    if [ -f /ofs_install/scripts/env_n6001_latest.sh ]; then
        echo "[INFO] Sourcing /ofs_install/scripts/env_n6001_latest.sh"
        source /ofs_install/scripts/env_n6001_latest.sh
    elif [ -f ~/Aggregation/load_ofs.sh ]; then
        echo "[INFO] Sourcing ~/Aggregation/load_ofs.sh"
        source ~/Aggregation/load_ofs.sh
    else
        echo "[ERROR] Cannot find oneAPI / OFS setup script."
        exit 1
    fi
fi

MODE="${1:-link}" # "link" for early synthesis check, "fpga" for full Quartus bitstream

TARGETS=(
    fp32
    fp64
    fp16
    bf16
    fp8
    simd_fp16x2
    simd_fp32x2
    simd_fp8x2
    simd_fp8x4
    streaming_accel
    fpx2
)

mkdir -p build logs

echo "=============================================================="
echo " Starting FPGA Hardware Build Suite"
echo " Mode   : ${MODE} (link = Early II & resource check, fpga = Full bitstream)"
echo " Board  : ofs_n6001 (Intel Agilex N6001)"
echo " Clock  : 650 MHz"
echo " Date   : $(date)"
echo "=============================================================="

TOTAL_START=$(date +%s)

for TARGET in "${TARGETS[@]}"; do
    CMD="make ${MODE}_${TARGET}"
    LOG_FILE="logs/${MODE}_${TARGET}.log"
    echo ""
    echo "--------------------------------------------------------------"
    echo " [$(date +'%H:%M:%S')] Starting: ${CMD}"
    echo " Logging to: ${LOG_FILE}"
    echo "--------------------------------------------------------------"
    
    START=$(date +%s)
    if ${CMD} 2>&1 | tee "${LOG_FILE}"; then
        ELAPSED=$(( $(date +%s) - START ))
        echo " [SUCCESS] ${TARGET} finished in ${ELAPSED} seconds."
    else
        ELAPSED=$(( $(date +%s) - START ))
        echo " [FAILURE] ${TARGET} failed after ${ELAPSED} seconds."
        exit 1
    fi
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "=============================================================="
echo " All ${#TARGETS[@]} Hardware Builds Completed Successfully!"
echo " Total Time: ${TOTAL_ELAPSED} seconds"
echo "=============================================================="
