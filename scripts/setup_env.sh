#!/usr/bin/env bash
# Setup environment for oneAPI HLS and OFS Agilex N6001
if [ -f /ofs_install/scripts/env_n6001_latest.sh ]; then
    source /ofs_install/scripts/env_n6001_latest.sh
elif [ -f ~/Aggregation/load_ofs.sh ]; then
    source ~/Aggregation/load_ofs.sh
else
    echo "Warning: No OFS environment setup script found."
fi
