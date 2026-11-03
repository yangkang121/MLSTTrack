#!/usr/bin/env bash

# ----------------------------------------------------------------------
# Log settings
# Save all terminal output (training + testing + errors) while still
# displaying it in the VS Code terminal.
# A new timestamped log file is created for each run.
# Example: ./logs/mlsttrack_20260809_233200.log
# ----------------------------------------------------------------------
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/mlsttrack_$(date +%Y%m%d_%H%M%S).log"

# Redirect both stdout and stderr of the whole script through tee.
# This avoids losing earlier terminal output when VS Code scrollback is full.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Full log will be saved to: ${LOG_FILE}"
echo

# python tracking/train.py --script mlsttrack --config MLSTTrack --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
# chmod +x train.sh
# ./train.sh

echo '----------------------------- Train MLSTTrack_256_lasher  -----------------------------'
python tracking/train.py --script mlsttrack --config MLSTTrack --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0
echo '----------------------------- Done MLSTTrack_256_lasher  -----------------------------'

for ((runid=20; runid>=20; runid--))
do
    echo ======================= Epoch $runid: Test LasHeR!!! =======================================
    python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mlsttrack --yaml_name MLSTTrack --dataset_name LasHeR --threads 4 --num_gpus 1 --epoch $runid --mode parallel && echo "LasHeR epoch $runid done"

    # echo ======================= Epoch $runid: Test RGBT234!!! ====================================
    # python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mlsttrack --yaml_name MLSTTrack --dataset_name RGBT234 --threads 2 --num_gpus 1 --epoch $runid --mode parallel && echo "RGBT234 epoch $runid done"

    # echo ======================= Epoch $runid: Test RGBT210!!! ====================================
    # python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mlsttrack --yaml_name MLSTTrack --dataset_name RGBT210 --threads 2 --num_gpus 1 --epoch $runid --mode parallel && echo "RGBT210 epoch $runid done"

    # echo ======================= Epoch $runid: Test GTOT!!! =======================================
    # python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mlsttrack --yaml_name MLSTTrack --dataset_name GTOT --threads 2 --num_gpus 1 --epoch $runid --mode parallel && echo "GTOT epoch $runid done"
done