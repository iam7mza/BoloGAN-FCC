#!/usr/bin/env bash

# Run from BoloGAN-FCC/condor; must keep using relative path to run on different machines
cd $(dirname ${BASH_SOURCE[0]})/../training

# ENV SETUP; NOTE: To be ran on INFN machines; Also, LCG_105a_cuda is a temporary choice.. TODO: Update

echo "================================"
echo "[INFO] Setting up environment..."
echo "================================"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh
# installing quickstats b.c. it's missing from LCG_105a_cuda
pip install quickstats --break-system-packages

echo "================================"
echo "[INFO] Running task with arguments: $@"
echo "================================"


task=$1
config_file=$2  # e.g. ../input/run42.json

if [[ $task == "train" ]]; then
    python3 train.py $config_file
elif [[ $task == "evaluate" ]]; then
    python3 evaluate.py $config_file
else
    echo "[ERROR] Unknown task '$task'. Expected 'train' or 'evaluate'." >&2
    cd ../condor
    return 1
fi