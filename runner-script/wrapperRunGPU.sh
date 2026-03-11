#!/bin/bash
#SBATCH --job-name=bologan_train
#SBATCH --partition=bare-metal-GPU
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

# ── Usage ────────────────────────────────────────────────────────────────────
# INPUT: <input_file> <config_name>
#
# Example:
#   sbatch wrapperRunGPU.sh ../input/dataset1/dataset_1_pions_1.hdf5 BNReLU_hpo27-M1
# ─────────────────────────────────────────────────────────────────────────────

INPUT=$1
CONFIG=$2

if [[ -z "$INPUT" || -z "$CONFIG" ]]; then
    echo "ERROR: Missing arguments."
    echo "Usage: sbatch wrapperRunGPU.sh <input_file> <config_name>"
    exit 1
fi

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Input      : $INPUT"
echo "Config     : $CONFIG"
echo "Started    : $(date)"
echo "=============================="

cd $SLURM_SUBMIT_DIR/../condor

# ── Training ─────────────────────────────────────────────────────────────────
t_train_start=$SECONDS
source wrapper.sh train $INPUT $CONFIG
t_train_end=$SECONDS
train_duration=$((t_train_end - t_train_start))

# ── Evaluation ───────────────────────────────────────────────────────────────
t_eval_start=$SECONDS
source wrapper.sh evaluate $INPUT $CONFIG
t_eval_end=$SECONDS
eval_duration=$((t_eval_end - t_eval_start))

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=============================="
echo "Finished       : $(date)"
printf "Training time  : %02dh %02dm %02ds\n" $((train_duration/3600)) $((train_duration%3600/60)) $((train_duration%60))
printf "Eval time      : %02dh %02dm %02ds\n" $((eval_duration/3600)) $((eval_duration%3600/60)) $((eval_duration%60))
printf "Total time     : %02dh %02dm %02ds\n" $(((train_duration+eval_duration)/3600)) $(((train_duration+eval_duration)%3600/60)) $(((train_duration+eval_duration)%60))
echo "=============================="