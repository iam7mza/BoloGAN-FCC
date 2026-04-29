#!/usr/bin/env bash

BOLOGAN_DIR=/afs/cern.ch/user/h/halhadda/BoloGAN-FCC
INPUT_JSON=${BOLOGAN_DIR}/$1

if [[ -z "$INPUT_JSON" ]]; then
    echo "[ERROR] Missing argument: <input_json>"
    exit 1
fi

echo "=============================="
echo "Cluster ID : $CONDOR_JOB_ID"
echo "Input JSON : $INPUT_JSON"
echo "Started    : $(date)"
echo "=============================="

cd ${BOLOGAN_DIR}/condor

# ── Training ──────────────────────────────────────────────────────────────────
t_train_start=$SECONDS
source wrapper.sh train $INPUT_JSON
t_train_end=$SECONDS
train_duration=$((t_train_end - t_train_start))

# ── Evaluation ────────────────────────────────────────────────────────────────
cd ${BOLOGAN_DIR}/condor #re-cd 
t_eval_start=$SECONDS
source wrapper.sh evaluate $INPUT_JSON
t_eval_end=$SECONDS
eval_duration=$((t_eval_end - t_eval_start))

# ── Summary ───────────────────────────────────────────────────────────────────
echo "=============================="
echo "Finished       : $(date)"
printf "Training time  : %02dh %02dm %02ds\n" $((train_duration/3600)) $((train_duration%3600/60)) $((train_duration%60))
printf "Eval time      : %02dh %02dm %02ds\n" $((eval_duration/3600)) $((eval_duration%3600/60)) $((eval_duration%60))
printf "Total time     : %02dh %02dm %02ds\n" $(((train_duration+eval_duration)/3600)) $(((train_duration+eval_duration)%3600/60)) $(((train_duration+eval_duration)%60))
echo "=============================="
