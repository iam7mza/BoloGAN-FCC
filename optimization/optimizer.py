import sys, os
import argparse
import optuna
import pandas as pd
import numpy as np
import time
import gc
import tensorflow as tf


TRAINPATH = "/afs/cern.ch/user/h/halhadda/BoloGAN-FCC/training"
sys.path.insert(0, TRAINPATH)
import train
import evaluate



PARTICLE = "pion"
PID = 11 if PARTICLE == "electron" else 211 # for now, only electrons and pions
CONFIG = {
    "dataset": {
        "input_file": f"../input/{PARTICLE}s/dataset_{PARTICLE}s.hdf5",
        "split_energy_position": "",
        "eta_slice": "00_05",
    },
    "preprocessing": {
        "preprocess": None,
        "mask": None,
        "add_noise": False,
        "label_scheme": "log_ratio",
    },
    "training": {
        "model": None,
        "max_iter": 200000,
        "loading": None,
        "debug": False, #NOTE: set to False for final run
        "config": None,
        "cache": True,
        "hp_config": None,          # always overridden per trial
        "config_type": "dict",
    },
    "meta": {
        "particle": PARTICLE,
        "PID": PID,
        "binning_file": f"../input/{PARTICLE}s/binning.xml",
    },
    "evaluate": {
        "preprocess": None,
        "checkpoint": False,
        "islice": None,
        "loading": None,
        "normalise": False,
        "save_h5": False,
        "istiming": False,
        "convert": False,
    },
}
# TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
TIMESTAMP = "20260523_121625"
BASE_OUTPUT = f"/eos/home-h/halhadda/runOptimization_{TIMESTAMP}/{PARTICLE}s"
CHECKPOINT_INTERVAL = 50000   # How often evaluate is called
#NOTE: change to 50k for final run
STUDY_DB_PATH = f"{BASE_OUTPUT}/optuna"
os.makedirs(STUDY_DB_PATH, exist_ok=True)
STUDY_DB = f"sqlite:////{BASE_OUTPUT}/optuna/bologan.db"


def compute_stability_metric_and_chi2(output_path: str) -> tuple:
    """
    Extract stability metric and chi2 from evaluate output at output_path.
    stability: slope of smoothed chi2 tail — positive = diverging, negative = converging.
    chi2: minimum chi2 seen so far across all checkpoints.
    Returns (float('inf'), float('inf')) on failure.
    """
    window = 10 # change to 50 for final run
    tail_frac = 0.95
    PATH = f"{output_path}/{PARTICLE}s_eta_00_05/evaluate/chi2.csv"

    try:
        df = pd.read_csv(PATH)
    except FileNotFoundError:
        return float('inf'), float('inf')

    x = df['ckpt'].values.astype(float)
    y = df['All'].values.astype(float)

    # Rolling median smoothing
    roll = pd.Series(y).rolling(window, center=True, min_periods=1)
    y_smooth = roll.median().values

    # Tail analysis
    tail_start = int(len(x) * (1 - tail_frac))
    x_tail = x[tail_start:]
    y_tail = y_smooth[tail_start:]

    # Guard: not enough tail points to fit a reliable slope
    if len(x_tail) < 3:
        return 0.0, float(y.min())  # inconclusive — don't prune on bad data

    # Convergence: linear slope on tail (positive = diverging, negative = converging)
    slope, _ = np.polyfit(x_tail, y_tail, 1)

    chi2_min = float(df['All'].min())

    return slope, chi2_min


# ─────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    hp_config = {
        # Optimizer
        "optimizer": "adam",
        "D_beta1": trial.suggest_float("D_beta1", 0.0, 0.9),
        "G_beta1": trial.suggest_float("G_beta1", 0.0, 0.9),
        "D_lr":    trial.suggest_float("D_lr", 1e-5, 1e-3, log=True),
        "G_lr":    trial.suggest_float("G_lr", 1e-5, 1e-3, log=True),

        # GAN training dynamics
        "batchsize": trial.suggest_categorical("batchsize", [128, 256, 512, 1024]),
        "dgratio":   trial.suggest_int("dgratio", 1, 10),
        "lam":       trial.suggest_float("lam", 1.0, 50.0, log=True),
        "latent_dim": trial.suggest_categorical("latent_dim", [50, 100, 200]),

        # Architecture — fix for now, expand search in phase 2
        "D_size": 1,
        "G_size": 1,
        "discriminatorLayers": [376, 376, 376] if PARTICLE == "electron" else [800, 400, 200],
        "generatorLayers": [100, 200, 400] if PARTICLE == "electron" else [200, 400, 800],
        "model": "BNswish" if PARTICLE == "electron" else "BNReLU",
        "dmodel": "spectral_norm",
        "use_bias": True,
    }

    output_path = f"{BASE_OUTPUT}/trial_{trial.number}"

    args = argparse.Namespace(
        **{k: v for section in CONFIG.values() for k, v in section.items()}
    )
    args.hp_config = hp_config
    args.output_path = output_path
    args.cache = True

    # ── Incremental train → evaluate → prune loop ──
    total_iter = args.max_iter
    chi2_final = float('inf')
    step = 0

    while step < total_iter:
        next_step = min(step + CHECKPOINT_INTERVAL, total_iter)
        args.max_iter = next_step

        train.main(args)

        args.train_path = output_path
        args.debug = True   # force single-process for evaluate (avoid CUDA fork)
        evaluate.args = args
        evaluate.main(args)
        args.debug = False  # restore for next training step

        stability, chi2 = compute_stability_metric_and_chi2(output_path)

        # Hard divergence gate: kill trial immediately if clearly blowing up.
        # This is checked before Optuna's pruner so we don't waste time on
        # obviously diverging runs regardless of how other trials are doing.
        SLOPE_TOLERANCE = 0.05  # tune after seeing a few runs
        if stability > SLOPE_TOLERANCE:
            tf.keras.backend.clear_session()
            gc.collect()
            raise optuna.TrialPruned()

        # Report chi2 to the pruner — same metric as the study objective so
        # pruning decisions are consistent with how trials are ranked.
        trial.report(chi2, step=next_step)

        # MedianPruner compares this trial's chi2 against the median of all
        # completed trials at the same step. Tolerates fluctuations better
        # than Hyperband since a single bad checkpoint won't kill the trial
        # unless it's consistently worse than the field.
        if trial.should_prune():
            tf.keras.backend.clear_session()
            gc.collect()
            raise optuna.TrialPruned()

        chi2_final = chi2
        step = next_step

    tf.keras.backend.clear_session()
    gc.collect()
    return chi2_final


# ─────────────────────────────────────────────
# Study
# ─────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(
        study_name=f"bologan_{PARTICLE}s",
        storage=STUDY_DB,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=5,   # random exploration before Bayesian kicks in — change to 10
            multivariate=True,    # models parameter correlations
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,   # don't prune at all until 5 trials have completed a step
            n_warmup_steps=2,     # don't prune a trial until it has reported at least 2 steps
            interval_steps=1,
        ),
    )

    study.optimize(objective, n_trials=10) #change to 20

    print("\n── Best trial ──")
    best = study.best_trial
    print(f"  chi2:  {best.value:.4f}")
    print(f"  params: {best.params}")