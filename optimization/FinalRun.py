# final_run.py
"""
Load the best hyperparameters from the completed Optuna study and run a
single 1 000 000-iteration final training with evaluate at the end.
"""
import sys, os
import argparse
import optuna
import time
import gc
import tensorflow as tf

TRAINPATH = "/afs/cern.ch/user/h/halhadda/BoloGAN-FCC/training"
sys.path.insert(0, TRAINPATH)
import train
import evaluate

# ── Mirror the study identity from the optimization script ──────────────────
PARTICLE   = "electron"   # "electron" or "pion"
PID        = 11 if PARTICLE == "electron" else 211
TIMESTAMP  = "20260523_010430"           # electron run # same timestamp as the study
# TIMESTAMP = "20260523_121625"           # pion run # same timestamp as the study
BASE_OUTPUT = f"/eos/home-h/halhadda/runOptimization_{TIMESTAMP}/{PARTICLE}s"
STUDY_DB   = f"sqlite:////{BASE_OUTPUT}/optuna/bologan.db"
STUDY_NAME = f"bologan_{PARTICLE}s"

FINAL_ITER    = 1_000_000
FINAL_OUTPUT  = f"{BASE_OUTPUT}/final_run"
os.makedirs(FINAL_OUTPUT, exist_ok=True)


def build_hp_config_from_best(best_params: dict) -> dict:
    """
    Reconstruct the full hp_config dict from the params Optuna stores.
    Fixed architecture values are filled in exactly as in the objective.
    """
    return {
        # Optimizer
        "optimizer":  "adam",
        "D_beta1":    best_params["D_beta1"],
        "G_beta1":    best_params["G_beta1"],
        "D_lr":       best_params["D_lr"],
        "G_lr":       best_params["G_lr"],
        # GAN dynamics
        "batchsize":  best_params["batchsize"],
        "dgratio":    best_params["dgratio"],
        "lam":        best_params["lam"],
        "latent_dim": best_params["latent_dim"],
        # Fixed architecture (phase-1 search)
        "D_size":     1,
        "G_size":     1,
        "discriminatorLayers": [376, 376, 376]     if PARTICLE == "electron" else [800, 400, 200],
        "generatorLayers":     [100, 200, 400]     if PARTICLE == "electron" else [200, 400, 800],
        "model":               "BNswish"           if PARTICLE == "electron" else "BNReLU",
        "dmodel":              "spectral_norm",
        "use_bias":            True,
    }


CONFIG = {
    "dataset": {
        "input_file": f"../input/{PARTICLE}s/dataset_{PARTICLE}s.hdf5",
        "split_energy_position": "",
        "eta_slice": "00_05",
    },
    "preprocessing": {
        "preprocess": None,
        "mask":       None,
        "add_noise":  False,
        "label_scheme": "log_ratio",
    },
    "training": {
        "model":       None,
        "max_iter":    FINAL_ITER,
        "loading":     None,
        "debug":       False,
        "config":      None,
        "cache":       True,
        "hp_config":   None,      # filled below
        "config_type": "dict",
    },
    "meta": {
        "particle":     PARTICLE,
        "PID":          PID,
        "binning_file": f"../input/{PARTICLE}s/binning.xml",
    },
    "evaluate": {
        "preprocess":  None,
        "checkpoint":  False,
        "islice":      None,
        "loading":     None,
        "normalise":   False,
        "save_h5":     False,
        "istiming":    False,
        "convert":     False,
    },
}


if __name__ == "__main__":
    # ── 1. Load study and extract best params ──────────────────────────────
    study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB)

    best = study.best_trial
    print(f"\n── Best trial from optimization ──")
    print(f"  trial number : {best.number}")
    print(f"  chi2         : {best.value:.4f}")
    print(f"  params       :")
    for k, v in best.params.items():
        print(f"    {k:20s} = {v}")

    hp_config = build_hp_config_from_best(best.params)

    # ── 2. Build args ──────────────────────────────────────────────────────
    args = argparse.Namespace(
        **{k: v for section in CONFIG.values() for k, v in section.items()}
    )
    args.hp_config    = hp_config
    args.output_path  = FINAL_OUTPUT
    args.max_iter     = FINAL_ITER
    args.cache        = True

    # ── 3. Train ───────────────────────────────────────────────────────────
    print(f"\n── Starting final run: {FINAL_ITER:,} iterations → {FINAL_OUTPUT} ──")
    train.main(args)

    # ── 4. Evaluate ────────────────────────────────────────────────────────
    print("\n── Running final evaluation ──")
    args.train_path = FINAL_OUTPUT
    args.debug      = True   # single-process evaluate
    evaluate.args   = args
    evaluate.main(args)
    args.debug      = False

    tf.keras.backend.clear_session()
    gc.collect()
    print("\n── Final run complete ──")