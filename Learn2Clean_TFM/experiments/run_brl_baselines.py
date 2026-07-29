"""
experiments/run_brl_baselines.py

B-RL Baselines — RL-trained policies evaluated with TabPFN
===========================================================
Computes B-RL-RF and B-RL-TFM rows for the main results table (Table 2).

Design
------
For each of the 10 benchmark datasets (MCAR 15%):

  B-RL-RF:
    Train PPO for N_STEPS using MultiObjectiveReward (RF evaluator).
    Apply trained policy greedily to the dirty dataset.
    Evaluate resulting cleaned dataset with TabPFN v2 → accuracy + ECE.

  B-RL-TFM:
    Start from the B-RL-RF checkpoint (warm start).
    Fine-tune for N_TFM_STEPS using TFMAwareReward (TabPFN evaluator).
    Apply fine-tuned policy greedily.
    Evaluate with TabPFN v2 → accuracy + ECE.

This avoids full TFMAwareReward training (which calls TabPFN at every step
= extremely slow) while still incorporating TFM feedback at fine-tune time.

Outputs
-------
  outputs/paper_ready/brl_baselines/
    results.csv         — (dataset, mode, tabpfn_acc, ece, steps, reward)
    brl_main.tex        — LaTeX rows for Table 2

Usage
-----
  conda activate l2c_torch
  cd Learn2Clean_TFM
  PYTHONPATH=src python experiments/run_brl_baselines.py
  PYTHONPATH=src python experiments/run_brl_baselines.py --datasets hepatitis ionosphere
  PYTHONPATH=src python experiments/run_brl_baselines.py --n-steps 3000 --n-tfm-steps 500
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
    import stable_baselines3  # noqa: F401
    TORCH_SB3_AVAILABLE = True
except ImportError:
    TORCH_SB3_AVAILABLE = False

if not TORCH_SB3_AVAILABLE:
    sys.exit("ERROR: Install torch + stable-baselines3 first (conda activate l2c_torch)")

try:
    import tabpfn as _tp  # noqa: F401
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

if not TABPFN_AVAILABLE:
    sys.exit("ERROR: Install tabpfn>=2.0 first (pip install tabpfn)")

from stable_baselines3 import PPO

from learn2clean_v3.actions import (
    DataFrameAction,
    ParameterizedImputer,
    ParameterizedOutlierCleaner,
    ParameterizedScaler,
)
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset
from learn2clean_v3.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
from learn2clean_v3.rewards import MultiObjectiveReward, TFMAwareReward

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MCAR_PROFILE = ErrorProfile("mcar", rate=0.15, seed=42)
OUT_DIR = Path(__file__).parents[1] / "outputs" / "paper_ready" / "brl_baselines"

DS_ORDER = [
    "hepatitis", "heart_statlog", "ionosphere", "blood_transfusion", "diabetes",
    "credit_g", "kr_vs_kp", "phoneme", "adult", "bank_marketing",
]
DS_LABELS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_actions() -> List[DataFrameAction]:
    return [
        ParameterizedImputer(strategy="mean"),
        ParameterizedImputer(strategy="median"),
        ParameterizedImputer(strategy="knn", n_neighbors=5),
        ParameterizedOutlierCleaner(method="iqr",    threshold=1.5),
        ParameterizedOutlierCleaner(method="zscore", threshold=3.0),
        ParameterizedScaler(method="minmax"),
        ParameterizedScaler(method="zscore"),
    ]


def train_ppo(
    env: SequentialCleaningEnvV3,
    n_steps: int,
    seed: int = 42,
    verbose: int = 0,
) -> PPO:
    """Train a PPO policy on the given environment."""
    model = PPO(
        "MlpPolicy", env,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        verbose=verbose,
        seed=seed,
    )
    model.learn(total_timesteps=n_steps)
    return model


def apply_policy_greedy(
    model: PPO,
    env: SequentialCleaningEnvV3,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply a trained PPO policy greedily and return the cleaned dataset."""
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return env.current_X.copy()


_NAN5 = (float("nan"),) * 5


def eval_tabpfn(
    X_clean: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
) -> Tuple[float, float, float, float, float]:
    """Evaluate cleaned dataset with TabPFN v2.
    Returns (accuracy, ECE, macro-F1, macro-precision, macro-recall)."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    from tabpfn import TabPFNClassifier

    numeric = X_clean.select_dtypes(include="number").fillna(0)
    if numeric.shape[1] == 0 or len(numeric) < 20:
        return _NAN5

    y_arr = np.asarray(y)
    le = LabelEncoder()
    try:
        y_enc = le.fit_transform(y_arr)
    except Exception:
        return _NAN5

    if len(np.unique(y_enc)) < 2:
        return _NAN5

    X_vals = numeric.values.astype(float)
    max_rows = 1024
    if len(X_vals) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_vals), max_rows, replace=False)
        X_vals, y_enc = X_vals[idx], y_enc[idx]

    test_size = float(np.clip(10.0 / len(X_vals), 0.2, 0.4))
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_vals, y_enc, test_size=test_size, random_state=seed, stratify=y_enc,
        )
    except ValueError:
        return _NAN5

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # random_state=seed: pin TabPFN's 8-estimator ensemble so seed-to-seed
            # variance is genuine (else the ensemble draws from the global torch RNG).
            clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True,
                                   random_state=seed)
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)
            pred  = clf.predict(X_te)

        acc = float(np.mean(pred == y_te))
        f1   = float(f1_score(y_te, pred, average="macro", zero_division=0))
        prec = float(precision_score(y_te, pred, average="macro", zero_division=0))
        rec  = float(recall_score(y_te, pred, average="macro", zero_division=0))
        conf    = proba.max(axis=1) if proba.ndim > 1 else proba.ravel()
        correct = (proba.argmax(axis=1) == y_te).astype(int) if proba.ndim > 1 else y_te.astype(int)
        bins = np.linspace(0, 1, 11)
        ece = sum(
            abs(conf[m].mean() - correct[m].mean()) * m.sum() / len(y_te)
            for i in range(10)
            if (m := (conf >= bins[i]) & (conf < bins[i + 1])).any()
        )
        return acc, float(ece), f1, prec, rec
    except Exception as exc:
        logger.debug("TabPFN eval failed: %s", exc)
        return _NAN5


def make_latex_rows(results_df: pd.DataFrame) -> str:
    """Generate LaTeX rows for Table 2 (B-RL-RF and B-RL-TFM accuracy/ECE)."""
    ds_to_label = dict(zip(DS_ORDER, DS_LABELS))
    lines = [
        "% B-RL baselines — accuracy rows",
    ]
    for mode, label in [("rf", "B-RL-RF"), ("tfm", "B-RL-TFM (ours)$^{\\dagger}$")]:
        subset = results_df[results_df["mode"] == mode].set_index("dataset")
        vals = [f"{subset.loc[ds, 'tabpfn_acc']:.4f}" if ds in subset.index else r"\todo{}"
                for ds in DS_ORDER]
        mean = results_df[results_df["mode"] == mode]["tabpfn_acc"].mean()
        mean_str = f"{mean:.4f}" if np.isfinite(mean) else r"\todo{}"
        lines.append(f"{label:<36} & {' & '.join(vals)} & {mean_str} \\\\")

    lines.append("% B-RL baselines — ECE rows")
    for mode, label in [("rf", "B-RL-RF"), ("tfm", "B-RL-TFM (ours)$^{\\dagger}$")]:
        subset = results_df[results_df["mode"] == mode].set_index("dataset")
        vals = [f"{subset.loc[ds, 'ece']:.4f}" if ds in subset.index else r"\todo{}"
                for ds in DS_ORDER]
        mean = results_df[results_df["mode"] == mode]["ece"].mean()
        mean_str = f"{mean:.4f}" if np.isfinite(mean) else r"\todo{}"
        lines.append(f"{label:<36} & {' & '.join(vals)} & {mean_str} \\\\")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    dataset_names: Optional[List[str]] = None,
    n_steps: int = 5_000,
    n_tfm_steps: int = 1_000,
    seed: int = 42,
) -> None:
    if dataset_names is None:
        dataset_names = DS_ORDER

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actions = build_actions()
    results: List[Dict] = []
    t0_total = time.time()

    for ds_name in dataset_names:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        t0 = time.time()

        try:
            X, y, spec = load_dataset(ds_name, use_cache=True)
        except Exception as exc:
            print(f"  [SKIP] Load failed: {exc}")
            continue

        X_dirty, y_dirty = apply_error_profile(X, y, MCAR_PROFILE)
        print(f"  Loaded: {len(X)} rows × {X.shape[1]} cols  "
              f"| MCAR 15% → missing={X_dirty.isna().mean().mean():.2%}")

        # ── B-RL-RF: train with RF reward ─────────────────────────────────
        print(f"  [B-RL-RF] Training PPO for {n_steps} steps with RF reward …", end=" ", flush=True)
        t_rf = time.time()
        rf_reward = MultiObjectiveReward(
            weight_accuracy=0.5, weight_retention=0.3, weight_quality=0.2,
            drift_penalty_coeff=0.1, eval_model="random_forest",
            eval_metric=spec.eval_metric, eval_cv_folds=1,
        )
        try:
            env_rf = SequentialCleaningEnvV3(
                X=X_dirty, y=y_dirty,
                actions=actions, reward_fn=rf_reward, max_steps=3,
            )
            model_rf = train_ppo(env_rf, n_steps=n_steps, seed=seed)
            X_rl_rf = apply_policy_greedy(model_rf, env_rf, seed=seed)
            acc_rf, ece_rf, f1_rf, prec_rf, rec_rf = eval_tabpfn(X_rl_rf, env_rf.current_y, seed=seed)
            print(f"acc={acc_rf:.4f}  F1={f1_rf:.4f}  ECE={ece_rf:.4f}  ({time.time()-t_rf:.0f}s)")
            results.append({
                "dataset": ds_name, "mode": "rf",
                "tabpfn_acc": acc_rf, "ece": ece_rf,
                "f1": f1_rf, "prec": prec_rf, "rec": rec_rf,
                "steps": n_steps, "train_time_s": round(time.time() - t_rf, 1),
            })
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({"dataset": ds_name, "mode": "rf",
                             "tabpfn_acc": float("nan"), "ece": float("nan"),
                             "f1": float("nan"), "prec": float("nan"), "rec": float("nan"),
                             "steps": n_steps, "train_time_s": float("nan")})
            model_rf = None

        # ── B-RL-TFM: warm-start from RF policy, fine-tune with TFM reward ─
        print(f"  [B-RL-TFM] Fine-tuning for {n_tfm_steps} steps with TFM reward …", end=" ", flush=True)
        t_tfm = time.time()
        tfm_reward = TFMAwareReward(
            weight_accuracy=0.50, weight_retention=0.35, weight_quality=0.15,
            drift_penalty_coeff=0.05, eval_model="tabpfn",
            eval_metric=spec.eval_metric,
        )
        try:
            env_tfm = SequentialCleaningEnvV3(
                X=X_dirty, y=y_dirty,
                actions=actions, reward_fn=tfm_reward, max_steps=3,
            )
            if model_rf is not None:
                # Warm start: transfer RF model's policy weights
                model_tfm = PPO(
                    "MlpPolicy", env_tfm,
                    n_steps=256, batch_size=64, n_epochs=4,
                    learning_rate=1e-4,  # smaller LR for fine-tuning
                    verbose=0, seed=seed,
                )
                # Copy policy parameters from RF model
                model_tfm.policy.load_state_dict(model_rf.policy.state_dict())
            else:
                model_tfm = train_ppo(env_tfm, n_steps=n_tfm_steps, seed=seed)
                n_tfm_steps_actual = 0  # already trained from scratch

            model_tfm.learn(total_timesteps=n_tfm_steps)
            X_rl_tfm = apply_policy_greedy(model_tfm, env_tfm, seed=seed)
            acc_tfm, ece_tfm, f1_tfm, prec_tfm, rec_tfm = eval_tabpfn(X_rl_tfm, env_tfm.current_y, seed=seed)
            print(f"acc={acc_tfm:.4f}  F1={f1_tfm:.4f}  ECE={ece_tfm:.4f}  ({time.time()-t_tfm:.0f}s)")
            results.append({
                "dataset": ds_name, "mode": "tfm",
                "tabpfn_acc": acc_tfm, "ece": ece_tfm,
                "f1": f1_tfm, "prec": prec_tfm, "rec": rec_tfm,
                "steps": n_steps + n_tfm_steps,
                "train_time_s": round(time.time() - t_tfm, 1),
            })
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({"dataset": ds_name, "mode": "tfm",
                             "tabpfn_acc": float("nan"), "ece": float("nan"),
                             "f1": float("nan"), "prec": float("nan"), "rec": float("nan"),
                             "steps": n_steps + n_tfm_steps,
                             "train_time_s": float("nan")})

        print(f"  Dataset total: {time.time()-t0:.0f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    if not results:
        print("\nNo results to save.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "results.csv", index=False)

    print(f"\n{'='*60}")
    print("B-RL Baselines Summary:")
    for mode in ["rf", "tfm"]:
        subset = results_df[results_df["mode"] == mode]
        mean_acc = subset["tabpfn_acc"].mean()
        mean_ece = subset["ece"].mean()
        print(f"  {mode:4}: mean acc={mean_acc:.4f}  mean ECE={mean_ece:.4f}  ({len(subset)} datasets)")

    latex = make_latex_rows(results_df)
    (OUT_DIR / "brl_main.tex").write_text(latex)

    print(f"\nResults saved to {OUT_DIR}/")
    print(f"Total time: {time.time()-t0_total:.0f}s")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B-RL-RF and B-RL-TFM baselines — PPO + TabPFN evaluation"
    )
    parser.add_argument("--datasets", nargs="*", default=None, metavar="NAME",
                        help=f"Subset of datasets. Available: {DS_ORDER}")
    parser.add_argument("--n-steps", type=int, default=5_000,
                        help="PPO training steps for B-RL-RF (default: 5000)")
    parser.add_argument("--n-tfm-steps", type=int, default=1_000,
                        help="Additional fine-tuning steps with TFM reward (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        dataset_names=args.datasets,
        n_steps=args.n_steps,
        n_tfm_steps=args.n_tfm_steps,
        seed=args.seed,
    )
