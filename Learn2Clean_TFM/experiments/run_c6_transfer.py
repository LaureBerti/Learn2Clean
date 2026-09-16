
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import stable_baselines3
    TORCH_SB3_AVAILABLE = True
except ImportError:
    TORCH_SB3_AVAILABLE = False

if not TORCH_SB3_AVAILABLE:
    print(
        "ERROR: torch and/or stable_baselines3 are not installed.\n"
        "Install them with:\n"
        "  pip install torch\n"
        "  pip install stable-baselines3\n"
        "Then re-run this script.",
        file=sys.stderr,
    )
    sys.exit("Install torch and stable-baselines3 first")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from learn2clean_v3.actions import (
    DataFrameAction,
    ParameterizedImputer,
    ParameterizedOutlierCleaner,
    ParameterizedScaler,
)
from learn2clean_v3.data.error_injection import ErrorProfile, apply_error_profile
from learn2clean_v3.data.openml_loader import BENCHMARK_DATASETS, load_dataset
from learn2clean_v3.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
from learn2clean_v3.rewards import MultiObjectiveReward
from learn2clean_v3.transfer import PretrainedPolicyLoader

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class C6Config:
    train_datasets: List[str] = field(default_factory=lambda: [
        "hepatitis", "heart_statlog", "ionosphere",
        "blood_transfusion", "diabetes", "credit_g", "kr_vs_kp",
    ])
    held_out_datasets: List[str] = field(default_factory=lambda: [
        "phoneme", "adult", "bank_marketing",
    ])
    n_train: int = 20_000
    n_finetune: int = 5_000
    n_seeds: int = 1
    eval_interval: int = 100
    mcar_rate: float = 0.15
    checkpoint_dir: str = "outputs/c6_checkpoints"
    output_dir: str = "outputs/paper_ready/c6_transfer"
    seed: int = 42
    parity_threshold: float = 0.05
    parity_step: int = 2_000



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


def build_env(
    ds_name: str,
    mcar_rate: float,
    seed: int,
    eval_metric: str,
) -> SequentialCleaningEnvV3:
    X, y, spec = load_dataset(ds_name, use_cache=True)
    profile = ErrorProfile("mcar", rate=mcar_rate, seed=seed)
    X_dirty, y_dirty = apply_error_profile(X, y, profile)

    reward_fn = MultiObjectiveReward(
        weight_accuracy=0.5, weight_retention=0.3, weight_quality=0.2,
        drift_penalty_coeff=0.1, eval_model="random_forest",
        eval_metric=eval_metric, eval_cv_folds=1,
    )
    actions = build_actions()
    env = SequentialCleaningEnvV3(
        X=X_dirty,
        y=y_dirty,
        actions=actions,
        reward_fn=reward_fn,
        max_steps=5,
        allow_repeated_actions=False,
    )
    return env



class RewardRecorderCallback(BaseCallback):

    def __init__(self, eval_interval: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.eval_interval = eval_interval
        self.records: List[Tuple[int, float]] = []
        self._episode_rewards: List[float] = []
        self._current_episode_reward: float = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0.0])
        if hasattr(reward, "__len__"):
            self._current_episode_reward += float(reward[0])
        else:
            self._current_episode_reward += float(reward)

        dones = self.locals.get("dones", [False])
        done = dones[0] if hasattr(dones, "__len__") else bool(dones)
        if done:
            self._episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0

        if self.num_timesteps % self.eval_interval == 0:
            mean_r = (
                float(np.mean(self._episode_rewards[-10:]))
                if self._episode_rewards
                else 0.0
            )
            self.records.append((self.num_timesteps, mean_r))

        return True



def pretrain_on_dataset(
    ds_name: str,
    cfg: C6Config,
    ckpt_dir: Path,
    seed: int,
) -> Path:
    try:
        _, _, spec = load_dataset(ds_name, use_cache=True)
        eval_metric = spec.eval_metric
    except Exception as exc:
        logger.warning("Could not load spec for %s: %s", ds_name, exc)
        eval_metric = "f1"

    ckpt_path = ckpt_dir / f"{ds_name}_ppo.zip"
    if ckpt_path.exists():
        print(f"    Checkpoint already exists: {ckpt_path} — skipping pre-train")
        return ckpt_path

    print(f"    Pre-training on {ds_name!r} for {cfg.n_train} steps …")
    t0 = time.time()
    try:
        env = build_env(ds_name, cfg.mcar_rate, seed, eval_metric)
    except Exception as exc:
        raise RuntimeError(f"Failed to build env for {ds_name}: {exc}") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
        )
        model.learn(total_timesteps=cfg.n_train)

    model.save(str(ckpt_path))
    print(f"    Saved checkpoint: {ckpt_path}  ({time.time()-t0:.1f}s)")
    return ckpt_path



def train_policy(
    ds_name: str,
    cfg: C6Config,
    n_steps: int,
    seed: int,
    pretrained_ckpt: Optional[Path] = None,
) -> List[Tuple[int, float]]:
    try:
        _, _, spec = load_dataset(ds_name, use_cache=True)
        eval_metric = spec.eval_metric
    except Exception as exc:
        logger.warning("Could not load spec for %s: %s", ds_name, exc)
        eval_metric = "f1"

    env = build_env(ds_name, cfg.mcar_rate, seed, eval_metric)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pretrained_ckpt is not None and pretrained_ckpt.exists():
            loader = PretrainedPolicyLoader(pretrained_ckpt, freeze_layers=0)
            model = loader.load_into(
                env,
                algorithm_class=PPO,
                verbose=0,
                seed=seed,
                learning_rate=1e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
            )

        callback = RewardRecorderCallback(eval_interval=cfg.eval_interval)
        model.learn(total_timesteps=n_steps, callback=callback)

    return callback.records



def make_latex_table_c6(curves_df: pd.DataFrame, parity_step: int, threshold: float) -> str:
    at_step = curves_df[curves_df["step"] <= parity_step].groupby(
        ["dataset", "mode"]
    )["reward"].last().reset_index()

    pivot = at_step.pivot(index="dataset", columns="mode", values="reward")
    held_out = curves_df["dataset"].unique()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{C6 — Policy reward at step {parity_step:,} for fine-tuned "
        r"(pre-trained on 7 datasets) vs.\ scratch-trained policy on 3 held-out "
        rf"datasets. Within {int(threshold*100)}\% gap validates C6.}}",
        r"\label{tab:c6_transfer}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Fine-tune & Scratch & Gap (\%) \\",
        r"\midrule",
    ]

    n_parity = 0
    for ds in sorted(held_out):
        ft  = pivot.loc[ds, "finetune"] if ds in pivot.index and "finetune" in pivot.columns else float("nan")
        scr = pivot.loc[ds, "scratch"]  if ds in pivot.index and "scratch"  in pivot.columns else float("nan")

        ft_s  = f"{ft:.4f}"  if np.isfinite(ft)  else "---"
        scr_s = f"{scr:.4f}" if np.isfinite(scr) else "---"

        if np.isfinite(ft) and np.isfinite(scr) and abs(scr) > 1e-9:
            gap = (scr - ft) / abs(scr)
            gap_s = f"{gap*100:.1f}\\%"
            within = gap < threshold
            if within:
                n_parity += 1
                ft_s = r"\textbf{" + ft_s + r"}"
        else:
            gap_s = "---"

        ds_tex = ds.replace("_", r"\_")
        lines.append(f"  {ds_tex} & {ft_s} & {scr_s} & {gap_s} \\\\")

    lines += [
        r"\midrule",
        rf"  Within {int(threshold*100)}\% & \multicolumn{{3}}{{c}}{{{n_parity}/3}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)



def main(cfg: C6Config) -> None:
    ckpt_dir = Path(cfg.checkpoint_dir)
    out_dir  = Path(cfg.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"C6 Transfer Learning Experiment")
    print(f"  Training datasets  : {cfg.train_datasets}")
    print(f"  Held-out datasets  : {cfg.held_out_datasets}")
    print(f"  N_TRAIN            : {cfg.n_train}")
    print(f"  N_FINETUNE         : {cfg.n_finetune}")
    print(f"  N_SEEDS            : {cfg.n_seeds}")
    print(f"  Parity threshold   : {cfg.parity_threshold:.0%} at step {cfg.parity_step}")

    all_curves: List[Dict] = []
    t0_total = time.time()

    print(f"\n{'='*60}")
    print("Phase 1 — Pre-training")
    ckpt_paths: Dict[str, Path] = {}
    for ds_name in cfg.train_datasets:
        print(f"\n  {ds_name}")
        for s in range(cfg.n_seeds):
            seed = cfg.seed + s
            try:
                ckpt = pretrain_on_dataset(ds_name, cfg, ckpt_dir, seed=seed)
                ckpt_paths[ds_name] = ckpt
            except Exception as exc:
                print(f"    [ERROR] Pre-training failed on {ds_name}: {exc}")

    print(f"\n{'='*60}")
    print("Phase 2+3 — Fine-tuning and scratch training on held-out datasets")

    for ds_name in cfg.held_out_datasets:
        print(f"\n  Held-out dataset: {ds_name}")

        pretrained_ckpt: Optional[Path] = None
        if ckpt_paths:
            pretrained_ckpt = next(iter(ckpt_paths.values()))
            print(f"    Using pre-trained checkpoint: {pretrained_ckpt}")

        for s in range(cfg.n_seeds):
            seed = cfg.seed + s

            print(f"    [seed={seed}] Fine-tuning for {cfg.n_finetune} steps …")
            t0 = time.time()
            try:
                ft_records = train_policy(
                    ds_name, cfg, cfg.n_finetune, seed=seed,
                    pretrained_ckpt=pretrained_ckpt,
                )
                for step, reward in ft_records:
                    all_curves.append({
                        "dataset": ds_name,
                        "mode":    "finetune",
                        "seed":    seed,
                        "step":    step,
                        "reward":  reward,
                    })
                print(f"    Fine-tune done ({time.time()-t0:.1f}s)  "
                      f"final_reward={ft_records[-1][1]:.4f}" if ft_records else "    No records")
            except Exception as exc:
                print(f"    [ERROR] Fine-tuning failed: {exc}")

            print(f"    [seed={seed}] Training from scratch for {cfg.n_finetune} steps …")
            t0 = time.time()
            try:
                scratch_records = train_policy(
                    ds_name, cfg, cfg.n_finetune, seed=seed,
                    pretrained_ckpt=None,
                )
                for step, reward in scratch_records:
                    all_curves.append({
                        "dataset": ds_name,
                        "mode":    "scratch",
                        "seed":    seed,
                        "step":    step,
                        "reward":  reward,
                    })
                print(f"    Scratch done ({time.time()-t0:.1f}s)  "
                      f"final_reward={scratch_records[-1][1]:.4f}" if scratch_records else "    No records")
            except Exception as exc:
                print(f"    [ERROR] Scratch training failed: {exc}")

    if not all_curves:
        print("\nNo training curves to save.")
        return

    curves_df = pd.DataFrame(all_curves)
    curves_df.to_csv(out_dir / "training_curves.csv", index=False)

    agg_curves = curves_df.groupby(["dataset", "mode", "step"])["reward"].mean().reset_index()

    print(f"\n{'='*60}")
    print(f"C6 — Reward parity at step {cfg.parity_step}:")
    at_step = (
        agg_curves[agg_curves["step"] <= cfg.parity_step]
        .groupby(["dataset", "mode"])["reward"].last().reset_index()
    )
    n_within = 0
    total = len(cfg.held_out_datasets)
    for ds_name in cfg.held_out_datasets:
        ft_rows  = at_step[(at_step["dataset"] == ds_name) & (at_step["mode"] == "finetune")]
        scr_rows = at_step[(at_step["dataset"] == ds_name) & (at_step["mode"] == "scratch")]
        if ft_rows.empty or scr_rows.empty:
            print(f"  {ds_name:20s}  no data")
            continue
        ft_r  = float(ft_rows["reward"].values[0])
        scr_r = float(scr_rows["reward"].values[0])
        if abs(scr_r) > 1e-9:
            gap = (scr_r - ft_r) / abs(scr_r)
            within = gap < cfg.parity_threshold
            if within:
                n_within += 1
            print(f"  {ds_name:20s}  finetune={ft_r:.4f}  scratch={scr_r:.4f}  "
                  f"gap={gap*100:.1f}%  within_{int(cfg.parity_threshold*100)}%={within}")
        else:
            print(f"  {ds_name:20s}  finetune={ft_r:.4f}  scratch={scr_r:.4f}  gap=N/A")

    print(f"\n  Parity (≤{cfg.parity_threshold:.0%} gap): {n_within}/{total} held-out datasets")

    latex = make_latex_table_c6(agg_curves, cfg.parity_step, cfg.parity_threshold)
    (out_dir / "c6_transfer.tex").write_text(latex)

    print(f"\nResults saved to {out_dir}/")
    print(f"Total time: {time.time() - t0_total:.1f}s")



def write_hydra_config(cfg: C6Config, conf_dir: Path) -> None:
    import yaml

    conf_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {
        "n_train":           cfg.n_train,
        "n_finetune":        cfg.n_finetune,
        "n_seeds":           cfg.n_seeds,
        "eval_interval":     cfg.eval_interval,
        "mcar_rate":         cfg.mcar_rate,
        "checkpoint_dir":    cfg.checkpoint_dir,
        "output_dir":        cfg.output_dir,
        "seed":              cfg.seed,
        "parity_threshold":  cfg.parity_threshold,
        "parity_step":       cfg.parity_step,
        "train_datasets":    cfg.train_datasets,
        "held_out_datasets": cfg.held_out_datasets,
    }
    conf_path = conf_dir / "c6_transfer.yaml"
    with open(conf_path, "w") as fh:
        yaml.safe_dump(config_dict, fh, default_flow_style=False, sort_keys=True)
    print(f"Default config written to {conf_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C6 Transfer Learning experiment — requires torch + stable-baselines3"
    )
    parser.add_argument(
        "--n-train", type=int, default=20_000,
        help="PPO steps per training dataset (default: 20000)",
    )
    parser.add_argument(
        "--n-finetune", type=int, default=5_000,
        help="Fine-tune / scratch steps on held-out datasets (default: 5000)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=1,
        help="Number of random seeds (default: 1)",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=100,
        help="Record reward every N steps (default: 100)",
    )
    parser.add_argument(
        "--mcar-rate", type=float, default=0.15,
        help="MCAR injection rate for all datasets (default: 0.15)",
    )
    parser.add_argument(
        "--checkpoint-dir", default="outputs/c6_checkpoints",
        help="Directory for PPO checkpoints (default: outputs/c6_checkpoints)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output files (default: outputs/paper_ready/c6_transfer/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--parity-threshold", type=float, default=0.05,
        help="Gap threshold for 'within X%%' parity criterion (default: 0.05)",
    )
    parser.add_argument(
        "--parity-step", type=int, default=2_000,
        help="Fine-tune step at which to check parity (default: 2000)",
    )
    parser.add_argument(
        "--write-config", action="store_true",
        help="Write default config to conf/c6_transfer.yaml and exit",
    )
    args = parser.parse_args()

    cfg = C6Config(
        n_train=args.n_train,
        n_finetune=args.n_finetune,
        n_seeds=args.n_seeds,
        eval_interval=args.eval_interval,
        mcar_rate=args.mcar_rate,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir or "outputs/paper_ready/c6_transfer",
        seed=args.seed,
        parity_threshold=args.parity_threshold,
        parity_step=args.parity_step,
    )

    if args.write_config:
        conf_dir = Path(__file__).parents[1] / "conf"
        write_hydra_config(cfg, conf_dir)
        sys.exit(0)

    main(cfg)
