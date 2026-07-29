# L2C2_TFM — Prior-Aligned Data Cleaning for Tabular Foundation Models

Learn2Clean reframes tabular data cleaning as selecting a sequence of cleaning
operators; **V1** ([Berti-Équille, WWW 2019](https://doi.org/10.1145/3308558.3313602)) used tabular
Q-learning, **V2** used deep RL agents (no publication), and **Learn2Clean-TFM (V3) (this code)** targets tabular
foundation models (TabPFN v2), extending the operator set and the reward function.

## Held-out protocol evaluation

Pipeline selection and final evaluation are separated by a **strict nested protocol**: an outer
20% test split is held out and never seen by any selection/reward computation; pipelines are
selected on an inner-validation split only; all reported accuracy/ECE come from the untouched outer
test. This is implemented in [`experiments/run_c2_tfm_reward_nested.py`](experiments/run_c2_tfm_reward_nested.py)
and reused by every experiment below. All headline results use 8 seeds with paired Wilcoxon tests.

## Setup

```bash
pip install numpy pandas scipy scikit-learn pyarrow pandera xgboost imbalanced-learn "tabpfn==2.2.1"
export PYTHONPATH=src:experiments      # required by every command below
```

All experiments run on **CPU** (no GPU needed): the models are small and a single TabPFN v2 call
on 512 rows takes ~0.3 s. Every script is self-documented (`--help`), takes `--seeds` and
`--output-dir`, and writes CSVs under `outputs/paper_ready/<name>/`. The headline results use the
**8-seed** set `42 1 2 3 4 5 6 7` with paired Wilcoxon tests.

## Datasets (not bundled — fetched / linked)

Benchmark datasets are pulled from **OpenML** on first use by `src/learn2clean_v3/data/openml_loader.py`
(cached locally): hepatitis, heart_statlog, ionosphere, blood_transfusion, diabetes, credit_g,
kr_vs_kp, phoneme, adult, bank_marketing — see <https://www.openml.org>. The three naturally-dirty
SAGA-family datasets are obtained from their public sources: EEG Eye State (OpenML id 1471), Titanic
(OpenML / Kaggle), Animal Shelter Outcomes (Kaggle "Shelter Animal Outcomes"); place their CSVs under
`data/saga/` as referenced in `experiments/run_saga_comparison.py`.

## Reproducing the experiments

Set `export PYTHONPATH=src:experiments` first (see Setup). Commands are grouped by paper section;
each writes to `outputs/paper_ready/<name>/`. `SEEDS="42 1 2 3 4 5 6 7"` is the 8-seed set.

**Step 0 — fetch and corrupt the data (run once):**
```bash
python experiments/run_load_datasets.py           # pull OpenML datasets into data/ (cached)
python experiments/run_inject_errors.py           # add the controlled MCAR/MAR/OUT/DUP corruptions
```

**Headline result — held-out-protocol TFM-reward vs RF-reward gate (§5.3, Table 3):**
```bash
python experiments/run_c2_tfm_reward_nested.py --seeds 42 1 2 3 4 5 6 7 \
       --output-dir outputs/paper_ready/d1_8seed
python experiments/merge_d1_seeds.py \
       --orig outputs/paper_ready/d1_8seed/results_per_seed.csv \
       --out  outputs/paper_ready/d1_8seed          # aggregate + paired Wilcoxon
```

**Contributions C1–C6:**
```bash
python experiments/run_c1_reward_benchmark.py                                  # C1 reward taxonomy (Fig 1)
python experiments/run_c2_factorial_nested.py  --seeds 42 1 2 3 4 5 6 7        # C2 2x2 reward x selection
python experiments/run_c3_8seed.py             --seeds 42 1 2 3 4 5 6 7 \
       --output-dir outputs/paper_ready/c3_8seed                               # C3 calibration (ECE)
python experiments/run_c4_8seed.py             --seeds 42 1 2 3 4 5 6 7 \
       --output-dir outputs/paper_ready/c4_8seed                               # C4 error-rate sensitivity
python experiments/run_c5_param_ablation.py    --mode both                     # C5 parameterized vs discrete (reward-level +0.0007)
python experiments/run_c5_taskeval_par.py      --seeds 42 1 2 3 4 5 6 7 --workers 4 \
       --output-dir outputs/paper_ready/c5_taskeval                            # C5 TASK-LEVEL delta (§5.9): dacc=-0.0002, p=0.77
python experiments/run_c6_transfer.py          --n-train 30000 --n-finetune 5000   # C6 transfer
```

**Baselines and diagnostics (§5.5–§5.11):**
```bash
python experiments/run_baselines_nested.py     --seeds 42 1 2 3 4 5 6 7        # non-RL cleaning baselines
python experiments/run_saga_richops.py         --seeds 42 1 2 3 4 5 6 7        # rich operator pool; F1/ECE arms
python experiments/run_saga_comparison.py      --seeds 42 1 2 3 4              # approx. SAGA-style search
python experiments/run_automl_baselines.py     --method autogluon --seeds 42 1 2   # AutoGluon end-to-end
python experiments/run_tabpfn_agsplit.py                                       # ours on the AutoML split
python experiments/run_weight_robustness.py                                    # reward-weight simplex sweep
python experiments/run_h1_retention.py         --seeds 42 1 2 3 4 5 6 7        # retention-term test
python experiments/run_force_label.py          --datasets EEG Titanic --seeds 42 1 2   # label-noise selectability
python experiments/run_prior_distance.py       --seeds 42 1 2 3 4 5 6 7        # 4 prior-distance estimators
python experiments/analyze_prior_distance.py                                   # summarise the above (raw correlations)
python experiments/analyze_prior_mediation.py                                  # §5.4/Table 4: prior-distance mediation (partials)
python experiments/run_scalability.py                                          # per-reward-term latency vs n
```

Tables are regenerated from the CSVs with `gen_table3_latex.py` / `gen_table3_full_latex.py` /
`fill_paper_tables.py`. Figure scripts (`plot_*.py`) are not bundled in this artifact.

## Script index

| Script | What it produces |
|---|---|
| `run_c2_tfm_reward_nested.py` + `merge_d1_seeds.py` | Held-out protocol TFM-reward vs RF-reward (8-seed gate) |
| `run_saga_richops.py` | Same contrast on a 13.4k-pipeline rich operator pool; F1/ECE selection arms |
| `run_weight_robustness.py` | Reward weight-simplex sweep; reward-collapse taxonomy |
| `run_h1_retention.py` | Retention-term test under outlier corruption |
| `run_saga_weighttune.py` | Drift-term ablation; SAGA weight tuning / oracle |
| `run_saga_comparison.py` | Comparison vs an approximate SAGA-style search (genetic top-$k$; no Multi-Armed Bandit stage) |
| `run_saga_rich_logreg.py` | Operator-richness vs SAGA with the model held constant (LogReg) |
| `run_corruption_sweep.py` + `merge_corruption_tuning.py` | Cleaning across all corruption types; TabPFN-tuning robustness |
| `run_force_label.py` | Forced label-cleaning under label noise (selectability) |
| `run_scalability.py` | Per-reward-term latency vs n |
| `investigate_tabpfn_tuning.py` | Null robustness across TabPFN configs |
| `run_automl_baselines.py` | AutoGluon / Auto-sklearn 2.0 end-to-end baselines |
| `run_tabpfn_agsplit.py` | Ours (clean+TabPFN, R7acc + R7F1) on the AutoML baselines' identical split |
| `run_prior_distance.py` + `analyze_prior_distance.py` | Four estimators of distance-to-the-TabPFN-prior |
| `analyze_prior_mediation.py` | §5.4 / Table 4: prior-distance mediation — raw + partial Spearman (distance-to-prior subsumes drift) |
| `run_c5_taskeval_par.py` (+ `run_c5_taskeval.py`) | C5 task-level: parameterized vs discrete on TabPFN test accuracy, 8-seed held-out (§5.9) |
| `run_divergence_pollution.py` | Engineered pollution targeting the reward mechanism |
| `run_noise_robust_reward.py`, `run_noise_robust_rich.py` + `merge_f1_selection.py` | R7 redesign: F1- / confidence- / margin- / prior-distance selection under label noise |

Each script is self-documented and writes its results under `outputs/`.
