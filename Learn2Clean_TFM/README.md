# Learn2Clean V3 (L2C2) — Prior-Aligned Data Cleaning for Tabular Foundation Models

Revision artifact. Learn2Clean reframes tabular data cleaning as selecting a sequence of cleaning
operators; **V1** ([Berti-Équille, WWW 2019](https://doi.org/10.1145/3308558.3313602)) used tabular
Q-learning, **V2** used deep RL agents (no publication), and **V3 (this code)** targets tabular
foundation models (TabPFN v2), extending the operator set and the reward function.

## Leak-free evaluation (key revision change)

Pipeline selection and final evaluation are separated by a **strict nested protocol**: an outer
20% test split is held out and never seen by any selection/reward computation; pipelines are
selected on an inner-validation split only; all reported accuracy/ECE come from the untouched outer
test. This is implemented in [`experiments/run_c2_tfm_reward_nested.py`](experiments/run_c2_tfm_reward_nested.py)
and reused by every experiment below. All headline results use 8 seeds with paired Wilcoxon tests.

## Setup

```bash
pip install numpy pandas scipy scikit-learn pyarrow pandera xgboost imbalanced-learn "tabpfn==2.2.1"
export PYTHONPATH=src:experiments
```

## Datasets (not bundled — fetched / linked)

Benchmark datasets are pulled from **OpenML** on first use by `src/learn2clean_v3/data/openml_loader.py`
(cached locally): hepatitis, heart_statlog, ionosphere, blood_transfusion, diabetes, credit_g,
kr_vs_kp, phoneme, adult, bank_marketing — see <https://www.openml.org>. The three naturally-dirty
SAGA-family datasets are obtained from their public sources: EEG Eye State (OpenML id 1471), Titanic
(OpenML / Kaggle), Animal Shelter Outcomes (Kaggle "Shelter Animal Outcomes"); place their CSVs under
`data/saga/` as referenced in `experiments/run_saga_comparison.py`.

## Experiments → paper

| Script | What it produces |
|---|---|
| `run_c2_tfm_reward_nested.py` + `merge_d1_seeds.py` | Leak-free TFM-reward vs RF-reward (8-seed gate) |
| `run_saga_richops.py` | Same contrast on a 13.4k-pipeline rich operator pool; F1/ECE selection arms |
| `run_weight_robustness.py` | Reward weight-simplex sweep; reward-collapse taxonomy |
| `run_h1_retention.py` | Retention-term test under outlier corruption |
| `run_saga_weighttune.py` | Drift-term ablation; SAGA weight tuning / oracle |
| `run_saga_comparison.py` | Comparison vs SAGA published numbers |
| `run_saga_rich_logreg.py` | Operator-richness vs SAGA with the model held constant (LogReg) |
| `run_corruption_sweep.py` + `merge_corruption_tuning.py` | Cleaning across all corruption types; TabPFN-tuning robustness |
| `run_force_label.py` | Forced label-cleaning under label noise (selectability) |
| `run_scalability.py` | Per-reward-term latency vs n |
| `investigate_tabpfn_tuning.py` | Null robustness across TabPFN configs |
| `run_automl_baselines.py` | AutoGluon / Auto-sklearn 2.0 end-to-end baselines |
| `run_prior_distance.py` + `analyze_prior_distance.py` | Four estimators of distance-to-the-TabPFN-prior |
| `run_divergence_pollution.py` | Engineered pollution targeting the reward mechanism |
| `run_noise_robust_reward.py`, `run_noise_robust_rich.py` + `merge_f1_selection.py` | R7 redesign: F1- / confidence- / margin- / prior-distance selection under label noise |

Each script is self-documented and writes its results under `outputs/`.
