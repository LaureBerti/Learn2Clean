"""Dataset loading and error injection utilities for Learn2Clean V3 experiments."""

from learn2clean_v3.data.openml_loader import (
    BENCHMARK_DATASETS,
    DatasetSpec,
    load_all_datasets,
    load_dataset,
)
from learn2clean_v3.data.error_injection import (
    ErrorProfile,
    apply_error_profile,
    generate_all_profiles,
    inject_duplicates,
    inject_missing_mar,
    inject_missing_mcar,
    inject_outliers,
)

__all__ = [
    # loader
    "BENCHMARK_DATASETS",
    "DatasetSpec",
    "load_dataset",
    "load_all_datasets",
    # injection
    "ErrorProfile",
    "inject_missing_mcar",
    "inject_missing_mar",
    "inject_outliers",
    "inject_duplicates",
    "apply_error_profile",
    "generate_all_profiles",
]
