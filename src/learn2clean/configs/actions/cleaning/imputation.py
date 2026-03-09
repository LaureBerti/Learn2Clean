from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from ..base import ActionConfig


# --- Iterative / Advanced Imputers ---


@dataclass
class EMImputerConfig(ActionConfig):
    """
    Configuration for Expectation-Maximization Imputation.
    """

    name: str = "EMImputer"
    _target_: str = "learn2clean.actions.EMImputer"

    # Maximum number of iterations for the algorithm to converge
    max_iter: int = 20
    # Tolerance stopping criterion
    tol: float = 1e-3
    # Seed for reproducibility
    random_state: int | None = None


@dataclass
class MICEImputerConfig(ActionConfig):
    """
    Configuration for MICE (Multiple Imputation by Chained Equations).
    """

    name: str = "MICEImputer"
    _target_: str = "learn2clean.actions.MICEImputer"

    # Maximum number of imputation rounds
    # Increased to 50 for better convergence with tree-based estimators
    max_iter: int = 50
    # Seed for reproducibility (essential for MICE)
    random_state: int = 42


@dataclass
class KNNImputerConfig(ActionConfig):
    """
    Configuration for K-Nearest Neighbors Imputation.
    """

    name: str = "KNNImputer"
    _target_: str = "learn2clean.actions.KNNImputer"

    # Number of neighbors to use
    n_neighbors: int = 5
    # Weight function used in prediction ('uniform' or 'distance')
    weights: str = "uniform"
    # Distance metric for searching neighbors
    metric: str = "nan_euclidean"


@dataclass
class MFImputerConfig(ActionConfig):
    """
    Configuration for Matrix Factorization Imputation.
    """

    name: str = "MFImputer"
    _target_: str = "learn2clean.actions.MFImputer"

    # Note: Previous params like 'fill_value' were likely copy-paste errors.
    # MF usually requires rank/n_components and iterations.

    # Rank of the matrix decomposition (latent factors)
    rank: int = 10
    # Maximum number of optimization iterations
    max_iter: int = 200
    # Seed for reproducibility
    random_state: int | None = None


# --- Simple / Statistical Imputers ---


@dataclass
class MeanImputerConfig(ActionConfig):
    """
    Configuration for Mean Imputation.
    Simple strategy: replaces missing values with the column mean.
    """

    name: str = "MeanImputer"
    _target_: str = "learn2clean.actions.MeanImputer"
    # No hyperparameters needed for standard mean imputation.


@dataclass
class MedianImputerConfig(ActionConfig):
    """
    Configuration for Median Imputation.
    Robust strategy: replaces missing values with the column median.
    """

    name: str = "MedianImputer"
    _target_: str = "learn2clean.actions.MedianImputer"
    # No hyperparameters needed for standard median imputation.


@dataclass
class RandomImputerConfig(ActionConfig):
    """
    Configuration for Random Imputation.
    Replaces missing values with a random valid value from the column.
    """

    name: str = "RandomImputer"
    _target_: str = "learn2clean.actions.RandomImputer"

    # Seed for reproducibility
    random_state: int | None = None


# --- Registration ---


def register_imputation_configs(cs: ConfigStore) -> None:
    grp = "action/imputation"

    cs.store(group=grp, name="em", node=EMImputerConfig)
    cs.store(group=grp, name="knn", node=KNNImputerConfig)
    cs.store(group=grp, name="mean", node=MeanImputerConfig)
    cs.store(group=grp, name="median", node=MedianImputerConfig)
    cs.store(group=grp, name="mf", node=MFImputerConfig)
    cs.store(group=grp, name="mice", node=MICEImputerConfig)
    cs.store(group=grp, name="random", node=RandomImputerConfig)
