from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DistanceConfig:
    """
    Base configuration schema for a distance metric.

    Attributes:
        _target_: The full Python path to the class to be instantiated by Hydra.
    """

    _target_: str = MISSING
    name: str | None = None


@dataclass
class WeightedDistanceConfig:
    """
    Configuration wrapper to apply a weight to a specific distance metric.

    Attributes:
        distance: The distance metric configuration.
        weight: A float multiplier for the distance (default is 1.0).
    """

    distance: DistanceConfig
    weight: float = 1.0


@dataclass
class DistancesConfig:
    """
    Configuration container for a list of distance metrics.

    Attributes:
        distances: A list of DistanceConfig objects.
    """

    distances: list[DistanceConfig] = field(default_factory=list)


@dataclass
class WeightedDistancesConfig:
    """
    Configuration container for a list of weighted distance metrics.

    Attributes:
        distances: A list of WeightedDistanceConfig objects.
    """

    distances: list[WeightedDistanceConfig] = field(default_factory=list)


def register_distance_configs(cs: ConfigStore) -> None:
    """
    Registers the distance metric configurations into Hydra's ConfigStore.

    This function populates the 'distance' group in Hydra with:
    1. Individual named metrics (e.g., 'distance/wasserstein').
    2. List containers for composition (e.g., 'distance/distances').
    """
    group = "distance"
    base_package = "learn2clean.distance"

    distances_registry: dict[str, str] = {
        # 1. DISTRIBUTION
        "wasserstein": "wasserstein.WassersteinDistance",
        "kl_divergence": "kullback_leibler.KullbackLeiblerDistance",
        "jensen_shannon": "jensen_shannon.JensenShannonDistance",
        "chi_squared": "chi_squared.ChiSquaredDistance",
        "hist_intersection": "histogram_intersection.HistogramIntersectionDistance",
        "hist_correlation": "histogram_correlation.HistogramCorrelationDistance",
        # 2. MOMENTS (Skewness)
        "skewness_mean": "skewness.MeanSkewnessDistance",
        "skewness_max": "skewness.MaxSkewnessDistance",
        "skewness_median": "skewness.MedianSkewnessDistance",
        # 2. MOMENTS (Kurtosis)
        "kurtosis_mean": "kurtosis.MeanKurtosisDistance",
        "kurtosis_max": "kurtosis.MaxKurtosisDistance",
        "kurtosis_median": "kurtosis.MedianKurtosisDistance",
        # 2. MOMENTS (Variance)
        "variance_mean": "variance.MeanVarianceDistance",
        "variance_max": "variance.MaxVarianceDistance",
        "variance_median": "variance.MedianVarianceDistance",
        # 3. STRUCTURAL
        "correlation_matrix": "correlation_matrix.CorrelationMatrixDistance",
        "missingness": "missingness.MissingnessRatioDistance",
        "row_count": "row_count.RowCountDistance",
    }

    # Register individual distance metrics dynamically
    for name, relative_path in distances_registry.items():
        cs.store(
            group=group,
            name=name,
            node=DistanceConfig(_target_=f"{base_package}.{relative_path}"),
        )

    cs.store(name="distances", node=DistancesConfig)
    cs.store(name="weighted_distances", node=WeightedDistancesConfig)
