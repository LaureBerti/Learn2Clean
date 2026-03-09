import importlib

import pytest
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from learn2clean.configs.distances import register_distance_configs
from learn2clean.distance.chi_squared import ChiSquaredDistance
from learn2clean.distance.correlation_matrix import CorrelationMatrixDistance
from learn2clean.distance.histogram_correlation import HistogramCorrelationDistance
from learn2clean.distance.histogram_intersection import HistogramIntersectionDistance
from learn2clean.distance.jensen_shannon import JensenShannonDistance
from learn2clean.distance.kullback_leibler import KullbackLeiblerDistance
from learn2clean.distance.kurtosis import (
    MaxKurtosisDistance,
    MeanKurtosisDistance,
    MedianKurtosisDistance,
)
from learn2clean.distance.missingness import MissingnessRatioDistance
from learn2clean.distance.row_count import RowCountDistance
from learn2clean.distance.skewness import (
    MaxSkewnessDistance,
    MeanSkewnessDistance,
    MedianSkewnessDistance,
)
from learn2clean.distance.variance import (
    MaxVarianceDistance,
    MeanVarianceDistance,
    MedianVarianceDistance,
)
from learn2clean.distance.wasserstein import WassersteinDistance


def get_target(cls) -> str:
    """Helper to reconstruct the full dotted path: 'module.ClassName'."""
    return f"{cls.__module__}.{cls.__name__}"


@pytest.fixture(scope="module", autouse=True)
def setup_config_store():
    """
    Fixture that runs once per module.
    It initializes the Hydra ConfigStore and registers all distance configurations
    so they are available for testing.
    """
    ConfigStore.instance().__init__()
    register_distance_configs(ConfigStore.instance())


class TestDistanceConfigs:
    """
    Test suite for Hydra Metric Configurations.
    Verifies that configs are correctly registered, contain the right parameters,
    and point to valid Python classes.
    """

    def test_distances_are_registered(self):
        """Ensure that the 'distance' group exists in ConfigStore and is not empty."""
        cs = ConfigStore.instance()
        assert "distance" in cs.repo
        assert (
            len(cs.repo["distance"]) > 0
        ), "No distances were registered in the ConfigStore."

    @pytest.mark.parametrize(
        "distance_name, expected_target, expected_params",
        [
            ("chi_squared", get_target(ChiSquaredDistance), {}),
            ("correlation_matrix", get_target(CorrelationMatrixDistance), {}),
            ("hist_correlation", get_target(HistogramCorrelationDistance), {}),
            ("hist_intersection", get_target(HistogramIntersectionDistance), {}),
            ("jensen_shannon", get_target(JensenShannonDistance), {}),
            ("kl_divergence", get_target(KullbackLeiblerDistance), {}),
            ("kurtosis_max", get_target(MaxKurtosisDistance), {}),
            ("kurtosis_mean", get_target(MeanKurtosisDistance), {}),
            ("kurtosis_median", get_target(MedianKurtosisDistance), {}),
            ("missingness", get_target(MissingnessRatioDistance), {}),
            ("row_count", get_target(RowCountDistance), {}),
            ("skewness_max", get_target(MaxSkewnessDistance), {}),
            ("skewness_mean", get_target(MeanSkewnessDistance), {}),
            ("skewness_median", get_target(MedianSkewnessDistance), {}),
            ("variance_max", get_target(MaxVarianceDistance), {}),
            ("variance_mean", get_target(MeanVarianceDistance), {}),
            ("variance_median", get_target(MedianVarianceDistance), {}),
            ("wasserstein", get_target(WassersteinDistance), {}),
        ],
    )
    def test_config_content(self, distance_name, expected_target, expected_params):
        """
        Validates the content of each registered configuration.
        Checks if '_target_' matches the actual class path and if parameters are correct.
        """
        cs = ConfigStore.instance()

        # Retrieve the config node from the store
        node = cs.repo["distance"][f"{distance_name}.yaml"]
        cfg = OmegaConf.create(node.node)

        # 1. Verify the target class path
        assert cfg._target_ == expected_target, (
            f"Invalid target for '{distance_name}'. "
            f"Expected: {expected_target}, Got: {cfg._target_}"
        )

        # 2. Verify specific default parameters
        for param, value in expected_params.items():
            assert (
                cfg[param] == value
            ), f"Incorrect value for parameter '{param}' in '{distance_name}'."

    @pytest.mark.parametrize(
        "distance_name",
        [
            "chi_squared",
            "correlation_matrix",
            "hist_correlation",
            "hist_intersection",
            "jensen_shannon",
            "kl_divergence",
            "kurtosis_max",
            "kurtosis_mean",
            "kurtosis_median",
            "missingness",
            "row_count",
            "skewness_max",
            "skewness_mean",
            "skewness_median",
            "variance_max",
            "variance_mean",
            "variance_median",
            "wasserstein",
        ],
    )
    def test_target_class_resolves(self, distance_name):
        """
        Critically important test:
        Verifies that the string defined in '_target_' can actually be imported
        and corresponds to an existing class in the codebase.
        """
        cs = ConfigStore.instance()
        node = cs.repo["distance"][f"{distance_name}.yaml"]
        cfg = OmegaConf.create(node.node)

        target_path = cfg._target_
        module_name, class_name = target_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(
                f"The module '{module_name}' defined in config '{distance_name}' "
                f"could not be imported. Error: {e}"
            )

        assert hasattr(module, class_name), (
            f"The class '{class_name}' does not exist in module '{module_name}' "
            f"(Config: {distance_name})"
        )

    def test_instantiation_dry_run(self):
        """
        Smoke test: Attempts to actually instantiate one of the configs via Hydra.
        This ensures the constructor signature is compatible with the config.
        """
        cs = ConfigStore.instance()
        # Testing 'skewness' as a representative simple distance
        node = cs.repo["distance"]["kurtosis_mean.yaml"]
        cfg = OmegaConf.create(node.node)

        try:
            obj = instantiate(cfg)
            assert obj is not None
            # Ideally, assert isinstance(obj, BaseDistance)
        except Exception as e:
            pytest.fail(
                f"Hydra failed to instantiate the 'skewness' config. Error: {e}"
            )
