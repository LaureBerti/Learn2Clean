from hydra.utils import instantiate
from omegaconf import OmegaConf

from learn2clean.actions import (
    ChiSquareSelector,
    LinearCorrelationSelector,
    MutualInformationSelector,
    RandomForestSelector,
    VarianceThresholdSelector,
)
from learn2clean.configs.actions.preparation.feature_selection import (
    ChiSquareSelectorConfig,
    LinearCorrelationSelectorConfig,
    MutualInformationSelectorConfig,
    RandomForestSelectorConfig,
    VarianceThresholdSelectorConfig,
)


class TestFeatureSelectionConfigs:
    def test_chi_square_selector_config(self):
        cfg = OmegaConf.structured(ChiSquareSelectorConfig)
        action = instantiate(cfg)
        assert isinstance(action, ChiSquareSelector)
        assert action.name == "ChiSquareSelector"
        assert action.k == 10
        cfg_override = OmegaConf.structured(ChiSquareSelectorConfig(k=5))
        action_override = instantiate(cfg_override)
        assert action_override.k == 5

    def test_linear_correlation_selector_config(self):
        cfg = OmegaConf.structured(LinearCorrelationSelectorConfig())
        action: LinearCorrelationSelector = instantiate(cfg)
        assert isinstance(action, LinearCorrelationSelector)
        assert action.k == 10
        cfg_override = OmegaConf.structured(LinearCorrelationSelectorConfig(k=8))
        action_override = instantiate(cfg_override)
        assert action_override.k == 8

    def test_mutual_information_selector_config(self):
        cfg = OmegaConf.structured(MutualInformationSelectorConfig)
        action = instantiate(cfg)
        assert isinstance(action, MutualInformationSelector)
        assert action.k == 10
        cfg_override = OmegaConf.structured(
            MutualInformationSelectorConfig(k=20, random_state=None, n_neighbors=5)
        )
        action_override = instantiate(cfg_override)
        assert action_override.k == 20
        assert action_override.random_state is None
        assert action_override.n_neighbors == 5

    def test_random_forest_selector_config(self):
        cfg = OmegaConf.structured(RandomForestSelectorConfig)
        action = instantiate(cfg)
        assert isinstance(action, RandomForestSelector)
        assert action.n_estimators == 100
        assert action.threshold == "median"
        assert action.max_features is None
        assert action.random_state == 42
        cfg_override = OmegaConf.structured(
            RandomForestSelectorConfig(
                n_estimators=50, threshold="mean", max_features=5, random_state=None
            )
        )
        action_override = instantiate(cfg_override)
        assert action_override.n_estimators == 50
        assert action_override.threshold == "mean"
        assert action_override.max_features == 5
        assert action_override.random_state is None

    def test_variance_threshold_selector_config(self):
        cfg = OmegaConf.structured(VarianceThresholdSelectorConfig)
        action = instantiate(cfg)
        assert isinstance(action, VarianceThresholdSelector)
        assert action.threshold == 0.0
        cfg_override = OmegaConf.structured(
            VarianceThresholdSelectorConfig(threshold=0.1)
        )
        action_override = instantiate(cfg_override)
        assert action_override.threshold == 0.1

    def test_selector_inheritance(self):
        cfg = OmegaConf.structured(
            ChiSquareSelectorConfig(
                columns=["feature1", "feature2"], exclude=["target"]
            )
        )
        action = instantiate(cfg)
        assert action.columns == ["feature1", "feature2"]
        assert action.exclude == ["target"]
