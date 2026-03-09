from hydra.utils import instantiate
from omegaconf import OmegaConf

from learn2clean.actions import (
    EMImputer,
    KNNImputer,
    MeanImputer,
    MedianImputer,
    MFImputer,
    MICEImputer,
    RandomImputer,
)
from learn2clean.configs.actions.cleaning.imputation import (
    EMImputerConfig,
    KNNImputerConfig,
    MeanImputerConfig,
    MedianImputerConfig,
    MFImputerConfig,
    MICEImputerConfig,
    RandomImputerConfig,
)


class TestImputationConfigs:
    def test_mean_imputer_config(self):
        cfg = OmegaConf.structured(MeanImputerConfig)
        action = instantiate(cfg)
        assert isinstance(action, MeanImputer)
        assert action.name == "MeanImputer"

    def test_median_imputer_config(self):
        cfg = OmegaConf.structured(MedianImputerConfig(columns=["age"]))
        action = instantiate(cfg)
        assert isinstance(action, MedianImputer)
        assert action.name == "MedianImputer"
        assert action.columns == ["age"]

    def test_knn_imputer_config(self):
        cfg = OmegaConf.structured(KNNImputerConfig)
        action = instantiate(cfg)
        assert isinstance(action, KNNImputer)
        assert action.params.get("n_neighbors") == 5
        cfg_override = OmegaConf.structured(
            KNNImputerConfig(n_neighbors=10, weights="distance")
        )
        action_override = instantiate(cfg_override)
        assert action_override.params.get("n_neighbors") == 10
        assert action_override.params.get("weights") == "distance"

    def test_mice_imputer_config(self):
        cfg = OmegaConf.structured(MICEImputerConfig)
        action = instantiate(cfg)
        assert isinstance(action, MICEImputer)
        cfg_override = OmegaConf.structured(MICEImputerConfig())
        action_override = instantiate(cfg_override)
        assert action_override.params.get("max_iter") == 50
        assert action_override.params.get("random_state") == 42

    def test_em_imputer_config(self):
        cfg = OmegaConf.structured(EMImputerConfig(max_iter=50, tol=1e-3))
        action = instantiate(cfg)
        assert isinstance(action, EMImputer)
        assert action.params.get("max_iter") == 50
        assert action.params.get("tol") == 1e-3

    def test_mf_imputer_config(self):
        cfg = OmegaConf.structured(MFImputerConfig())
        action = instantiate(cfg)
        assert isinstance(action, MFImputer)

    def test_random_imputer_config(self):
        cfg = OmegaConf.structured(RandomImputerConfig(random_state=123))
        action = instantiate(cfg)
        assert isinstance(action, RandomImputer)
        assert action.params.get("random_state") == 123

    def test_imputation_inheritance_base_config(self):
        cfg = OmegaConf.structured(
            MeanImputerConfig(columns=["col1", "col2"], exclude=["col3"])
        )
        action = instantiate(cfg)
        assert action.columns == ["col1", "col2"]
        assert action.exclude == ["col3"]
