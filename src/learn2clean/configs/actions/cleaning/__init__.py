from hydra.core.config_store import ConfigStore

from .deduplication import register_deduplication_configs
from .imputation import register_imputation_configs
from .inconsistency_detection import register_inconsistency_detection_configs
from .outlier_detection import register_outlier_configs


def register_cleaning_configs(cs: ConfigStore) -> None:
    register_deduplication_configs(cs)
    register_imputation_configs(cs)
    register_inconsistency_detection_configs(cs)
    register_outlier_configs(cs)
