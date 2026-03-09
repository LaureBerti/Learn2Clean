from typing import TypeVar

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

# Define a generic type variable to indicate this function returns a list of "Something"
# (e.g., list[DataFrameAction] or list[BaseDistance])
T = TypeVar("T")


def instantiate_list(cfg: DictConfig | ListConfig | None) -> list[T]:
    """
    Instantiates a list of objects from a Hydra configuration.

    Hydra usually returns a DictConfig (index -> config) when composing groups.
    This helper converts the values of that dict into a standard Python list of objects.

    Args:
        cfg: The configuration object (DictConfig or ListConfig) containing the items.

    Returns:
        list[T]: A list of instantiated objects (e.g., Actions, Distances).
    """
    if not cfg:
        return []

    return list(dict(instantiate(cfg)).values())
