"""
Base interface for all dataset loaders in the Learn2Clean project.

This module defines the abstract contract (DatasetLoader) that all concrete
data loaders (CSVLoader, OpenMLLoader, etc.) must implement.
"""

from abc import ABC, abstractmethod

import pandas as pd  # Type hint for the returned data structure

from learn2clean.utils.logging_mixin import LoggingMixin


class DatasetLoader(ABC, LoggingMixin):
    """
    Abstract Base Class (ABC) for all dataset loaders.

    Defines the standard interface for loading datasets. Any concrete subclass
    must implement the abstract methods defined here. The main responsibility
    is to provide data as a pandas DataFrame.
    """

    def __init__(self, target_col: str | None = None):
        """
        Initializes the DatasetLoader base.

        Calls the constructors of the parent classes (ABC and LoggingMixin)
        via super() to ensure the Method Resolution Order (MRO) is respected.
        """
        super().__init__()
        self.target_col = target_col
        # Logs initialization event, useful for tracking object creation in Hydra/RL loop
        self.log.debug(f"Initialized base loader for type: {self.__class__.__name__}")

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Loads the dataset from its source and returns it as a pandas DataFrame.

        Subclasses must implement the logic to fetch, read, and structure the data.

        Returns
        -------
        pd.DataFrame
            The loaded dataset as a DataFrame.
        """
        # Raises an error if a concrete class fails to implement this method.
        raise NotImplementedError(
            "The 'load' method must be implemented by concrete subclasses."
        )

    @abstractmethod
    def get_source_info(self) -> dict:
        """
        Returns metadata about the dataset source (e.g., file path, OpenML ID, Kaggle slug).

        This method is crucial for tracking the origin of the data used in an experiment.

        Returns
        -------
        dict
            A dictionary containing source-specific metadata.
        """
        pass
