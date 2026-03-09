from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium import spaces
from learn2clean.types import Features, OptionalTarget


class BaseObserver(ABC):
    """
    Abstract Base Class (Strategy Pattern) for data observation.

    This class defines the interface for extracting state representations
    from the current dataset and environment context. By subclassing this,
    you can easily switch between different observation strategies (e.g.,
    simple meta-features, deep learning embeddings, raw statistics) without
    modifying the core Environment logic.

    Attributes:
        n_actions (int): The total number of available actions in the environment.
                         Useful for observers that need to encode action history
                         (e.g., creating One-Hot vectors).
    """

    def __init__(self, n_actions: int = 0):
        """
        Initializes the Observer.

        Args:
            n_actions: The size of the action space. This is typically set
                       by the Environment upon initialization.
        """
        self.n_actions = n_actions

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """
        Defines the Gymnasium observation space structure.

        This method must return a valid `gymnasium.spaces` object (e.g., Box, Dict)
        that describes the shape and bounds of the observation vector/tensor.
        It is called by the Environment during its initialization.

        Returns:
            spaces.Space: The Gymnasium space definition.
        """
        pass

    @abstractmethod
    def observe(
        self,
        X: Features,
        y: OptionalTarget,
        action_history: np.ndarray | None = None,
    ) -> Any:
        """
        Transforms the current state into an agent-readable observation.

        Args:
            X: The current features (DataFrame).
            y: The current target variable.
            action_history: An optional binary vector representing the
                            history of executed actions (used in Sequential Environments).

        Returns:
            Any: The observation (numpy array, dictionary, etc.) matching
                 the structure defined in `get_observation_space()`.
        """
        pass
