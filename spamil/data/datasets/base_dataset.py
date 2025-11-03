"""The base dataset class from which each modality's dataset class inherits to train transforms and models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch.utils.data


class BaseDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for multimodal datasets.

    Subclasses must define:
        - self.modalities: list[str]
        - self.features_dataframes: dict[str, pd.DataFrame]
        - self.labels_dataframe: dict[str, pd.DataFrame]
        - self.metadata_dataframe: pd.DataFrame

    And must implement:
        - get_x(modality: str) -> list[dict[str, np.ndarray]]
        - __getitem__(idx)
        - __len__()
    """

    def __init__(self):
        super().__init__()

        self.modalities: list[str] = []
        self.features_dataframes: dict[str, pd.DataFrame] = {}
        self.labels_dataframe: pd.DataFrame = pd.DataFrame()
        self.metadata_dataframe: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    def get_x(self, modality: str) -> list[dict[str, np.ndarray]]:
        """Return a list of feature dicts for a given modality, one feature dict per patient."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Return a single sample (features, label, metadata)."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        pass
