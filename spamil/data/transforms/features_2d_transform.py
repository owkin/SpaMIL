"""Base transform for visium counts."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA

from spamil.data.datasets.base_dataset import BaseDataset
from spamil.data.transforms.base_transform import BaseTransform


class Features2dTransform(BaseTransform):
    """Apply base transform (var columns filtering, scaling...).

    Parameters
    ----------
    feature_name : str
        Name of the feature to transform.
    pca_dim : int | None
        The number of components to keep in the PCA. If None, no PCA is applied.
    """

    def __init__(
        self,
        feature_name: str,
        pca_dim: int | None = None,
    ):
        super().__init__()
        self.feature_name = feature_name

        if pca_dim is not None:
            self.pca = PCA(n_components=pca_dim, random_state=42)
        else:
            self.pca = None

    def fit(self, train_dataset: BaseDataset) -> Features2dTransform:
        """Compute gene list and fit the scaler used for later transformation.

        Parameters
        ----------
        train_dataset : BaseDataset
            Dataset with untransformed visium RNA-seq data

        Returns
        -------
        Features2dTransform

        Raises
        ------
        ValueError
            if visium counts are not found in the dataset.
        """
        if self.feature_name not in train_dataset.modalities:
            raise ValueError("Visium counts not found in the dataset.")

        if self.pca is not None:
            feats = np.concatenate(train_dataset.get_x(self.feature_name), axis=0)
            self.pca.fit(feats)

        return self

    def transform(self, batch):
        """Transform data following preprocessing steps."""
        if self.feature_name not in batch["features"]:
            raise ValueError("Visium counts not found in the dataset.")
        if self.pca is not None:
            feats = batch["features"][self.feature_name]
            batch_size, n_spots, n_feats = feats.shape
            feats = feats.reshape((batch_size * n_spots, n_feats))
            feats = self.pca.transform(feats)
            feats = feats.reshape((batch_size, n_spots, feats.shape[-1]))
            batch["features"][self.feature_name] = feats
        return batch


def train_features_2d_transforms(
    train_datasets: Sequence[BaseDataset],
    other_datasets: dict[str, BaseDataset] | None = None,
    **kwargs,
) -> Sequence[Features2dTransform]:
    """Train instances of :class:`Features2dTransform` transforms."""
    del other_datasets
    return [
        Features2dTransform(**kwargs).fit(train_dataset)
        for train_dataset in train_datasets
    ]
