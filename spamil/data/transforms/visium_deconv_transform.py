"""Base transform for visium deconvolution."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA

from spamil.data.datasets.base_dataset import BaseDataset
from spamil.data.transforms.base_transform import BaseTransform
from spamil.utils.constants import VISIUM_DECONV


CELLTYPE_CATEGORIES = {
    "immune": [
        "B cell",
        "Plasma B",
        "NK",
        "CD4/CD8",
        "DC",
        "TAM-BDM",
        "TAM-MG",
        "Mono",
        "Neutrophil",
        "Mast",
    ],
    "malignant": ["AC-like", "MES-like", "NPC-like", "OPC-like"],
    "stromal": ["Endothelial", "Mural cell", "Astrocyte", "Oligodendrocyte"],
    "rest": ["Neuron", "RG", "OPC"],
}


class VisiumDeconvTransformAblation(BaseTransform):
    """Apply base transformations to Visium deconvolution features, including optional PCA.

    This transform retrieves the `visium_deconv` modality from the provided dataset
    and can optionally apply a PCA-based dimensionality reduction, as well as exclude
    user-chosen cell type categories.

    Arguments
    ---------
    pca_dim: int | None
        The number of components to keep in the pca method. If None, no pca is applied.
    exclude_categories: Sequence[str] | None
        The cell type categories to exclude.
    use_fractions: bool
        If True, it will use the cell type fractions. Else, it will use the cell type weights.
    """

    def __init__(
        self,
        pca_dim: Optional[int] = None,
        exclude_categories: Optional[Sequence[str]] = None,
        use_fractions: bool = True,
    ) -> None:
        super().__init__()
        self.pca_dim = pca_dim
        self.pca: Optional[PCA] = None
        self.exclude_categories = exclude_categories if exclude_categories else []
        self.use_fractions = use_fractions

        self.all_cell_types = None
        self.indices_to_keep = None

    def fit(self, train_dataset: BaseDataset):
        """Identify which columns to keep and optionally fit PCA."""
        modality = VISIUM_DECONV
        if modality not in train_dataset.modalities:
            raise ValueError(f"{modality} not found in the dataset.")

        list_feats = train_dataset.get_x(modality)
        feats = np.concatenate([feat["features"] for feat in list_feats], axis=0)

        # Get cell type names from the metadata_dataframe
        self.all_cell_types = list(train_dataset.metadata_dataframe["cell_type"].values)

        # Figure out which cell types are in categories we want to exclude
        excluded_celltypes = set()
        for cat in self.exclude_categories:
            excluded_celltypes.update(CELLTYPE_CATEGORIES[cat])

        # Build a list of indices we want to keep
        self.indices_to_keep = []
        for i, ct in enumerate(self.all_cell_types):
            if ct not in excluded_celltypes:
                self.indices_to_keep.append(i)

        # Subset feats to only those columns
        feats = feats[:, self.indices_to_keep]

        # If we want fractions, re-normalize sums to 1
        if self.use_fractions:
            sum_over_kept = feats.sum(axis=-1, keepdims=True) + 1e-8
            feats = feats / sum_over_kept

        # Fit PCA
        if self.pca_dim is not None:
            self.pca = PCA(n_components=self.pca_dim, random_state=42)
            self.pca.fit(feats)

        return self

    def transform(self, batch: dict) -> dict:
        """Subset columns based on exclude_categories, optionally re-normalize, and apply PCA."""
        modality = VISIUM_DECONV
        if modality not in batch["features"]:
            raise ValueError(f"{modality} not found in the batch.")

        # feats shape: (batch_size, n_spots, n_celltypes)
        feats = np.stack([feat["features"] for feat in batch["features"][modality]])

        # Subset columns
        feats = feats[:, :, self.indices_to_keep]

        # Re-normalize if using fractions
        if self.use_fractions:
            sum_over_kept = feats.sum(axis=-1, keepdims=True) + 1e-8
            feats = feats / sum_over_kept

        # PCA transform
        if self.pca is not None:
            batch_size, n_spots, n_feats = feats.shape
            feats_reshaped = feats.reshape((batch_size * n_spots, n_feats))
            feats_reshaped = self.pca.transform(feats_reshaped)
            feats = feats_reshaped.reshape((batch_size, n_spots, self.pca_dim))

        batch["features"][modality] = [{"features": feat} for feat in feats]
        return batch


def train_visium_deconv_transforms(
    train_datasets: Sequence[BaseDataset],
    other_datasets: dict[str, BaseDataset] | None = None,
    **kwargs,
) -> Sequence[VisiumDeconvTransformAblation]:
    """Train instances of :class:`VisiumDeconvTransformAblation` transforms."""
    del other_datasets
    return [
        VisiumDeconvTransformAblation(**kwargs).fit(train_dataset)
        for train_dataset in train_datasets
    ]
