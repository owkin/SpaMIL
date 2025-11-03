"""Base transform for clinical data."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from spamil.data.datasets.base_dataset import BaseDataset
from spamil.data.transforms.base_transform import BaseTransform


class ClinicalTransform(BaseTransform):
    """Apply base transform for clinical data."""

    def __init__(
        self,
    ):
        super().__init__()
        self.statistics_per_column = {}

    def fit(self, train_dataset: BaseDataset) -> ClinicalTransform:
        """Compute useful statistics of the train set.

        Parameters
        ----------
        train_dataset : BaseDataset
            Dataset with raw clinical data

        Returns
        -------
        ClinicalTransform

        Raises
        ------
        ValueError
            If the clinical data is not found in the dataset.
        """
        if "clinical" not in train_dataset.modalities:
            raise ValueError("Clinical data not found in the dataset.")

        clinical_train = train_dataset.features_dataframes["clinical"]
        for index_col, column in enumerate(clinical_train.columns):
            self.statistics_per_column[index_col] = {}
            if clinical_train[column].dtype == "O":
                self.statistics_per_column[index_col]["dtype"] = "O"
                most_frequent_value = clinical_train[column].mode()[0]
                self.statistics_per_column[index_col]["most_frequent_value"] = (
                    most_frequent_value
                )
            else:
                self.statistics_per_column[index_col]["dtype"] = "float64"
                median_value = clinical_train[column].median()
                self.statistics_per_column[index_col]["median"] = median_value
                standard_scaler = StandardScaler()
                standard_scaler.fit(np.array(clinical_train[column]).reshape(-1, 1))
                self.statistics_per_column[index_col]["standard_scaler"] = (
                    standard_scaler
                )

        return self

    def transform(self, batch):
        """Transform data following preprocessing steps."""
        if "clinical" not in batch["features"]:
            raise ValueError("Clinical data not found in the dataset.")
        feats = batch["features"]["clinical"].copy()
        for index_col in range(feats.shape[1]):
            if self.statistics_per_column[index_col]["dtype"] == "O":
                # Replace missing values with most frequent one
                mask_nan = [
                    isinstance(x, float) and np.isnan(x) for x in feats[:, index_col]
                ]
                feats[np.array(mask_nan), index_col] = self.statistics_per_column[
                    index_col
                ]["most_frequent_value"]
                mask_unknown = feats[:, index_col] == "Unknown"
                feats[mask_unknown, index_col] = self.statistics_per_column[index_col][
                    "most_frequent_value"
                ]
                # Convert to binary categories to integers
                if len(np.unique(feats[:, index_col])) > 2:
                    raise ValueError(
                        "One clinical variable is categorical and has more than 2 categories. "
                        "Change the ClinicalTransform class or use only categorical variables with 2 categories."
                    )
                feats[:, index_col] = (
                    feats[:, index_col]
                    == self.statistics_per_column[index_col]["most_frequent_value"]
                ).astype(int)
            else:
                # Replace missing values by median
                mask_nan = [
                    isinstance(x, float) and np.isnan(x) for x in feats[:, index_col]
                ]
                feats[np.array(mask_nan), index_col] = self.statistics_per_column[
                    index_col
                ]["median"]
                # Standard Scale the data
                feats[:, index_col] = (
                    self.statistics_per_column[index_col]["standard_scaler"]
                    .transform(feats[:, index_col].reshape(-1, 1))
                    .flatten()
                )

        batch["features"]["clinical"] = feats.astype(float)
        return batch

def train_clinical_transforms(
    train_datasets: Sequence[BaseDataset],
    other_datasets: dict[str, BaseDataset] | None = None,
    **kwargs,
) -> Sequence[ClinicalTransform]:
    """Train instances of :class:`ClinicalTransform` transforms."""
    del other_datasets
    return [
        ClinicalTransform(**kwargs).fit(train_dataset)
        for train_dataset in train_datasets
    ]
