"""Defines default cross-val split function: repeated stratified group k-fold."""

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from spamil.data.datasets.base_dataset import BaseDataset


TrainIndices = Sequence[int]
ValidationIndices = Sequence[int]


def default_split(
    full_dataset: BaseDataset,
    stratify_for_survival_or_binary_classification: bool = False,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 123,
) -> Sequence[Tuple[TrainIndices, ValidationIndices]]:
    """Return train and validation indices according to StratifiedGroupKFold.

    Parameters
    ----------
    full_dataset:
        Dataset containing all data
    stratify_for_survival_or_binary_classification:
        If True, stratify based on (labels > 0)
    n_splits: int
        Number of cross-validation splits
    n_repeats: int
        Number of cross-validation repetitions
    seed:
        Seed to define random state
    """
    stratify = stratify_for_survival_or_binary_classification

    train_and_validation_indices = []

    # Get cross-validation splits
    labels_dataframes = full_dataset.labels_dataframes
    df_label = labels_dataframes[sorted(full_dataset.tasks)[0]]
    groups = df_label.index.values
    if stratify:
        col = df_label.columns.values[0]
        stratify_on = df_label[col].values > 0
    else:
        stratify_on = None

    split_generator = repeated_stratified_group_kfold(
        n=len(df_label),
        groups=groups,
        stratify_on=stratify_on,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
    )

    for _, (train_indices, val_indices) in enumerate(split_generator):
        train_and_validation_indices.append((train_indices, val_indices))
    return train_and_validation_indices


def repeated_stratified_group_kfold(
    n: int,
    groups: Optional[np.ndarray] = None,
    stratify_on: Optional[np.ndarray] = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 123,
):
    """Stratified Group K-Fold.

    Generate splits stratified on `stratify_on`, forcing patients within `groups` to
    belong to the same fold.

    The code assumes that the values in the stratify_on array, if given, are constant in each group. For instance,
    it can be used if there are several histology slides per patient, groups=patient_ids
    and stratify_on represents survival labels, which are the same for all slides of a single patient.

    Parameters
    ----------
    n: int
        number of elements
    groups : list, np.array (1D), pd.Series (optional)
        Group encoding (should be categorical).
        Elements of the same group will always find themselves in the same fold.
        Typical use is when several patients have several slides.
    stratify_on : list, np.array (1D), pd.Series (optional)
        Element stratification (should be categorical)
        Folds will be optimized to keep the proportion of this stratification variable within each fold.
        Usage examples: can be used to respect label balance in each fold, or to maintain cancer
        proportions in pan-cancer models.
        If given, the code assumes that the values of stratify_on are the same in each group.
    n_splits : int, optional
        number of splits in the cross validation, by default 5
    n_repeats : int, optional
        number of repetitions, by default 1
    seed : int, optional
        random seed, by default 123

    Returns
    -------
    cv_idx: list[tuple]
        n_repeats*n_splits tuples containing (train_indices, test_indices)
    """
    # groups: if not provided, affect each sample to a different group
    groups = np.array(groups) if groups is not None else np.array(range(n))
    # stratify
    if stratify_on is not None:
        stratify_on = np.array(stratify_on)
    else:
        stratify_on = np.zeros(n)
    # sklearn's splitting class
    SplittingClass = RepeatedStratifiedKFold if stratify_on is not None else RepeatedKFold  # noqa: N806
    # split on unique groups and then retrieve sample indices
    unique_groups, indices = np.unique(groups, return_index=True)
    rkf = SplittingClass(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    def degroup_idx(grp_idx):
        return np.where(np.isin(groups, unique_groups[grp_idx]))[0]

    split_gen = (
        (degroup_idx(train_idx_grp), degroup_idx(test_idx_grp))
        for train_idx_grp, test_idx_grp in rkf.split(indices, stratify_on[indices])
    )
    return split_gen
