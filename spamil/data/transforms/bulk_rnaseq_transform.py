"""Preprocessing steps and basic feature selection for RNA-seq data."""

import pathlib
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from spamil.data.datasets.base_dataset import BaseDataset
from spamil.data.transforms.base_transform import BaseTransform
from spamil.data.transforms.median_of_ratios import MedianRatioScaler
from spamil.data.transforms.utils import (
    difference_variance,
    ratio_variance,
    wasserstein_distance_vectorized,
)


def _log2_scaler():
    return FunctionTransformer(func=lambda x: np.log2(x + 1))


def _log_scaler():
    return FunctionTransformer(func=np.log1p)


def _min_max_scaler():
    return MinMaxScaler()


def _mean_scaler():
    return StandardScaler(with_std=False)


def _mean_std_scaler():
    return StandardScaler(with_std=True)


def _median_of_ratios_scaler():
    return MedianRatioScaler()


def _identity_scaler():
    return FunctionTransformer(func=None)


class RowWiseZScoreNormalizer:
    """Normalize each row of a matrix to have zero mean and unit variance."""

    def fit(self, X, *args, **kwargs):  # noqa: N803
        """Do nothing."""
        del X, args, kwargs

    def fit_transform(self, X, *args, **kwargs):  # noqa: N803
        """Transform the input data using the normalizer."""
        del args, kwargs
        return self.transform(X)

    def transform(self, X, *args, **kwargs):  # noqa: N803
        """Transform the input data using the normalizer."""
        del args, kwargs
        if isinstance(X, pd.DataFrame):
            X = X.values  # noqa: N806
        return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)


# Sample-wise Z score
def _z_score_scaler():
    return RowWiseZScoreNormalizer()


DISTANCE_FUNCTIONS = {
    "wasserstein": wasserstein_distance_vectorized,
    "ratio_variance": ratio_variance,
    "difference_variance": difference_variance,
}


SCALERS = {
    "min_max": _min_max_scaler,
    "mean": _mean_scaler,
    "mean_std": _mean_std_scaler,
    "median_of_ratios": _median_of_ratios_scaler,
    "log": _log_scaler,
    "log2": _log2_scaler,
    "identity": _identity_scaler,
    "z_score": _z_score_scaler,
    "log-min_max": lambda: Pipeline([("log", _log_scaler()), ("min_max", _min_max_scaler())]),
    "log-mean": lambda: Pipeline([("log", _log_scaler()), ("mean", _mean_scaler())]),
    "log-mean_std": lambda: Pipeline([("log", _log_scaler()), ("mean_std", _mean_std_scaler())]),
    "log-z_score": lambda: Pipeline([("log", _log_scaler()), ("z_score", _z_score_scaler())]),
}

CLASSIC_FILTERS = {"mad", "variance", "log-mad", "log-variance", "seuratv3"}

DISTANCE_BASED_FILTERS = {
    "wasserstein",
    "ratio_variance",
    "difference_variance",
    "scaled-wasserstein",
    "scaled-ratio_variance",
    "scaled-difference_variance",
}
ALL_FILTERS = set.union(CLASSIC_FILTERS, DISTANCE_BASED_FILTERS)


class RnaSeqPreprocessing(BaseTransform):
    """Preprocesses normalized RNAseq data.

    Steps to train the transformation:
        1. Select genes in given gene list, if specified.
        2. If filtering_method is not None, compute ranks of genes according to filtering method. If a list of methods
           is given, the mininum rank across methods is used. Select top max_genes genes based on their ranks.
        3. Train scaler

    Transformation:
        1. Filter columns
        2. Apply scaler

    Parameters
    ----------
    scaling_method: str
        Scaling method to apply. Possible choices:
            * "min_max": scale between 0 and 1
            * "mean": zero-mean
            * "mean_std": zero-mean and unit-variance
            * "median_of_ratios": see :class:`spamil.data.transforms.median_of_ratios.MedianRatioScaler`
            * "log": log-transform x -> log(1+x)
            * "identity": no transformation
            * "log-min_max": log-transform followed by "min_max"
            * "log-mean": log-transform followed by  "mean"
            * "log-mean_std": log-transform followed by "mean_std"
    max_genes: int = -1
        Number of genes with highest variance to keep. Keep all genes if max_genes <= 0.
    gene_list : Sequence[str] or str, optional
        List of strings containing the gene names, or path to csv file whose first column contains gene names.
    filtering_method: Union[str, list[str]] = 'variance'
        Method used to rank genes. Possible values:
            * "mad": select genes with highest median_abs_deviation
            * "variance": select genes with highest variance
            * "log-mad": apply log-transform and select genes with highest median_abs_deviation
            * "log-variance": apply log-transform and select genes with highest variance
            * "wasserstein": select genes with highest Wasserstein distance
                to healthy tissue (requires get_healthy_rnaseq)
            * "ratio_variance": select genes with highest variance ratio with respect
                to healthy tissue (requires get_healthy_rnaseq)
            * "difference_variance": select genes with highest difference in variance with respect
                to healthy tissue (requires get_healthy_rnaseq)
            * "scaled-wasserstein": same as "wasserstein", but data are scaled (with scaling_method)
                before computing distances
            * "scaled-ratio_variance": same as "ratio_variance", but data are scaled (with scaling_method)
                before computing distances
            * "scaled-difference_variance": same as "difference_variance", but data are scaled (with scaling_method)
                before computing distances
    get_healthy_rnaseq: Callable[[], pd.DataFrame]
        Function that returns a dataframe containing RNASeq for healthy tissue.
        Required if filtering_method contains 'wasserstein', 'ratio_variance', 'difference_variance'.

    Attributes
    ----------
    scaler:
        Scaler class selected according to the scaling_method string
    genes_kept: Sequence[str]
        Names of genes which are kept.
    gene_indices: Sequence[int]
        Index of genes which are kept.

    Raises
    ------
    AssertionError
        An AssertionError if the gene list is smaller than max_genes.
    AssertionError
        An AssertionError if the scaler method is unknown.
    AssertionError
        An AssertionError if the log scaling parameter is unknown.
    NotImplementedError
        A NotImplementedError if no healthy reference tissue is given for a
        distance-based gene selection method.
    """

    def __init__(
        self,
        max_genes: int = 5000,
        scaling_method: str = "log-mean",
        filtering_method: Union[str, list[str]] = "log-mad",
        gene_list: Union[str, Sequence[str]] = "",
        get_healthy_rnaseq: Callable[[], pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        self.scaling_method = scaling_method
        self.max_genes = max_genes
        self._genes_kept = None  # initialized in fit()
        self.get_healthy_rnaseq = get_healthy_rnaseq
        self.compare_to_healthy_tissue = self.get_healthy_rnaseq is not None

        self.predefined_gene_list = gene_list or []
        if len(gene_list) > 0:
            if isinstance(gene_list, str) and pathlib.Path(gene_list).is_file():
                # Path to file containing the list of genes
                self.predefined_gene_list = pd.read_csv(gene_list).iloc[:, 0].tolist()

        if (
            filtering_method is not None
            and len(self.predefined_gene_list) > 0
            and self.max_genes > len(self.predefined_gene_list)
        ):
            raise AssertionError(
                f"Trying to select {self.max_genes} genes but only selecting {len(self.predefined_gene_list)} "
                "genes based on the given `gene_list`. Either reduce `max_genes` or provide a different `gene_list`."
            )
        try:
            self.scaler = SCALERS[self.scaling_method]()
        except KeyError as exc:
            raise AssertionError(f"Scaling method must be {SCALERS.keys()}, got '{self.scaling_method}'") from exc

        # List of filtering methods
        self._filters_to_apply = None
        if filtering_method is not None:
            if isinstance(filtering_method, str):
                self._filters_to_apply = [filtering_method]
            else:
                self._filters_to_apply = filtering_method
            # Check that filtering methods are valid
            for method in self._filters_to_apply:
                if method not in ALL_FILTERS:
                    raise AssertionError(f"{method} is not a valid filtering method.")
                if (method in DISTANCE_BASED_FILTERS) and (self.get_healthy_rnaseq is None):
                    raise AssertionError(
                        f"Gene selection method {method} needs a reference healthy tissue. "
                        "Please provide a get_healthy_rnaseq function to RnaSeqPreprocessing."
                    )
        self.columns_to_keep = []

    @property
    def genes_kept(self) -> Tuple[str]:
        """List of gene names that are kept after feature selection."""
        return tuple(self._genes_kept)

    @property
    def gene_indices(self) -> Sequence[str]:
        """List of gene names that are kept after feature selection."""
        return self.columns_to_keep

    def fit(self, train_dataset: BaseDataset):
        """Compute gene list and fit the scaler used for later transformation.

        Parameters
        ----------
        train_dataset: BaseDataset
            Dataset with untransformed RNA-seq data

        Returns
        -------
        self

        Raises
        ------
        NotImplementedError
            Raises error for invalid gene selection methods
        """
        if "rnaseq" in train_dataset.modalities:
            X = train_dataset.features_dataframes["rnaseq"].copy()  # noqa: N806
            original_columns = X.columns

            # Create dummy batch, retrieve batch if any
            batch = pd.Series(0, index=X.index)

            #
            # Step 1: Intersection with input gene list and protein coding genes
            #
            if len(self.predefined_gene_list) > 0:
                X = X.loc[:, X.columns.intersection(self.predefined_gene_list)]  # noqa: N806

            #
            # Step 2: compute gene ranks and take the top max_genes
            #
            if self._filters_to_apply and self.max_genes > 0:
                gene_ranks = self.rank_genes(X, batch)
                self._genes_kept = gene_ranks.sort_values()[: self.max_genes].index
            else:
                self._genes_kept = X.columns

            #
            # Step 3: Scaler
            #

            # Save gene list for future fit_transform but keep the same columns.
            self.columns_to_keep = [original_columns.get_loc(g) for g in self._genes_kept]

            X = X[self._genes_kept].values  # noqa: N806
            self.scaler.fit(X)

            # Future fit_transform with this object will re-use the same genes and just scale the data.
            self._filters_to_apply = None

        return self

    def rank_genes(  # pylint: disable=too-many-branches,too-many-statements
        self, X: pd.DataFrame, batch: pd.Series  # noqa: N803
    ) -> pd.Series:
        """Rank genes, then take the minimum rank of each gene across methods.

        Parameters
        ----------
        X : pd.DataFrame
            RNA-seq data (Filtered on gene list, possibly log-scaled)
        batch : pd.Series
            Series containing the batch ID for each sample

        Returns
        -------
        pd.Series


        Raises
        ------
        NotImplementedError
            Raises error for invalid gene selection methods
        """
        gene_ranks = {}
        X_healthy = None  # noqa: N806

        for filtering in self._filters_to_apply:
            for _batch in batch.unique():
                X_batch = X[batch == _batch]  # noqa: N806
                method = filtering.lower()
                #
                # Classic filters
                #
                if method in {"variance", "log-variance"}:
                    if "log-" in method:
                        variances = np.var(np.log1p(X_batch), axis=0)
                    else:
                        variances = np.var(X_batch, axis=0)
                    # Ranks, highest variance first
                    ranks = (-variances).argsort().argsort()
                elif method in {"mad", "log-mad"}:
                    if "log-" in method:
                        mads = scipy.stats.median_abs_deviation(np.log1p(X_batch.values), axis=0, scale=1)
                    else:
                        mads = scipy.stats.median_abs_deviation(X_batch.values, axis=0, scale=1)
                    # Ranks, highest mad first
                    ranks = pd.Series((-mads), index=X_batch.columns).argsort().argsort()
                #
                # Distance-based filters
                #
                else:
                    # handle cases where method starts with "scaled-"
                    scale_before_distance = "scaled-" in method
                    method = method.split("-")[-1]

                    assert self.get_healthy_rnaseq is not None
                    # Tumor / healthy comparison selection
                    try:
                        distance_function = DISTANCE_FUNCTIONS[method]
                    except KeyError as exc:
                        raise NotImplementedError(
                            f"Gene selection method must be in {DISTANCE_FUNCTIONS.keys()}, got '{method}'"
                        ) from exc
                    if X_healthy is None:
                        X_healthy = self.get_healthy_rnaseq()  # noqa: N806
                    common_genes = X_batch.columns.intersection(X_healthy.columns)
                    if scale_before_distance:
                        # Use normalized data to compute distances
                        X_norm = pd.DataFrame(  # noqa: N806
                            self.scaler.fit_transform(X_batch[common_genes]),
                            columns=common_genes,
                        )
                        X_healthy_norm = pd.DataFrame(  # noqa: N806
                            self.scaler.transform(X_healthy[common_genes]),
                            columns=common_genes,
                        )
                        xx, x_healthy = X_norm, X_healthy_norm
                    else:
                        xx, x_healthy = X_batch[common_genes], X_healthy[common_genes]
                    distances = distance_function(xx.values, x_healthy.values)
                    distances = pd.DataFrame({"distance": distances}, index=common_genes)
                    # Ranks, greatest distance first
                    ranks = distances.rank(ascending=False)
                # Add ranks to results dict
                gene_ranks[f"{filtering}_{_batch}"] = ranks

        # Perform union of methods : take min ranks across methods and batches
        min_gene_ranks = pd.concat([ranks for _, ranks in gene_ranks.items()], axis=1).apply(min, axis=1)

        return min_gene_ranks

    def transform(self, batch):
        """Transform data following preprocessing steps."""
        if "rnaseq" in batch["features"]:
            X = batch["features"]["rnaseq"]  # noqa: N806
            X = X[..., self.columns_to_keep]  # noqa: N806
            X = self.scaler.transform(X)  # noqa: N806
            batch["features"]["rnaseq"] = X
            if "feature_names" not in batch:  # avoid losing previously stored names
                batch["feature_names"] = {}
            batch["feature_names"]["rnaseq"] = self.genes_kept
        return batch


def train_rnaseq_transforms(
    train_datasets: Sequence[BaseDataset],
    other_datasets: Optional[dict[str, BaseDataset]] = None,
    **kwargs,
) -> Sequence[RnaSeqPreprocessing]:
    """Train instances of :class:`RnaSeqPreprocessing` transforms."""
    del other_datasets
    return [RnaSeqPreprocessing(**kwargs).fit(train_dataset) for train_dataset in train_datasets]
