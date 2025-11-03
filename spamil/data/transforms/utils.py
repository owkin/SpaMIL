"""Utils for the spamil related transforms."""

from __future__ import annotations

import concurrent.futures
import functools
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import wasserstein_distance

from spamil.data.datasets.base_dataset import BaseDataset


def apply_inclusion_exclusion_gene_lists(
    init_gene_ids: list,
    gene_list_to_keep: list,
    gene_list_to_exclude: list,
) -> list:
    """Filter out the genes based on the gene lists to keep and exclude.

    Parameters
    ----------
    init_gene_ids: list
        All the genes from the train_dataset
    gene_list_to_keep: list
        The loaded gene list to keep
    gene_list_to_exclude: list
        The loaded gene list to exclude

    Returns
    -------
    genes_kept: list
        The list of genes to keep.
    """
    genes_kept_after_inclusion = set(init_gene_ids.copy())
    if len(gene_list_to_keep) > 0:
        genes_kept_after_inclusion = set(init_gene_ids) & set(gene_list_to_keep)
    genes_kept_after_exclusion = set(init_gene_ids.copy())
    if len(gene_list_to_exclude) > 0:
        genes_kept_after_exclusion = set(init_gene_ids) - set(gene_list_to_exclude)
    genes_kept = list(genes_kept_after_inclusion & genes_kept_after_exclusion)

    return genes_kept


def select_genes_pseudobulk_variance(
    train_dataset: BaseDataset, genes_kept: list, modality: str, n_genes: int
) -> list:
    """Select genes based on pseudobulk variance.

    Parameters
    ----------
    train_dataset : BaseDataset
        Training data (Visium or scRNAseq supported)
    genes_kept : list
        The genes kept so far.
    modality: str
        The modality to use
    n_genes: int
        The number of variable genes to keep.

    Returns
    -------
    genes_kept: list
        The updated list of genes to keep.

    Raises
    ------
    ValueError
        If the number of genes to filter is higher than the resulting set of genes
        after application of inclusion/exclusion gene lists.
    """
    logger.info("Selecting genes based on pseudobulk variance")
    pseudobulk = pd.DataFrame(
        np.stack(
            [
                counts["features"].mean(axis=0)
                for counts in train_dataset.get_x(modality)
            ]
        ),
        columns=train_dataset.metadata_dataframe["gene_id"],
        index=train_dataset.get_dataframe_indices(),
    )
    intersection = list(set(genes_kept) & set(pseudobulk.columns))
    if n_genes > len(intersection):
        raise ValueError(
            "After removing genes based on inclusion/exclusion gene lists, the size of "
            f"the remaining gene set ({len(intersection)}) is smaller than the number "
            f"of variable genes to keep ({n_genes}). Please reduce the number of "
            "variable genes to keep."
        )
    genes_kept = (
        pseudobulk[intersection]
        .var(axis=0)
        .sort_values(ascending=False)
        .iloc[:n_genes]
        .index.tolist()
    )
    return genes_kept


def select_genes_random(genes_kept: list, n_genes: int) -> list:
    """Select random genes.

    Parameters
    ----------
    genes_kept : list
        The genes kept so far.
    n_genes: int
        The number of variable genes to keep.

    Returns
    -------
    genes_kept: list
        The updated list of genes to keep.

    Raises
    ------
    ValueError
        If the number of genes to filter is higher than the resulting set of genes
        after application of inclusion/exclusion gene lists.
    """
    logger.info("Selecting random genes.")
    if n_genes > len(genes_kept):
        raise ValueError(
            "After removing genes based on inclusion/exclusion gene lists, the size of "
            f"the remaining gene set ({len(genes_kept)}) is smaller than the number of "
            f"variable genes to keep ({n_genes}). Please reduce the number of variable "
            "genes to keep."
        )
    genes_kept = np.random.choice(
        genes_kept,
        size=n_genes,
        replace=False,
    ).tolist()
    return genes_kept


def select_genes_moran(
    train_dataset: BaseDataset,
    genes_kept: list,
    modality: str,
    svg_path: str | None,
    radius: int,
    n_genes: int,
) -> list:
    """Select genes based on Moran's I.

    Parameters
    ----------
    train_dataset : BaseDataset
        Training visium data
    genes_kept: list
        The genes kept so far.
    modality: str
        The modality to use
    svg_path: str | None
        The path to the svg folder.
    radius: int
        The radius or degree for neighbor definition, used for Moran's I calculation.
    n_genes: int
        The number of variable genes to keep.

    Returns
    -------
    genes_kept: list
        The updated list of genes to keep.

    Raises
    ------
    ValueError
        If the number of genes to filter is higher than the resulting set of genes
        after application of inclusion/exclusion gene lists.
    """
    logger.info("Selecting genes based on Moran's I")
    indices = train_dataset.get_dataframe_indices()

    morans = []
    for sample_id in indices:
        morans.append(
            load_morans(
                sample_id,
                svg_path=svg_path,
                radius=radius,
                sample_zarr_path=train_dataset.features_dataframes[modality].loc[
                    sample_id, "path"
                ],
                normalization="raw_count",
            )
        )
    morans_df = pd.concat(morans, axis=1)
    intersection = list(set(genes_kept) & set(morans_df.index))
    if n_genes > len(intersection):
        raise ValueError(
            "After removing genes based on inclusion/exclusion gene lists, the size of "
            f"the remaining gene set ({len(intersection)}) is smaller than the number "
            f"of variable genes to keep ({n_genes}). Please reduce the number of "
            "variable genes to keep."
        )
    genes_kept = (
        morans_df.loc[intersection]
        .mean(axis=1)
        .sort_values(ascending=False)
        .iloc[:n_genes]
        .index.tolist()
    )
    return genes_kept


def load_morans(
    sample_id: str,
    svg_path: str,
    radius: int,
    normalization: str,
) -> pd.DataFrame:
    """Load Moran's I values for a given sample. If it does not exist, compute it.

    Parameters
    ----------
    sample_id: str
        id of the sample
    svg_path: str | None
        the path to the svg folder
    radius: int
        radius or degree for neighbor definition
    sample_zarr_path: str
        path to load the zarr of the sample, used only if the Moran's I values are not
        already computed.
    normalization: str
        normalization method used to compute the Moran's I values

    Returns
    -------
    pd.DataFrame
        Moran's I for all the genes in the sample.
    """
    folder = f"{svg_path}radius_{radius}_{normalization}"
    if not os.path.exists(f"{folder}/morans_{sample_id}.parquet"):
        raise ValueError(
            f"No Moran's I was found for sample {sample_id}. "
            "Compute it using `scanpy.metrics.morans_i`"
        )
    else:
        morans = pd.read_parquet(f"{folder}/morans_{sample_id}.parquet")
    return morans


def read_gene_list(gene_list: str | Sequence[str] = "") -> list:
    """Read gene list from different input formats.

    Parameters
    ----------
    gene_list: str | Sequence[str]
        The list of genes. If a path to a file is given, the genes are loaded from the
        file.

    Returns
    -------
    predefined_gene_list: list
        The loaded list of genes to keep or exclude.

    Raises
    ------
    ValueError
        If the gene list is neither a string or a sequence or strings.
    """
    predefined_gene_list = []
    if len(gene_list) > 0:
        if isinstance(gene_list, str) and Path(gene_list).is_file():
            # Path to file containing the list of genes
            predefined_gene_list = pd.read_csv(gene_list).iloc[:, 0].tolist()
        elif isinstance(gene_list, Sequence):
            predefined_gene_list = list(gene_list)
        else:
            raise ValueError("gene_list must be a string or a sequence of strings.")

    return predefined_gene_list


def ratio_variance(x: np.ndarray, y: np.ndarray, threshold=0.05):
    """Return the ratio of the variances of two gene distributions.

    Parameters
    ----------
    x : array_like, shape (n_obs_x, n_genes)
        Values observed in the empirical distribution of the first gene.
    y : array_like, shape (n_obs_y, n_genes)
        Values observed in the empirical distribution of a second gene.
    threshold : float
        Minimum variance

    Returns
    -------
    array_like, shape (n_genes,)
        The computed ratio between the distributions per gene.
    """
    assert x.shape[1] == y.shape[1]
    var_x = np.var(x, axis=0)
    var_y = np.var(y, axis=0)
    ratios = var_x / var_y
    mask = var_y <= threshold
    ratios[mask] = 0.0
    return ratios


def difference_variance(x: np.ndarray, y: np.ndarray):
    """Return the difference of the variances of two gene distributions, x and y.

    Parameters
    ----------
    x : array_like, shape (n_obs_x, n_genes)
        Values observed in the empirical distribution of the first gene.
    y : array_like, shape (n_obs_y, n_genes)
        Values observed in the empirical distribution of a second gene.

    Returns
    -------
    array_like, shape (n_genes,)
        The computed difference between the distributions, per gene.
    """
    assert x.shape[1] == y.shape[1]
    return np.var(x, axis=0) - np.var(y, axis=0)


def wasserstein_distance_vectorized(x: np.ndarray, y: np.ndarray, n_jobs: Optional[int] = None):
    """Return the difference of the variances of two gene distributions, x and y.

    Parameters
    ----------
    x : array_like, shape (n_obs_x, n_genes)
        Values observed in the empirical distribution of the first gene.
    y : array_like, shape (n_obs_y, n_genes)
        Values observed in the empirical distribution of a second gene.
    n_jobs: int, default=None
        Number of threads to use. If None, use maximum value.

    Returns
    -------
    array_like, shape (n_genes,)
        The computed distance between the gene distributions, per gene.
    """
    assert x.shape[1] == y.shape[1]
    n_genes = x.shape[1]
    executor_class = functools.partial(concurrent.futures.ThreadPoolExecutor)
    distances = np.zeros(n_genes)

    def _compute_wassertein_for_one_gene(gene_idx):
        distances[gene_idx] = wasserstein_distance(x[:, gene_idx], y[:, gene_idx])
        return gene_idx

    with executor_class(max_workers=n_jobs) as executor:
        futures = []
        for ii in range(n_genes):
            futures.append(executor.submit(_compute_wassertein_for_one_gene, gene_idx=ii))
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return distances
