"""Base transform for single cell RNAseq."""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from loguru import logger
from scipy import sparse
from sklearn.decomposition import PCA

from spamil.data.datasets.base_dataset import BaseDataset
from spamil.data.transforms.base_transform import BaseTransform
from spamil.data.transforms.utils import (
    apply_inclusion_exclusion_gene_lists,
    read_gene_list,
    select_genes_pseudobulk_variance,
    select_genes_random,
)
from spamil.utils.constants import SINGLE_CELL_RNASEQ


SUPPORTED_GENE_SELECTION_METHODS = {"pseudobulk_variance", "random_genes"}
SUPPORTED_RPZ_METHODS = {"pca", "scvi"}


# disable logger.info for the scvi transform part for get_latent_representation
scvi_logger = logging.getLogger("scvi.model.base._base_model")  # noqa: F811
scvi_logger.setLevel(logging.WARNING)


def check_parameters(
    select_based_on: str | None = None,
    n_genes: int | None = None,
    rpz_method: str | None = None,
    rpz_dim: int | None = None,
) -> Tuple[int | None, int | None]:
    """Check the parameters of the transform.

    Parameters
    ----------
    select_based_on: str | None
        The type of variance to use for gene selection. Either 'pseudobulk_variance' or
        'random_genes'.
    n_genes : int | None
        The number of variable genes to keep. If None, no filtering is applied.
    rpz_method: str | None
        The representation method to apply to reduce dimension. If None, no rpz method
        is applied.
    rpz_dim: int | None
        The number of components to keep in the rpz method. If None, no rpz method is
        applied.

    Returns
    -------
    n_genes: int | None
        The number of variable genes to keep.
    rpz_dim: int | None
        The number of components to keep in the rpz method.

    Raises
    ------
    NotImplementedError
        Raised in any of the following cases:
        - If the chosen gene filtering method is not implemented.
        - If the chosen rpz method is not implemented.
    ValueError
        Raised in any of the following cases:
        - If n_genes was set to None even though a filtering method was provided.
        - If rpz_dim was set to None even though a rpz_method was provided.
        - If rpz_dim is higher than n_genes.
    """
    if rpz_method is None and rpz_dim is not None:
        logger.warning(
            f"rpz_dim was set to {rpz_dim} even though rpz_method is None. Ignoring"
            " rpz_dim, thus no dimensionality reduction will be applied."
        )
        rpz_dim = None
    if select_based_on is None and n_genes is not None:
        logger.warning(
            "n_genes was not set to None even though select_based_on is None. "
            "Ignoring n_genes, thus applying no gene filtering method."
        )
        n_genes = None
    if (
        select_based_on is not None
        and select_based_on not in SUPPORTED_GENE_SELECTION_METHODS
    ):
        raise NotImplementedError(
            "The chosen gene selection technique is not implemented. "
            f"Available options are: {SUPPORTED_GENE_SELECTION_METHODS}."
        )
    if rpz_method is not None and rpz_method not in SUPPORTED_RPZ_METHODS:
        raise NotImplementedError(
            "The chosen rpz_method is not implemented. "
            f"Available options are: {SUPPORTED_RPZ_METHODS}"
        )
    if n_genes is None and select_based_on is not None:
        raise ValueError(
            "n_genes was set to None even though a gene filtering method was "
            f"provided. ({select_based_on}). Please choose a n_genes."
        )
    if rpz_dim is None and rpz_method is not None:
        raise ValueError(
            "rpz_dim was set to None even though a rpz_method was provided "
            f"({rpz_method}). Please choose a rpz_dim."
        )
    if rpz_dim is not None and n_genes is not None and rpz_dim > n_genes:
        raise ValueError("The rpz dim is greater than the number of selected genes.")
    if rpz_method is not None and select_based_on is not None:
        logger.warning(
            "Both rpz_method and select_based_on have values inside the provided "
            "transform config. The RPZ method will be performed on selected genes."
        )

    return n_genes, rpz_dim


class SingleCellRNASeqTransform(BaseTransform):
    """Apply scRNAseq base transform (gene filtering, scaling...).

    Parameters
    ----------
    gene_list_to_keep : str | Sequence[str]
        The list of genes to keep. If a path to a file is given, the genes are loaded
        from the file.
    gene_list_to_exclude : str | Sequence[str]
        The list of genes to exclude. If a path to a file is given, the genes are loaded
         from the file.
    select_based_on: str | None
        The type of variance to use for gene selection. Either 'pseudobulk_variance' or
        'random_genes'.
    n_genes : int | None
        The number of variable genes to keep. If None, no filtering is applied.
    rpz_method: str | None
        The representation method to apply to reduce dimension. If None, no rpz method
        is applied.
    rpz_dim: int | None
        The number of components to keep in the rpz method. If None, no rpz method is
        applied.
    scvi_model_kwargs: dict | None
        The model hyperparameters for scvi
    scvi_train_kwargs: dict | None
        The training hyperparameters for scvi.
    """

    def __init__(
        self,
        gene_list_to_keep: str | Sequence[str] = "",
        gene_list_to_exclude: str | Sequence[str] = "",
        select_based_on: str | None = None,
        n_genes: int | None = None,
        rpz_method: str | None = None,
        rpz_dim: int | None = None,
        scvi_model_kwargs: dict | None = None,
        scvi_train_kwargs: dict | None = None,
    ):
        super().__init__()

        n_genes, rpz_dim = check_parameters(
            rpz_dim=rpz_dim,
            rpz_method=rpz_method,
            n_genes=n_genes,
            select_based_on=select_based_on,
        )

        self.rpz_method = rpz_method
        self.rpz_dim = rpz_dim
        self.select_based_on = select_based_on
        self.n_genes = n_genes
        self.scvi_model_kwargs = scvi_model_kwargs
        self.scvi_train_kwargs = scvi_train_kwargs

        self.gene_list_to_keep = read_gene_list(gene_list_to_keep)
        self.gene_list_to_exclude = read_gene_list(gene_list_to_exclude)

    def init_scvi(self, feats: np.ndarray, min_counts: int | None = 100):
        """Initialise the scvi rpz method.

        Parameters
        ----------
        feats: np.ndarray
            The concatenated (across spots and patients) features to use.
        min_counts: int | None
            If not None, all cells with a UMI below min_counts will be filtered out for scVI training.
        """
        # Create anndata
        adata = ad.AnnData(
            X=sparse.csr_matrix(feats),
            obs=self.obs,
        )
        if min_counts is not None:
            sc.pp.filter_cells(adata, min_counts=min_counts)
        # Initialize model
        scvi.model.SCVI.setup_anndata(
            adata,
        )
        self.rpz_method_func = scvi.model.SCVI(
            adata, n_latent=self.rpz_dim, **self.scvi_model_kwargs
        )
        del adata

    def fit(
        self, train_dataset: BaseDataset, min_counts: int | None = 100
    ) -> SingleCellRNASeqTransform:
        """Compute gene list and fit the scaler used for later transformation.

        Parameters
        ----------
        train_dataset : BaseDataset
            Dataset with scRNA-seq data
        min_counts: int | None
            If not None, all cells with a UMI below min_counts will be filtered out for scVI training.

        Returns
        -------
        SingleCellRNASeqTransform

        Raises
        ------
        ValueError
            If single cell RNAseq is not found in the dataset.
        """
        if SINGLE_CELL_RNASEQ not in train_dataset.modalities:
            raise ValueError("Single cell RNAseq not found in the dataset.")

        init_gene_ids = train_dataset.metadata_dataframe["gene_id"]
        genes_kept = apply_inclusion_exclusion_gene_lists(
            init_gene_ids, self.gene_list_to_keep, self.gene_list_to_exclude
        )
        # Set the columns to keep based on the selected genes selection method
        if self.n_genes is not None:
            if self.select_based_on == "random_genes":
                genes_kept = select_genes_random(
                    genes_kept=genes_kept,
                    n_genes=self.n_genes,
                )  # random genes, to compare with most variable ones
            elif self.select_based_on == "pseudobulk_variance":
                genes_kept = select_genes_pseudobulk_variance(
                    train_dataset=train_dataset,
                    genes_kept=genes_kept,
                    modality=SINGLE_CELL_RNASEQ,
                    n_genes=self.n_genes,
                )

        # map gene names to their original gene ids
        self.columns_to_keep = np.where(np.isin(init_gene_ids, genes_kept))[0].tolist()

        # Fit the rpz method
        if self.rpz_method is not None:
            list_feats = train_dataset.get_x(SINGLE_CELL_RNASEQ)
            feats = np.concatenate([feat["features"] for feat in list_feats], axis=0)
            if len(self.columns_to_keep) == 0:
                raise ValueError(
                    "The selected genes are empty. Please choose a different "
                    "selected_based_on and/or gene_list."
                )
            feats = feats[:, self.columns_to_keep]
            if self.rpz_method == "pca":
                # pca method
                logger.info("Fitting PCA dimensionality reduction method.")
                self.rpz_method_func = PCA(n_components=self.rpz_dim, random_state=42)
                self.rpz_method_func.fit(feats)
            elif self.rpz_method == "scvi":
                # scvi method
                logger.info("Fitting scVI dimensionality reduction method.")
                self.obs: pd.DataFrame = pd.DataFrame(
                    index=[f"cell_{i}" for i in range(feats.shape[0])],
                )
                self.init_scvi(feats, min_counts=min_counts)
                self.rpz_method_func.train(
                    train_size=1,
                    accelerator="gpu" if torch.cuda.is_available() else "auto",
                    **self.scvi_train_kwargs,
                )

        return self

    def transform(self, batch):
        """Transform data following preprocessing steps."""
        if SINGLE_CELL_RNASEQ not in batch["features"]:
            raise ValueError("Visium counts not found in the dataset.")
        feats = np.stack(
            [feat["features"] for feat in batch["features"][SINGLE_CELL_RNASEQ]]
        )
        feats = feats[:, :, self.columns_to_keep]
        if self.rpz_method is not None:
            batch_size, n_cells, n_feats = feats.shape
            feats = feats.reshape((batch_size * n_cells, n_feats))
            if self.rpz_method == "pca":
                feats = self.rpz_method_func.transform(feats)
            elif self.rpz_method == "scvi":
                obs = pd.DataFrame(
                    index=[f"cell_{i}" for i in range(feats.shape[0])],
                )
                adata = ad.AnnData(X=sparse.csr_matrix(feats), obs=obs)
                feats = self.rpz_method_func.get_latent_representation(adata)
                adata.X = (
                    None  # to avoid memory leak of the sparse matrices at each batch
                )
                del adata

            feats = feats.reshape((batch_size, n_cells, feats.shape[-1]))

        batch["features"][SINGLE_CELL_RNASEQ] = np.array(
            [{"features": feat} for feat in feats]
        )
        return batch


def train_single_cell_rnaseq_transforms(
    train_datasets: Sequence[BaseDataset],
    other_datasets: dict[str, BaseDataset] | None = None,
    **kwargs,
) -> Sequence[SingleCellRNASeqTransform]:
    """Train instances of :class:`SingleCellRNASeqTransform` transforms."""
    del other_datasets
    return [
        SingleCellRNASeqTransform(**kwargs).fit(train_dataset)
        for train_dataset in train_datasets
    ]
