"""Module implementing median of ratios scaling for bulk RNAseq data."""

import numpy as np
import pandas
from sklearn.base import BaseEstimator


class MedianRatioScaler(BaseEstimator):
    """Implements the scaling method used in DESeq2 as described [1].

    [1] https://scienceparkstudygroup.github.io/research-data-management-lesson/median_of_ratios_manual_normalization/

    Attributes
    ----------
    filtered_genes: np.array
        List of Genes to keep when scaling
    log_means: np.array
        Genes' pseudo-references based on train set
    """

    def __init__(self):
        super().__init__()
        self.filtered_genes = None
        self.log_means = None

    def fit(self, raw_counts: np.ndarray):
        """Compute the pseudo-reference sample.

        Parameters
        ----------
        raw_counts: pandas.DataFrame
            Raw counts matrix, samples * genes
        """
        raw_counts = pandas.DataFrame(data=raw_counts)
        # Compute gene-wise mean log counts
        log_counts = raw_counts.apply(np.log)
        logmeans = log_counts.mean(0)
        # Filter out genes with -∞ log means
        filtered_genes = ~np.isinf(logmeans).values
        self.filtered_genes = filtered_genes
        self.log_means = logmeans

    def transform(self, raw_counts: np.ndarray):
        """Normalize the raw counts matrix using pseudo-reference sample.

        Parameters
        ----------
        raw_counts: pandas.DataFrame
            Raw counts matrix, samples * genes

        Returns
        -------
            pandas.DataFrame: Normalized counts using Median of Ratios method
        """
        raw_counts = pandas.DataFrame(data=raw_counts)
        log_counts = raw_counts.apply(np.log)
        # Subtract filtered log means from log counts
        log_ratios = log_counts.iloc[:, self.filtered_genes] - self.log_means[self.filtered_genes]
        # Compute sample-wise median of log ratios
        log_medians = log_ratios.median(1)
        # Return raw counts divided by size factors (exponential of log ratios)
        # and size factors
        size_factors = np.exp(log_medians)
        deseq2_counts = raw_counts.div(size_factors, 0)
        return np.log2(deseq2_counts + 1).values

    def fit_transform(self, raw_counts: np.ndarray):
        """Compute the pseudo-reference sample and normalize the raw counts matrix.

        Parameters
        ----------
        raw_counts: pandas.DataFrame
            Raw counts matrix, samples * genes

        Returns
        -------
            pandas.DataFrame: Normalized counts using Median of Ratios method
        """
        raw_counts = pandas.DataFrame(data=raw_counts)
        self.fit(raw_counts)
        return self.transform(raw_counts)
