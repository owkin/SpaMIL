"""Common metrics."""

from typing import Optional

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index


def compute_cindex(logits: np.ndarray, labels: np.ndarray, df_index: Optional[np.ndarray] = None):
    """Compute C-Index. Logits represent the risk.

    Parameters
    ----------
    logits: array_like
        Predicted risk (not survival time). Shape (n_samples, 1) or (n_samples,).
    labels: array_like
        Shape (n_samples, 1) or (n_samples,)
    df_index: array_like, optional.
        If given, predictions and targets with the same index are replaced by their average.
        Useful, for instance, to average different predictions for a same patient (e.g., with different slides).

    Returns
    -------
    C-Index, or -1 if error.
    """
    if logits.ndim == 2:
        logits = logits.squeeze(axis=1)
    if labels.ndim == 2:
        labels = labels.squeeze(axis=1)
    logits, labels = avg_per_unique_index(logits, labels, df_index)
    try:
        times, events = np.abs(labels), 1 * (labels > 0)
        return concordance_index(times, -logits, events)
    except Exception:
        return -1


def avg_per_unique_index(predictions, targets, df_index):
    """Use for averaging predictions per index value (e.g., patient ID).

    Parameters
    ----------
    predictions: array_like
        Shape (n,) or (n, a)
    targets: array_like
        Shape (n,) or (n, b)
    df_index: array_like, optional
        Shape (n,)

    Returns
    -------
    predictions_avg: array_like
        Shape (n_unique, a), where n_unique is the number of unique values in df_index.
    targets_avg: array_like
        Shape (n_unique, b), where n_unique is the number of unique values in df_index.
    """
    if df_index is None:
        df_index = np.arange(targets.shape[0])

    # Number of unique indices
    n_unique = len(np.unique(df_index))

    # Get the shapes of the input arrays
    shape_1 = predictions.shape
    shape_2 = targets.shape

    # Determine the dimensions of the output array
    if len(shape_1) == 1:
        # If predictions is 1-dimensional, reshape it to (n, 1)
        predictions = np.reshape(predictions, (-1, 1))

    if len(shape_2) == 1:
        # If targets is 1-dimensional, reshape it to (n, 1)
        targets = np.reshape(targets, (-1, 1))

    # Concatenate the arrays along the second axis (axis=1)
    preds_and_targets = np.concatenate((predictions, targets), axis=1)

    df = pd.DataFrame(data=preds_and_targets, index=df_index)
    df.index.names = ["df_index"]
    dfavg_per_unique_index = df.groupby("df_index").mean()

    predictions_avg = dfavg_per_unique_index.values[:, : predictions.shape[1]].astype(predictions.dtype)
    targets_avg = dfavg_per_unique_index.values[:, predictions.shape[1] :].astype(targets.dtype)

    # Reshape output
    if len(shape_1) == 1:
        predictions_avg = predictions_avg.reshape(n_unique)
    else:
        predictions_avg = predictions_avg.reshape((n_unique, *shape_1[1:]))
    if len(shape_2) == 1:
        targets_avg = targets_avg.reshape(n_unique)
    else:
        targets_avg = targets_avg.reshape((n_unique, *shape_2[1:]))

    return predictions_avg, targets_avg
