"""Collate functions for MultiDatasetSplit."""

from typing import Any, Optional, Tuple

import numpy as np


def pad_collate_fn(
    batch: list[Tuple[dict[str, np.ndarray], Any]],
    batch_first: bool = True,
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """Pad together sequences of imaging features of arbitrary lengths.

    Adds to the samples:
     * a mask of the padding; can later be used to ignore padding in activation functions.

    Expected to be used in combination of a torch.utils.datasets.DataLoader.

    Expect the sequences to be padded to be the first one in the sample tuples.
    Others members will be batched using default_collate

    Parameters
    ----------
    batch: list[dict[np.ndarray]]
        Batch
    batch_first: bool = True
        Either return (B, N_TILES, F) or (N_TILES, B, F)
    max_len: int
        Maximum length of axis 1

    Returns
    -------
    padded_sequences, masks: Tuple[np.ndarray, np.ndarray(bool)]
        - if batch_first: Tuple[(B, N_TILES, F), (B, N_TILES, 1), ...]
        - else: Tuple[(N_TILES, B, F), (N_TILES, B, 1), ...]

        with N_TILES = max_len if max_len is not None
        or N_TILES = max length of the training samples.
    """
    # Expect the sequences to be the first one in the sample tuples
    sequences = []
    others = []
    for sample in batch:
        sequences.append(sample[0])
        others.append(sample[1:])

    dtype = sequences[0]["features"].dtype
    if max_len is None:
        max_len = max([s["features"].shape[0] for s in sequences])  # pylint: disable=R1728
    trailing_dims = sequences[0]["features"].shape[1:]

    if batch_first:
        padded_dims = (len(sequences), max_len) + trailing_dims
        masks_dims = (len(sequences), max_len, 1)
    else:
        padded_dims = (max_len, len(sequences)) + trailing_dims
        masks_dims = (max_len, len(sequences), 1)

    padded_sequences = np.zeros(padded_dims, dtype=dtype)
    masks = np.ones(masks_dims, dtype=bool)

    for i, sample in enumerate(sequences):
        array = sample["features"]
        length = array.shape[0]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            padded_sequences[i, :length, ...] = array[:max_len, ...]
            masks[i, :length, ...] = False
        else:
            padded_sequences[:length, i, ...] = array[:max_len, ...]
            masks[:length, i, ...] = False

    del others

    return (padded_sequences, masks)
