"""Extreme layer module for Chowder."""

from typing import Optional

import torch
from loguru import logger


class ExtremeLayer(torch.nn.Module):
    """Concatenate the n_top top tiles and n_bottom bottom tiles.

    Parameters
    ----------
    n_top: int
        number of top tiles to select
    n_bottom: int
        number of bottom tiles to select
    dim: int
        dimension to select top/bottom tiles from
    return_indices: bool
        Whether to return the indices of the extreme tiles
    """

    def __init__(
        self,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        dim: int = 1,
        return_indices: bool = False,
    ):
        super(ExtremeLayer, self).__init__()

        if not (n_top is not None or n_bottom is not None):
            raise ValueError("one of n_top or n_bottom must have a value.")

        if not ((n_top is not None and n_top > 0) or (n_bottom is not None and n_bottom > 0)):
            raise ValueError("one of n_top or n_bottom must have a value > 0.")

        self.n_top = n_top
        self.n_bottom = n_bottom
        self.dim = dim
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """Apply module to batch.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, ...)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, ...)

        Returns
        -------
        extreme_tiles: torch.Tensor
            (B, N_TOP + N_BOTTOM, ...)
        """
        if self.n_top and self.n_bottom and ((self.n_top + self.n_bottom) > x.shape[self.dim]):
            logger.warning(
                f"Sum of tops is larger than the input tensor shape for dimension {self.dim}: "
                + f"{self.n_top + self.n_bottom} > {x.shape[self.dim]}. Values will appear twice (in top and in bottom)"
            )

        top, bottom = None, None
        top_idx, bottom_idx = None, None
        if mask is not None:
            if self.n_top:
                top, top_idx = x.masked_fill(mask, float("-inf")).topk(
                    k=self.n_top, sorted=True, dim=self.dim
                )
                top_mask = top.eq(float("-inf"))
                if top_mask.any():
                    logger.warning(
                        "The top tiles contain masked values, they will be set to zero."
                    )
                    top[top_mask] = 0

            if self.n_bottom:
                bottom, bottom_idx = x.masked_fill(mask, float("inf")).topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )
                bottom_mask = bottom.eq(float("inf"))
                if bottom_mask.any():
                    logger.warning(
                        "The bottom tiles contain masked values, they will be set to zero."
                    )
                    bottom[bottom_mask] = 0
        else:
            if self.n_top:
                top, top_idx = x.topk(k=self.n_top, sorted=True, dim=self.dim)
            if self.n_bottom:
                bottom, bottom_idx = x.topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )

        if top is not None and bottom is not None:
            values = torch.cat([top, bottom], dim=self.dim)
            indices = torch.cat([top_idx, bottom_idx], dim=self.dim)
        elif top is not None:
            values = top
            indices = top_idx
        elif bottom is not None:
            values = bottom
            indices = bottom_idx
        else:
            raise ValueError

        if self.return_indices:
            return values, indices
        else:
            return values

