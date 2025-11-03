"""Implement multimodal Chowder (early fusion) algorithm."""

import warnings
from typing import Optional

import torch
import torch.nn

from spamil.modules.extreme_layer import ExtremeLayer
from spamil.modules.mlp import MLP
from spamil.modules.tile_layers import TilesMLP


class MultimodalChowderEarlyFusion(torch.nn.Module):
    """Multimodal Chowder (early fusion) model.

    Parameters
    ----------
    in_features: dict[str, int]
        Dictionnary containing the input dimension of each modality.
    out_features: dict[str, int]
        Dictionnary containing the output dimension of each task.
    histo_feature_name: str
        Name of the histo feature in the batch.
    visium_feature_name: str
        Name of the visium feature in the batch.
    task: str
        Which task/label is being predicted.
    n_extreme: Optional[int]
        Number of extreme tiles
    n_top: Optional[int]
        Number of max tiles
    n_bottom: Optional[int]
        Number of min tiles
    tiles_mlp_hidden_histo: Optional[list[int]]
        Size of tiles MLP layers for histo.
    mlp_hidden: Optional[list[int]]
        Size of prediction MLP layers.
    mlp_dropout: Optional[list[float]]
        List of dropout for each layer.
    mlp_activation: Optional[torch.nn.Module]
        Activation function.
    bias: bool
        Whether to use bias in linear layers of TilesMLP.

    Raises
    ------
    ValueError
        if n_top and n_extreme are None and n_extreme is None
        if mlp_hidden and mlp_dropout do not match in dimension and value
    """

    def __init__(
        self,
        in_features: dict[str, int],
        out_features: dict[str, int],
        histo_feature_name: str = "histo",
        visium_feature_name: str = "visium_counts",
        task: str = "survival",
        n_extreme: Optional[int] = None,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        tiles_mlp_hidden: Optional[list[int]] = None,
        mlp_hidden: Optional[list[int]] = None,
        mlp_dropout: Optional[list[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        super().__init__()

        self.task = task
        self.histo_feature_name = histo_feature_name
        self.histo_dim = in_features[self.histo_feature_name]
        self.visium_feature_name = visium_feature_name
        self.visium_dim = in_features[self.visium_feature_name]
        self.out_features = out_features[self.task]

        if n_extreme is not None:
            warnings.warn(
                DeprecationWarning(
                    f"Use `n_extreme=None, n_top={n_extreme if n_top is None else n_top}, "
                    f"n_bottom={n_extreme if n_bottom is None else n_bottom}` instead."
                ),
                stacklevel=2,
            )

            if n_top is not None:
                warnings.warn(
                    DeprecationWarning(
                        f"Overriding `n_top={n_top}`with `n_top=n_extreme={n_extreme}`."
                    ),
                    stacklevel=2,
                )
            if n_bottom is not None:
                warnings.warn(
                    DeprecationWarning(
                        f"Overriding `n_bottom={n_bottom}`"
                        f"with `n_bottom=n_extreme={n_extreme}`."
                    ),
                    stacklevel=2,
                )

            n_top = n_extreme
            n_bottom = n_extreme

        if n_top is None and n_bottom is None:
            raise ValueError("At least one of `n_top` or `n_bottom` must not be None.")

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(mlp_dropout), (
                    "mlp_hidden and mlp_dropout must have the same length"
                )
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.score_model = TilesMLP(
            self.histo_dim + self.visium_dim,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=self.out_features,
        )
        self.score_model.apply(self.weight_initialization)

        self.extreme_layer = ExtremeLayer(n_top=n_top, n_bottom=n_bottom)

        mlp_in_features = n_top + n_bottom  # type: ignore[operator]
        self.mlp = MLP(
            mlp_in_features,
            1,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        self.mlp.apply(self.weight_initialization)

    @staticmethod
    def weight_initialization(module: torch.nn.Module):
        """Intialize weights."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, batch: dict) -> tuple[dict, dict]:
        """Do forward pass.

        Parameters
        ----------
        batch: dict
            a dict containing input data streams

        Returns
        -------
        tuple[dict, dict]
            predictions
                a dict with the model outputs
            extras
                extra outputs
        """
        x_histo, x_histo_mask = (
            batch["features"][self.histo_feature_name],
            batch["features"][f"{self.histo_feature_name}_collate_0"],
        )
        x_visium, _ = (
            batch["features"][self.visium_feature_name],
            batch["features"][f"{self.visium_feature_name}_collate_0"],
        )
        x_fusion = torch.cat((x_histo, x_visium), 2)

        scores = self.score_model(
            x=x_fusion, mask=x_histo_mask
        )  # same number of tiles and spots so equivalent
        extreme_scores = self.extreme_layer(x=scores, mask=x_histo_mask)

        y = self.mlp(extreme_scores.transpose(1, 2))

        predictions = {
            self.task: y.squeeze(2),
        }
        extras = {
            "extreme_scores": extreme_scores,
        }
        return predictions, extras
