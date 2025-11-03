"""Implement Chowder algorithm."""

from typing import Optional

import torch

from spamil.modules.extreme_layer import ExtremeLayer
from spamil.modules.mlp import MLP
from spamil.modules.tile_layers import TilesMLP


class Chowder(torch.nn.Module):
    """Chowder algorithm.

    Parameters
    ----------
    in_features: dict[str, int]
        Dictionnary containing the input dimension of each modality.
    out_features: dict[str, int]
        Dictionnary containing the output dimension of each task.
    modality: str
        Which modality to use from the batch.
    task: str
        Which task/label is being predicted.
    n_top: int
        Number of max tiles
    n_bottom: int
        Number of min tiles
    tiles_mlp_hidden: Optional[list[int]] = None
        Size of tiles MLP layers.
    mlp_hidden: Optional[list[int]] = None
        Size of prediction MLP layers.
    mlp_dropout: Optional[list[float]] = None
        List of dropout for each layer.
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation function.
    bias: bool = True
        Whether to use bias in linear layers of TilesMLP.
    """

    def __init__(
        self,
        in_features: dict[str, int],
        out_features: dict[str, int],
        modality="visium_counts",
        task="survival",
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
        self.modality = modality
        self.task = task
        in_features=in_features[modality],
        out_features=out_features[task],

        if n_extreme is not None:
            n_top = n_extreme
            n_bottom = n_extreme

        if n_top is None and n_bottom is None:
            raise ValueError("At least one of `n_top` or `n_bottom` must not be None.")

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.score_model = TilesMLP(
            in_features, hidden=tiles_mlp_hidden, bias=bias, out_features=out_features
        )
        self.score_model.apply(self.weight_initialization)

        self.extreme_layer = ExtremeLayer(n_top=n_top, n_bottom=n_bottom)

        mlp_in_features = n_top + n_bottom
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
        """Initialize chowder weights."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, batch):
        """Apply model to batch."""
        features = batch["features"]
        mask = features[f"{self.modality}_collate_0"]

        scores = self.score_model(x=features[self.modality], mask=mask)
        extreme_scores = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        # Apply MLP to the N_TOP + N_BOTTOM scores
        y = self.mlp(extreme_scores.transpose(1, 2))  # (B, OUT_FEATURES, 1)

        predictions = {
            self.task: y.squeeze(2),
        }
        extras = {
            "extreme_scores": extreme_scores,
        }
        return predictions, extras
