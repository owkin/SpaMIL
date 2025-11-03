"""Implement abMIL algorithm."""

from typing import Optional

import torch

from spamil.modules.attention import GatedAttention
from spamil.modules.mlp import MLP
from spamil.modules.tile_layers import TilesMLP


class AbMIL(torch.nn.Module):
    """abMIL algorithm.

    Parameters
    ----------
    in_features: dict[str, int]
        Dictionnary containing the input dimension of each modality, e.g. `visium_counts`.
    out_features: dict[str, int]
        Dictionnary containing the output dimension of each task, e.g. `survival`.
    modality: str
        Which modality to use from the batch.
    task: str
        Which task/label is being predicted.
    d_model_attention: int = 128
        Dimension of attention.
    temperature: float = 1.0
        GatedAttention softmax temperature.
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
        d_model_attention: int = 128,
        temperature: float = 1.0,
        tiles_mlp_hidden: Optional[list[int]] = None,
        mlp_hidden: Optional[list[int]] = None,
        mlp_dropout: Optional[list[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        super().__init__()
        self.modality = modality
        self.task = task
        in_features = in_features[modality]
        out_features = out_features[task]

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.tiles_emb = TilesMLP(
            in_features,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=d_model_attention,
        )

        self.attention_layer = GatedAttention(
            d_model=d_model_attention, temperature=temperature
        )

        mlp_in_features = d_model_attention

        self.mlp = MLP(
            in_features=mlp_in_features,
            out_features=out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

    def score_model(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Get attention logits.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        tiles_emb = self.tiles_emb(x, mask)
        attention_logits = self.attention_layer.attention(tiles_emb, mask)
        return attention_logits

    def forward(self, batch):
        """Apply model to batch."""
        features = batch["features"]
        mask = features[f"{self.modality}_collate_0"]

        tiles_emb = self.tiles_emb(features[self.modality], mask)
        scaled_tiles_emb, attention_weights = self.attention_layer(tiles_emb, mask)
        logits = self.mlp(scaled_tiles_emb)

        predictions = {
            self.task: logits,
        }
        extras = {
            "attention_weights": attention_weights,
        }
        return predictions, extras
