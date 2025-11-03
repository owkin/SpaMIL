"""Implement multimodal Chowder (late fusion) algorithm."""

from typing import Optional

import torch
import torch.nn

from spamil.modules.extreme_layer import ExtremeLayer
from spamil.modules.mlp import MLP
from spamil.modules.tile_layers import TilesMLP


class MultimodalChowderLateFusion(torch.nn.Module):
    """Multimodal Chowder (late fusion) model.

    Parameters
    ----------
    in_features: dict[str, int]
        Dictionnary containing the input dimension of each modality. Example: :code:`{"clin": 10, "rnaseq": 5000}`.
    out_features: dict[str, int]
        Dictionnary containing the output dimension of each task.  Example: :code:`{"survival": 1}`.
    histo_feature_name: str
        Name of the histo feature in the batch.
    visium_feature_name: str
        Name of the visium feature in the batch.
    task: str
        Which task/label is being predicted.
    use_common_extremes: bool
        Use the extreme indices from histo for visium and vice versa.
    n_extreme: Optional[int]
        Number of extreme tiles
    n_top: Optional[int]
        Number of max tiles
    n_bottom: Optional[int]
        Number of min tiles
    tiles_mlp_hidden_histo: Optional[list[int]]
        Size of tiles MLP layers for histo.
    tiles_mlp_hidden_visium: Optional[list[int]]
        Size of tiles MLP layers for visium.
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
        if mlp_hidden and mlp_dropout do not match in dimension and value
    """

    def __init__(
        self,
        in_features: dict[str, int],
        out_features: dict[str, int],
        histo_feature_name: str = "histo",
        visium_feature_name: str = "visium_counts",
        task: str = "survival",
        use_common_extremes: bool = False,
        n_extreme: Optional[int] = None,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        tiles_mlp_hidden_histo: Optional[list[int]] = None,
        tiles_mlp_hidden_visium: Optional[list[int]] = None,
        mlp_hidden: Optional[list[int]] = None,
        mlp_dropout: Optional[list[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        super().__init__()
        self.histo_feature_name = histo_feature_name
        self.histo_dim = in_features[self.histo_feature_name]
        self.visium_feature_name = visium_feature_name
        self.visium_dim = in_features[self.visium_feature_name]
        self.task = task
        self.use_common_extremes = use_common_extremes

        self.chowder_histo = ChowderBranch(
            in_features=self.histo_dim,
            out_features=out_features[task],
            n_extreme=n_extreme,
            n_top=n_top,
            n_bottom=n_bottom,
            tiles_mlp_hidden=tiles_mlp_hidden_histo,
            bias=bias,
        )
        self.chowder_visium = ChowderBranch(
            in_features=self.visium_dim,
            out_features=out_features[task],
            n_extreme=n_extreme,
            n_top=n_top,
            n_bottom=n_bottom,
            tiles_mlp_hidden=tiles_mlp_hidden_visium,
            bias=bias,
        )

        # Predict survival
        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(mlp_dropout), (
                    "mlp_hidden and mlp_dropout must have the same length"
                )
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        if n_top is None or n_bottom is None:
            n_top = n_extreme
            n_bottom = n_extreme
        mlp_in_features = (
            4 * (n_top + n_bottom)  # type: ignore[operator]
            if self.use_common_extremes
            else 2 * (n_top + n_bottom)  # type: ignore[operator]
        )
        self.mlp = MLP(
            mlp_in_features,
            1,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        self.mlp.apply(weight_initialization)

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
        x_visium, x_visium_mask = (
            batch["features"][self.visium_feature_name],
            batch["features"][f"{self.visium_feature_name}_collate_0"],
        )

        # Get extreme scores
        scores_histo, extreme_scores_histo, extreme_indices_histo = self.chowder_histo(
            x_histo, x_histo_mask
        )
        scores_visium, extreme_scores_visium, extreme_indices_visium = (
            self.chowder_visium(x_visium, x_visium_mask)
        )
        if self.use_common_extremes:
            # Apply MLP to the N_TOP + N_BOTTOM scores of both common tiles <-> spots
            # i.e. 2*(N_TOP + N_BOTTOM) scores of each modality
            extreme_scores_histo_from_visium = torch.take_along_dim(
                scores_histo, extreme_indices_visium, dim=1
            )
            extreme_scores_visium_from_histo = torch.take_along_dim(
                scores_visium, extreme_indices_histo, dim=1
            )
            extreme_scores = torch.cat(
                [
                    extreme_scores_histo,
                    extreme_scores_histo_from_visium,
                    extreme_scores_visium,
                    extreme_scores_visium_from_histo,
                ],
                dim=1,
            )
        else:
            # Apply MLP to the N_TOP + N_BOTTOM scores of each modality
            extreme_scores = torch.cat(
                [
                    extreme_scores_histo,
                    extreme_scores_visium,
                ],
                dim=1,
            )
        out = self.mlp(extreme_scores.transpose(1, 2))  # (B, OUT_FEATURES, 1)
        out = out.squeeze(2)

        predictions = {
            self.task: out,
        }
        extras = {
            "extreme_scores": extreme_scores,
        }
        return predictions, extras


class ChowderBranch(torch.nn.Module):
    """Chowder branch in the multimodal late fusion network.

    Parameters
    ----------
    in_features: dict[str, int]
        Dictionnary containing the input dimension of each modality. Example: :code:`{"clin": 10, "rnaseq": 5000}`.
    out_features: dict[str, int]
        Dictionnary containing the output dimension of each task.  Example: :code:`{"survival": 1}`.
    n_extreme: Optional[int]
        Number of extreme tiles
    n_top: Optional[int]
        Number of max tiles
    n_bottom: Optional[int]
        Number of min tiles
    tiles_mlp_hidden: Optional[list[int]]
        Size of tiles MLP layers.
    bias: bool
        Whether to use bias in linear layers of TilesMLP.

    Raises
    ------
    ValueError
        if n_top and n_extreme are None and n_extreme is None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_extreme: Optional[int] = None,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        tiles_mlp_hidden: Optional[list[int]] = None,
        bias: bool = True,
    ):
        super().__init__()

        if n_extreme is not None:
            n_top = n_extreme
            n_bottom = n_extreme

        if n_top is None and n_bottom is None:
            raise ValueError("At least one of `n_top` or `n_bottom` must not be None.")

        self.score_model = TilesMLP(
            in_features, hidden=tiles_mlp_hidden, bias=bias, out_features=out_features
        )
        self.score_model.apply(weight_initialization)

        self.extreme_layer = ExtremeLayer(
            n_top=n_top, n_bottom=n_bottom, return_indices=True
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Do forward pass.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        scores, extreme_scores, extreme_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (B, N_TOP + N_BOTTOM, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        """
        scores = self.score_model(x=x, mask=mask)
        extreme_scores, extreme_indices = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, N_TOP + N_BOTTOM, OUT_FEATURES)

        return scores, extreme_scores, extreme_indices


def weight_initialization(module: torch.nn.Module):
    """Initialize weights."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)

        if module.bias is not None:
            module.bias.data.fill_(0.0)
