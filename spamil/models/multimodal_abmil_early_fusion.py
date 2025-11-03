"""Implement multimodal abMIL (early fusion) algorithm."""

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn

from spamil.modules.attention import GatedAttention
from spamil.modules.mlp import MLP


class MultimodalAbMILEarlyFusion(torch.nn.Module):
    """Build multimodal abMIL (early fusion) model.

    Parameters
    ----------
    in_features: dict
        Dictionary containing the input dimension of each modality. This dict
        must have "histo" and "visium_counts" as keys.
        ex: {"histo": 2048, "visium_counts": 3000}
    out_features: dict
        Dictionary mapping task to desired output shape.
    infer_with_histo: bool
        If true, inference will be done on histo data, otherwise it will be on both.
    d_model_attention_histo: int
        Size of the initial embedding output size before the attention layer of histo.
    d_model_attention_visium: int
        Size of the initial embedding output size before the attention layer of visium.
    temperature: float
        Temperature of the gated attention.
    tiles_mlp_hidden_histo: Optional[Sequence[int]]
        Size of the hidden layers of the initial embedding MLP of histology data.
    tiles_mlp_hidden_visium: Optional[Sequence[int]]
        Size of the hidden layers of the initial embedding MLP of visium counts data.
    mlp_hidden: Optional[Sequence[int]]
        Size of the hidden layers of the output MLP.
    mlp_activation: Optional[torch.nn.Module]
        Activation function to use in the output MLP.
    bias: bool
        Whether to use bias in the MLP layers.
    task: str
        Task name.
    histo_feature_name: str
        Name of the histo feature in the batch.
    visium_feature_name: str
        Name of the visium feature in the batch.

    Raises
    ------
    ValueError
        If d_model_attention_histo and d_model_attention_visium are not equal
        when inferring only with histo data.
    """

    def __init__(
        self,
        in_features: dict,
        out_features: dict,
        infer_with_histo: bool = False,
        d_model_attention_histo: int = 128,
        d_model_attention_visium: int = 128,
        temperature: float = 1.0,
        tiles_mlp_hidden_histo: Optional[Sequence[int]] = (256, 128),
        tiles_mlp_hidden_visium: Optional[Sequence[int]] = (256, 128),
        mlp_hidden: Optional[Sequence[int]] = (128, 64),
        mlp_activation: Optional[torch.nn.Module] = None,
        bias: bool = True,
        task: str = "OS",
        histo_feature_name: str = "histo",
        visium_feature_name: str = "visium_counts",
    ):
        super().__init__()

        if infer_with_histo and d_model_attention_histo != d_model_attention_visium:
            raise ValueError(
                "When inferring only with histo data, d_model_attention_histo must be "
                f"equal to d_model_attention_visium. Got respectively {d_model_attention_histo} "
                f"and {d_model_attention_visium} instead."
            )

        self.task = task
        self.infer_with_histo = infer_with_histo
        self.histo_feature_name = histo_feature_name
        self.visium_feature_name = visium_feature_name
        self.out_features = out_features[self.task]

        # Instance embedding network
        # no need for a TilesMLP since padding is already taken into account in the
        # gated attention network
        self.histo_dim = in_features[self.histo_feature_name]
        self.tiles_mlp_histo = MLP(
            in_features=self.histo_dim,
            out_features=d_model_attention_histo,
            hidden=tiles_mlp_hidden_histo,
            activation=torch.nn.Sigmoid(),
            bias=bias,
        )
        self.visium_dim = in_features[self.visium_feature_name]
        self.tiles_mlp_visium = MLP(
            in_features=self.visium_dim,
            out_features=d_model_attention_visium,
            hidden=tiles_mlp_hidden_visium,
            activation=torch.nn.Sigmoid(),
            bias=bias,
        )

        # Gated attention network
        self.gated_attention = GatedAttention(
            d_model=d_model_attention_histo + d_model_attention_visium,
            temperature=temperature,
        )

        # Output MLP
        self.out_mlp = MLP(
            in_features=d_model_attention_histo
            + d_model_attention_visium,  # either histo twice, or visium/histo concatenated
            out_features=self.out_features,
            hidden=mlp_hidden,
            activation=mlp_activation,
            bias=True,
        )

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
        if not self.infer_with_histo:
            x_histo, x_histo_mask = (
                batch["features"][self.histo_feature_name],
                batch["features"][f"{self.histo_feature_name}_collate_0"],
            )
            x_visium, _ = (
                batch["features"][self.visium_feature_name],
                batch["features"][f"{self.visium_feature_name}_collate_0"],
            )

        else:
            # create dummy X_visium, X_visium_mask for inference
            x_histo, x_histo_mask = (
                batch["features"][self.histo_feature_name],
                batch["features"][f"{self.histo_feature_name}_collate_0"],
            )
            batch_dim = x_histo.shape[0]
            visium_dim = self.visium_dim
            x_visium = torch.zeros((batch_dim, 1, visium_dim)).to(x_histo.device)

        assert len(x_histo) == len(x_visium), (
            "Histo and Visium features need to have matched tiles <-> spots, "
            f"but received differing numbers of instances ({len(x_histo)}) and "
            f"{len(x_visium)} respectively)."
        )

        # Initial embeddings
        instance_embeddings_histo = self.tiles_mlp_histo(x_histo)
        instance_embeddings_visium = self.tiles_mlp_visium(x_visium)
        if self.infer_with_histo:
            instance_embeddings_fusion = torch.cat(
                (instance_embeddings_histo, instance_embeddings_histo), 2
            )
        else:
            instance_embeddings_fusion = torch.cat(
                (instance_embeddings_histo, instance_embeddings_visium), 2
            )

        # Gated attention network
        fusion_embeddings, _ = self.gated_attention(
            instance_embeddings_fusion,
            x_histo_mask,  # same number of tiles and spots so equivalent
        )

        logits = self.out_mlp(fusion_embeddings)

        # format the output
        predictions = {self.task: logits}
        extras: dict = {}

        return predictions, extras
