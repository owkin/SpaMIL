"""Implement MabMIL (SpT-to-HE distillation) algorithm."""

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn
import torch.nn.modules.loss

from spamil.models.utils import loss_factory
from spamil.modules.attention import GatedAttention
from spamil.modules.mlp import MLP


class MabMILDistillation(torch.nn.Module):
    """Build MabMIL (SpT-to-HE distillation) model.

    Parameters
    ----------
    in_features: dict
        Dictionary containing the input dimension of each modality. This dict
        must have "histo" and "visium_counts" as keys.
        example: :code:`{"histo": 2048, "visium_counts": 3000}`.
    out_features: dict
        Dictionary mapping task to desired output shape, example: :code:`{"survival": 1}`.
    infer_with_histo: bool
        If true, inference will be done on histo data, otherwise it will be on visium.
    d_model_attention: int
        Size of the initial embedding output size before the attention layer.
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
    """

    def __init__(
        self,
        in_features: dict,
        out_features: dict,
        infer_with_histo: bool = False,
        d_model_attention: int = 128,
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
            out_features=d_model_attention,
            hidden=tiles_mlp_hidden_histo,
            activation=torch.nn.Sigmoid(),
            bias=bias,
        )
        self.visium_dim = in_features[self.visium_feature_name]
        self.tiles_mlp_visium = MLP(
            in_features=self.visium_dim,
            out_features=d_model_attention,
            hidden=tiles_mlp_hidden_visium,
            activation=torch.nn.Sigmoid(),
            bias=bias,
        )

        # Gated attention network
        self.gated_attention_histo = GatedAttention(
            d_model=d_model_attention, temperature=temperature
        )
        self.gated_attention_visium = GatedAttention(
            d_model=d_model_attention, temperature=temperature
        )

        # Output MLP
        self.out_mlp = MLP(
            in_features=d_model_attention,
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
        if self.training:
            x_histo, x_histo_mask = (
                batch["features"][self.histo_feature_name],
                batch["features"][f"{self.histo_feature_name}_collate_0"],
            )
            x_visium, x_visium_mask = (
                batch["features"][self.visium_feature_name],
                batch["features"][f"{self.visium_feature_name}_collate_0"],
            )
        elif self.infer_with_histo:
            # create dummy X_visium, X_visium_mask for inference
            x_histo, x_histo_mask = (
                batch["features"][self.histo_feature_name],
                batch["features"][f"{self.histo_feature_name}_collate_0"],
            )
            batch_dim = x_histo.shape[0]
            visium_dim = self.visium_dim
            x_visium = torch.zeros((batch_dim, 1, visium_dim)).to(x_histo.device)
            x_visium_mask = torch.zeros((batch_dim, 1, 1), dtype=torch.bool).to(
                x_histo.device
            )
        else:
            # create dummy X_histo, X_histo_mask for inference
            x_visium, x_visium_mask = (
                batch["features"][self.visium_feature_name],
                batch["features"][f"{self.visium_feature_name}_collate_0"],
            )
            batch_dim = x_visium.shape[0]
            histo_dim = self.histo_dim
            x_histo = torch.zeros((batch_dim, 1, histo_dim)).to(x_visium.device)
            x_histo_mask = torch.zeros((batch_dim, 1, 1), dtype=torch.bool).to(
                x_visium.device
            )

        # Initial embeddings
        instance_embeddings_histo = self.tiles_mlp_histo(x_histo)
        instance_embeddings_visium = self.tiles_mlp_visium(x_visium)

        # Gated attention network
        histo_embeddings, _ = self.gated_attention_histo(
            instance_embeddings_histo, x_histo_mask
        )
        visium_embeddings, _ = self.gated_attention_visium(
            instance_embeddings_visium, x_visium_mask
        )

        # Output MLP
        logits_from_histo = self.out_mlp(histo_embeddings)
        logits_from_visium = self.out_mlp(visium_embeddings)

        # format the output
        if self.infer_with_histo:
            predictions = {self.task: logits_from_histo}
            extras = {
                "histo_embeddings": histo_embeddings,
                "visium_embeddings": visium_embeddings,
                "logits_from_visium": logits_from_visium,
            }
        else:
            predictions = {self.task: logits_from_visium}
            extras = {
                "histo_embeddings": histo_embeddings,
                "visium_embeddings": visium_embeddings,
                "logits_from_histo": logits_from_histo,
            }

        return predictions, extras


class MabMILDistillationLoss(torch.nn.modules.loss._Loss):  # pylint: disable=W0212
    """Compute MabMILDistillation loss, sum of pred loss and MSE loss on the embeddings.

    Parameters
    ----------
    task: str
        Task name.
    prediction_loss: str
        Loss between predictions and labels. See docstring of loss_factory.
    train_with_histo: bool
        If true, the prediction loss will be computed on histo data, otherwise it will be on visium.
    detach_prediction_modality_branch: bool
        If True, the embedding loss will not be backpropagated inside the branch used for prediction.
    *args
        Additional positional arguments for the loss function.
    **kwargs
        Additional keyword arguments for the loss function.
    """

    def __init__(
        self,
        task: str,
        prediction_loss: str,
        train_with_histo: bool = True,
        detach_prediction_modality_branch: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task = task
        self.prediction_loss_fn = loss_factory(prediction_loss)
        self.train_with_histo = train_with_histo
        self.detach_prediction_modality_branch = detach_prediction_modality_branch

    def forward(self, predictions: dict, batch: dict, extras: dict) -> tuple[int, dict]:
        """Do forward pass.

        Parameters
        ----------
        predictions: dict
            a dict containing the predictions coming from Visium counts
        batch: dict
            a dict containing input data streams
        extras: dict
            a dict containing extra information, including the predictions coming from
            the histo data, which is how the prediction loss is computed

        Returns
        -------
        tuple[int, dict]
            total_loss
                the loss after the forward pass
            info
                extra information
        """
        labels = batch["labels"][self.task].float()
        if "logits_from_visium" in extras:
            visium_predictions = extras["logits_from_visium"]
            histo_predictions = predictions[self.task]
        else:
            visium_predictions = predictions[self.task]
            histo_predictions = extras["logits_from_histo"]
        histo_embeddings = extras["histo_embeddings"]
        visium_embeddings = extras["visium_embeddings"]

        if self.train_with_histo:
            # prediction loss (using histo)
            assert labels.shape == histo_predictions.shape, (
                "Logits and labels are not the same shape."
            )
            prediction_loss = self.prediction_loss_fn(histo_predictions, labels)
        else:
            # prediction loss (using visium)
            assert labels.shape == visium_predictions.shape, (
                "Logits and labels are not the same shape."
            )
            prediction_loss = self.prediction_loss_fn(visium_predictions, labels)

        # visium embedding loss
        assert histo_embeddings.shape == visium_embeddings.shape
        embedding_loss_fn = torch.nn.MSELoss()
        if self.detach_prediction_modality_branch:
            if self.train_with_histo:
                histo_embeddings = histo_embeddings.detach()
            else:
                visium_embeddings = visium_embeddings.detach()
        embedding_loss = embedding_loss_fn(histo_embeddings, visium_embeddings)

        # total loss
        total_loss = prediction_loss + embedding_loss

        info = {
            "prediction_loss": prediction_loss,
            "embedding_loss": embedding_loss,
            "total_loss": total_loss,
        }

        return total_loss, info
