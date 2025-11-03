"""Multi-layer perceptron algorithm."""

from typing import Optional, Sequence

import torch
import torch.nn

import spamil.modules.mlp


class MLP(torch.nn.Module):
    """MLP for multimodal data and multi-task problems.

    Modalities are concatenated at the input.

    Parameters
    ----------
    in_features: dict[str, int]
        Dictionnary containing the input dimension of each modality.
    out_features: dict[str, int]
        Dictionnary containing the output dimension of each task.
    hidden: Sequence[int]
        Size of hidden layers.
    activation: torch.nn.Module
        Activation function.
    last_activation: Optional[torch.nn.Module]
        activation function used for the last layer. If None, no activation is performed
         for the last layer.
    enabled_modalities: list[str], default = None
        List of modalities to use as input.
        If None, all modalities present in the input batch are concatenated.
    """

    def __init__(
        self,
        in_features: dict[str, int],
        out_features: dict[str, int],
        hidden: Sequence[int] = (256, 128),
        activation: torch.nn.Module = torch.nn.Sigmoid(),
        last_activation: Optional[torch.nn.Module] = None,
        enabled_modalities: Optional[list[str]] = None,
    ):
        super().__init__()
        self.enabled_modalities = enabled_modalities
        if self.enabled_modalities:
            input_dim = sum(  # pylint: disable=R1728
                [value for modality, value in in_features.items() if modality in self.enabled_modalities]
            )
        else:
            input_dim = sum(  # pylint: disable=R1728
                [value for key, value in in_features.items() if "collate" not in key]
            )  # pylint: disable=R1728
        self.tasks = sorted(out_features.keys())

        self.common_mlp = spamil.modules.mlp.MLP(
            in_features=input_dim,
            out_features=hidden[-1],
            hidden=hidden[:-1],
            activation=activation,
        )
        self.output_heads = torch.nn.ModuleDict(
            {task: torch.nn.Linear(in_features=hidden[-1], out_features=out_features[task]) for task in self.tasks}
        )
        self.last_activation = last_activation

    def forward(self, batch):
        """Apply model to batch."""
        features = batch["features"]
        if self.enabled_modalities:
            x = torch.cat(
                [features[key].float() for key in features if key in self.enabled_modalities],
                axis=1,
            )
        else:
            x = torch.cat(
                [features[key].float() for key in features if "collate" not in key],
                axis=1,
            )
        if self.last_activation is not None:
            predictions = {
                task: self.last_activation(self.output_heads[task](self.common_mlp(x))) for task in self.tasks
            }
        else:
            predictions = {task: self.output_heads[task](self.common_mlp(x)) for task in self.tasks}
        extras = {}
        return predictions, extras
