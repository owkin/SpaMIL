"""Multi-layer perceptron submodule for MIL models."""

from typing import List, Optional

import torch


class MLP(torch.nn.Sequential):
    """MLP Module.

    Parameters
    ----------
    in_features: int
    out_features: int
    hidden: Optional[List[int]] = None
    dropout: Optional[List[float]] = None,
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):

        if dropout is not None:
            if hidden is not None:
                assert len(hidden) == len(
                    dropout
                ), "hidden and dropout must have the same length"
            else:
                raise ValueError(
                    "hidden must have a value and have the same length as dropout if dropout is given."
                )

        d_model = in_features
        layers = []

        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [torch.nn.Linear(d_model, h, bias=bias)]
                d_model = h

                if activation is not None:
                    seq.append(activation)

                if dropout is not None:
                    seq.append(torch.nn.Dropout(dropout[i]))

                layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)
