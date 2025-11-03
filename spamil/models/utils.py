"""Utils for MIL models."""

import classic_algos.nn
import torch
import torch.nn
import torch.nn.functional as F
import torch.nn.modules.loss


class CoxLoss(torch.nn.modules.loss._Loss):
    """Use Cox Loss implemented in PyTorch.

    There should not be any zero value (because we could not determine censure).

    Parameters
    ----------
    reduction: str
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """

    def __init__(self, task, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.task = task

    def forward(self, predictions, batch, extras):  # pylint: disable = arguments-differ
        """Compute CoxLoss.

        Parameters
        ----------
        input: torch.Tensor
            risk prediction from the model (higher score means higher risk and lower survival)
        target: torch.Tensor
            labels of the event occurences. Negative values are the censored values.

        Returns
        -------
        cox_loss: torch.Tensor
        """
        del extras
        loss = F.cox(predictions[self.task], batch["labels"][self.task], reduction=self.reduction)
        info = {"loss": loss}
        return loss, info


def loss_factory(loss_name: str, **kwargs) -> torch.nn.modules.loss._Loss:
    """Create a loss function.

    Parameters
    ----------
    loss_name : str
        Name of the loss to create.
        * "bce_with_logits_loss": torch.nn.BCEWithLogitsLoss
        * "mse_loss": torch.nn.MSELoss
        * "cross_entropy_loss": torch.nn.CrossEntropyLoss
        * "cox_loss": classic_algos.nn.CoxLoss
        * "smooth_c_index_loss": classic_algos.nn.SmoothCindexLoss
    **kwargs
        Additional parameters for the loss function.

    Returns
    -------
    torch.nn.modules.loss._Loss
        The loss function.

    Raises
    ------
    ValueError
        If the loss name is invalid.
    """
    if loss_name == "bce_with_logits_loss":
        return torch.nn.BCEWithLogitsLoss(**kwargs)
    if loss_name == "mse_loss":
        return torch.nn.MSELoss(**kwargs)
    if loss_name == "cross_entropy_loss":
        return torch.nn.CrossEntropyLoss(**kwargs)
    if loss_name == "cox_loss":
        return classic_algos.nn.CoxLoss(**kwargs)
    if loss_name == "smooth_c_index_loss":
        return classic_algos.nn.SmoothCindexLoss(**kwargs)
    raise ValueError(f"Invalid loss: {loss_name}")


def attention_loss_factory(loss_name: str, **kwargs) -> torch.nn.modules.loss._Loss:
    """Create a loss function for the attention weights.

    Parameters
    ----------
    loss_name : str
        Name of the loss to create.
        * "mse_loss": torch.nn.MSELoss
        * "symmetrical_kl": custom symmetrical kl divergence loss
        * "cosine_similarity": custom symmetrical cosine similarity loss
    **kwargs
        Additional parameters for the loss function.

    Returns
    -------
    torch.nn.modules.loss._Loss
        The loss function.

    Raises
    ------
    ValueError
        If the loss name is invalid.
    """
    if loss_name == "mse_loss":
        return torch.nn.MSELoss(**kwargs)
    if loss_name == "symmetrical_kl":

        class SymmetricKLLoss(torch.nn.modules.loss._Loss):
            def __init__(self, reduction="mean", eps=1e-8):
                super().__init__(reduction=reduction)
                self.eps = eps

            def forward(self, attention_weights_a, attention_weights_b):
                attention_weights_a = attention_weights_a + self.eps
                attention_weights_b = attention_weights_b + self.eps
                kl_ab = torch.sum(
                    attention_weights_a
                    * torch.log(attention_weights_a / attention_weights_b),
                    dim=1,
                )
                kl_ba = torch.sum(
                    attention_weights_b
                    * torch.log(attention_weights_b / attention_weights_a),
                    dim=1,
                )
                loss = 0.5 * (kl_ab + kl_ba)
                if self.reduction == "mean":
                    return loss.mean()
                elif self.reduction == "sum":
                    return loss.sum()
                else:
                    return loss

        return SymmetricKLLoss(**kwargs)
    if loss_name == "cosine_similarity":

        class CosineSimilarityLoss(torch.nn.modules.loss._Loss):
            def __init__(self, reduction="mean", eps=1e-8):
                super().__init__(reduction=reduction)
                self.eps = eps

            def forward(self, attention_weights_a, attention_weights_b):
                attention_weights_a = F.normalize(
                    attention_weights_a, p=2, dim=1, eps=self.eps
                )
                attention_weights_b = F.normalize(
                    attention_weights_b, p=2, dim=1, eps=self.eps
                )
                cos_sim = torch.sum(attention_weights_a * attention_weights_b, dim=1)
                loss = 1 - cos_sim
                if self.reduction == "mean":
                    return loss.mean()
                elif self.reduction == "sum":
                    return loss.sum()
                else:
                    return loss

        return CosineSimilarityLoss(**kwargs)
    raise ValueError(f"Invalid attention loss: {loss_name}")
