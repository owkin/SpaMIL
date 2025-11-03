"""Gated attention module for abMIL."""

from typing import Optional, Tuple

import torch

from spamil.modules.tile_layers import MaskedLinear


class GatedAttention(torch.nn.Module):
    """Gated Attention, as defined in https://arxiv.org/abs/1802.04712.

    Permutation invariant Layer on dim 1.

    Parameters
    ----------
    d_model: int = 128
    d_hidden: Optional[int] = None
        Hidden dimension, if provided
    temperature: float = 1.0
        Attention Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 128,
        d_hidden: Optional[int] = None,
        temperature: float = 1.0,
    ):
        super(GatedAttention, self).__init__()
        if d_hidden is None:
            d_hidden = d_model

        self.att = torch.nn.Linear(d_model, d_hidden)
        self.gate = torch.nn.Linear(d_model, d_hidden)
        self.w = MaskedLinear(d_hidden, 1, "-inf")

        self.temperature = temperature

    def attention(
        self,
        v: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Get attention logits.

        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.

        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        h_v = self.att(v)
        h_v = torch.tanh(h_v)

        u_v = self.gate(v)
        u_v = torch.sigmoid(u_v)

        attention_logits = self.w(h_v * u_v, mask=mask) / self.temperature
        return attention_logits

    def forward(
        self, v: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply module to batch."""
        attention_logits = self.attention(v=v, mask=mask)

        attention_weights = torch.softmax(attention_logits, 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights
