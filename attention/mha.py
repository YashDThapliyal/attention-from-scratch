import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights.
        bias: Whether to include bias in projection layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass for multi-head attention.

        Args:
            query: Query tensor of shape (batch, seq_len_q, embed_dim).
            key: Key tensor of shape (batch, seq_len_k, embed_dim).
            value: Value tensor of shape (batch, seq_len_k, embed_dim).
            attn_mask: Optional mask of shape (seq_len_q, seq_len_k) or
                (batch, num_heads, seq_len_q, seq_len_k). Use -inf for masked positions.
            need_weights: If True, return attention weights along with output.

        Returns:
            Output tensor of shape (batch, seq_len_q, embed_dim).
            If need_weights=True, also returns attention weights of shape
            (batch, num_heads, seq_len_q, seq_len_k).
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = self._scaled_dot_product_attention(
            q, k, v, attn_mask, need_weights=need_weights
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)

        output = self.out_proj(attn_output)

        if need_weights:
            return output, attn_weights
        return output

    def _scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Compute scaled dot-product attention.

        Args:
            query: Shape (batch, num_heads, seq_len_q, head_dim).
            key: Shape (batch, num_heads, seq_len_k, head_dim).
            value: Shape (batch, num_heads, seq_len_k, head_dim).
            attn_mask: Optional attention mask.
            need_weights: Whether to return attention weights.

        Returns:
            Tuple of (attention output, attention weights).
        """
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_out = attn_weights.clone() if need_weights else attn_weights
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, value), attn_weights_out
