import pytest
import torch

from attention import MultiHeadAttention


class TestMultiHeadAttention:
    def test_output_shape_self_attention(self) -> None:
        """Test that self-attention produces correct output shape."""
        batch_size, seq_len, embed_dim, num_heads = 2, 10, 64, 8
        mha = MultiHeadAttention(embed_dim, num_heads)

        x = torch.randn(batch_size, seq_len, embed_dim)
        output = mha(x, x, x)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_output_shape_cross_attention(self) -> None:
        """Test cross-attention with different query and key/value lengths."""
        batch_size, seq_len_q, seq_len_kv, embed_dim, num_heads = 2, 5, 10, 64, 8
        mha = MultiHeadAttention(embed_dim, num_heads)

        query = torch.randn(batch_size, seq_len_q, embed_dim)
        key = torch.randn(batch_size, seq_len_kv, embed_dim)
        value = torch.randn(batch_size, seq_len_kv, embed_dim)

        output = mha(query, key, value)

        assert output.shape == (batch_size, seq_len_q, embed_dim)

    def test_attention_mask(self) -> None:
        """Test that attention mask properly blocks positions."""
        batch_size, seq_len, embed_dim, num_heads = 1, 4, 32, 4
        mha = MultiHeadAttention(embed_dim, num_heads)

        x = torch.randn(batch_size, seq_len, embed_dim)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )

        output_masked = mha(x, x, x, attn_mask=causal_mask)

        assert output_masked.shape == (batch_size, seq_len, embed_dim)
        assert not torch.isnan(output_masked).any()

    def test_gradient_flow(self) -> None:
        """Test that gradients flow correctly through the module."""
        batch_size, seq_len, embed_dim, num_heads = 2, 8, 64, 8
        mha = MultiHeadAttention(embed_dim, num_heads)

        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        output = mha(x, x, x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        for param in mha.parameters():
            assert param.grad is not None

    def test_invalid_embed_dim(self) -> None:
        """Test that invalid embed_dim raises ValueError."""
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadAttention(embed_dim=65, num_heads=8)

    def test_deterministic_without_dropout(self) -> None:
        """Test that output is deterministic when dropout is 0."""
        batch_size, seq_len, embed_dim, num_heads = 2, 8, 64, 8
        mha = MultiHeadAttention(embed_dim, num_heads, dropout=0.0)
        mha.eval()

        x = torch.randn(batch_size, seq_len, embed_dim)
        output1 = mha(x, x, x)
        output2 = mha(x, x, x)

        assert torch.allclose(output1, output2)

    def test_different_batch_sizes(self) -> None:
        """Test that the module handles various batch sizes."""
        embed_dim, num_heads = 64, 8
        mha = MultiHeadAttention(embed_dim, num_heads)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 10, embed_dim)
            output = mha(x, x, x)
            assert output.shape == (batch_size, 10, embed_dim)
