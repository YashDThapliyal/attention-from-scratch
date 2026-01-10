# Attention from Scratch

A minimal PyTorch implementation of multi-head self-attention.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Basic Self-Attention

```python
import torch
from attention import MultiHeadAttention

# Create model
mha = MultiHeadAttention(embed_dim=512, num_heads=8)

# Self-attention: query = key = value
x = torch.randn(2, 10, 512)  # (batch, seq_len, embed_dim)
output = mha(x, x, x)        # (2, 10, 512)
```

### Cross-Attention

```python
query = torch.randn(2, 5, 512)   # (batch, seq_q, embed_dim)
key = torch.randn(2, 20, 512)    # (batch, seq_k, embed_dim)
value = torch.randn(2, 20, 512)  # (batch, seq_k, embed_dim)

output = mha(query, key, value)  # (2, 5, 512)
```

### Causal Masking (for autoregressive models)

```python
seq_len = 10
causal_mask = torch.triu(
    torch.full((seq_len, seq_len), float("-inf")), diagonal=1
)
output = mha(x, x, x, attn_mask=causal_mask)
```

### Get Attention Weights

```python
output, attn_weights = mha(x, x, x, need_weights=True)
# attn_weights: (batch, num_heads, seq_q, seq_k)
```

## API Reference

```python
MultiHeadAttention(
    embed_dim: int,      # Total embedding dimension
    num_heads: int,      # Number of attention heads
    dropout: float = 0.0,
    bias: bool = True,
)

forward(
    query: Tensor,               # (batch, seq_q, embed_dim)
    key: Tensor,                 # (batch, seq_k, embed_dim)
    value: Tensor,               # (batch, seq_k, embed_dim)
    attn_mask: Tensor = None,    # Optional mask
    need_weights: bool = False,  # Return attention weights
)
```

## Scripts

### Run Tests

```bash
pytest tests/ -v
```

### Benchmark vs PyTorch

```bash
python benchmark.py
```

### Visualize Attention

```bash
python visualize.py
```

Generates `attention_viz.png` with attention heatmaps.

## Project Structure

```
attention-from-scratch/
├── attention/
│   ├── __init__.py
│   └── mha.py           # MultiHeadAttention implementation
├── tests/
│   └── test_mha.py      # Unit tests
├── benchmark.py         # Performance comparison
├── visualize.py         # Attention visualization
└── pyproject.toml
```
