#!/usr/bin/env python3
"""Visualize attention weights from MultiHeadAttention."""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from attention import MultiHeadAttention


def create_token_embeddings(tokens: list[str], embed_dim: int) -> torch.Tensor:
    """Create simple embeddings for tokens (random but deterministic per token)"""
    embeddings = []
    for i, token in enumerate(tokens):
        torch.manual_seed(hash(token) % 2**32)
        embeddings.append(torch.randn(embed_dim))
    return torch.stack(embeddings).unsqueeze(0)  # (1, seq_len, embed_dim)


def analyze_attention_patterns(
    attn_weights: torch.Tensor, tokens: list[str]
) -> list[str]:
    """Analyze what patterns each head learned."""
    insights = []
    num_heads = attn_weights.shape[1]
    weights = attn_weights[0].detach().numpy()  # (num_heads, seq, seq)

    for head in range(num_heads):
        head_weights = weights[head]

        # Check for diagonal pattern (self-attention)
        diag_strength = np.trace(head_weights) / len(tokens)

        # Check for uniform attention
        uniformity = 1 - np.std(head_weights)

        # Check for positional patterns (attending to nearby tokens)
        local_strength = 0
        for i in range(len(tokens)):
            for j in range(max(0, i - 1), min(len(tokens), i + 2)):
                local_strength += head_weights[i, j]
        local_strength /= len(tokens) * 3

        # Check for attending to first/last token
        first_token_attn = head_weights[:, 0].mean()
        last_token_attn = head_weights[:, -1].mean()

        # Determine dominant pattern
        patterns = []
        if diag_strength > 0.3:
            patterns.append("self-focus")
        if local_strength > 0.5:
            patterns.append("local-context")
        if first_token_attn > 0.25:
            patterns.append("start-anchored")
        if last_token_attn > 0.25:
            patterns.append("end-anchored")
        if uniformity > 0.8:
            patterns.append("uniform-spread")

        if not patterns:
            # Find which token gets most attention overall
            most_attended = tokens[head_weights.sum(axis=0).argmax()]
            patterns.append(f"focuses-on-'{most_attended}'")

        insight = f"Head {head + 1}: {', '.join(patterns)}"
        insights.append(insight)

    return insights


def visualize_attention(
    tokens: list[str],
    embed_dim: int = 64,
    num_heads: int = 4,
    output_path: str = "attention_viz.png",
) -> None:
    """Create attention visualization and save to file."""

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("viridis")

    # Create model and get attention weights
    torch.manual_seed(42)
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    mha.eval()

    # Create embeddings
    x = create_token_embeddings(tokens, embed_dim)

    # Forward pass with attention weights
    with torch.no_grad():
        _, attn_weights = mha(x, x, x, need_weights=True)

    # attn_weights shape: (1, num_heads, seq_len, seq_len)
    weights = attn_weights[0].numpy()  # (num_heads, seq_len, seq_len)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Multi-Head Self-Attention Visualization",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Add subtitle with input
    fig.text(
        0.5,
        0.94,
        f'Input: "{" ".join(tokens)}"',
        ha="center",
        fontsize=12,
        style="italic",
        color="gray",
    )

    # Create grid: 2 rows, top row has individual heads, bottom row has combined
    gs = fig.add_gridspec(2, num_heads, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    # Plot individual attention heads (top row)
    head_colors = ["Blues", "Oranges", "Greens", "Purples"]
    for head_idx in range(num_heads):
        ax = fig.add_subplot(gs[0, head_idx])
        sns.heatmap(
            weights[head_idx],
            xticklabels=tokens,
            yticklabels=tokens if head_idx == 0 else False,
            cmap=head_colors[head_idx % len(head_colors)],
            vmin=0,
            vmax=1,
            square=True,
            cbar=head_idx == num_heads - 1,
            cbar_kws={"shrink": 0.8} if head_idx == num_heads - 1 else {},
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
        )
        ax.set_title(f"Head {head_idx + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Key (attending to)", fontsize=9)
        if head_idx == 0:
            ax.set_ylabel("Query (from)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    # Plot combined attention (bottom left) - average across heads
    ax_combined = fig.add_subplot(gs[1, :2])
    combined_weights = weights.mean(axis=0)
    sns.heatmap(
        combined_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.8},
        ax=ax_combined,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
    )
    ax_combined.set_title("Combined (Average of All Heads)", fontsize=11, fontweight="bold")
    ax_combined.set_xlabel("Key (attending to)", fontsize=9)
    ax_combined.set_ylabel("Query (from)", fontsize=9)

    # Plot attention entropy / distribution (bottom right)
    ax_analysis = fig.add_subplot(gs[1, 2:])

    # Create bar chart showing how much each token is attended to (by all heads)
    total_attention = weights.sum(axis=(0, 1))  # Sum over heads and queries
    bars = ax_analysis.barh(tokens[::-1], total_attention[::-1], color=sns.color_palette("viridis", len(tokens)))
    ax_analysis.set_xlabel("Total Attention Received", fontsize=9)
    ax_analysis.set_title("Token Importance (Attention Received)", fontsize=11, fontweight="bold")
    ax_analysis.tick_params(axis="both", labelsize=9)

    # Add value labels on bars
    for bar, val in zip(bars, total_attention[::-1]):
        ax_analysis.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Visualization saved to: {output_path}\n")

    # Print insights
    insights = analyze_attention_patterns(attn_weights, tokens)
    print("=" * 60)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 60)
    for insight in insights:
        print(f"  {insight}")
    print()

    # Print summary statistics
    print("Token Importance (total attention received):")
    for token, attn in zip(tokens, total_attention):
        bar = "█" * int(attn * 10)
        print(f"  {token:>10}: {bar} ({attn:.2f})")


if __name__ == "__main__":
    tokens = ["the", "quick", "brown", "fox", "jumps"]
    visualize_attention(tokens, embed_dim=64, num_heads=4)
