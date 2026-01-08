#!/usr/bin/env python3
"""Benchmark comparing custom MultiHeadAttention vs torch.nn.MultiheadAttention."""

import gc
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from attention import MultiHeadAttention


@dataclass
class BenchmarkResult:
    name: str
    forward_time_ms: float
    memory_mb: float
    num_runs: int


def measure_memory_mb() -> float:
    """Get current GPU memory allocated in MB, or 0 if CPU."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_forward(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    num_warmup: int = 10,
    num_runs: int = 100,
) -> BenchmarkResult:
    """Benchmark forward pass speed and memory."""
    module.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = module(*inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure memory before
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_before = measure_memory_mb()

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = module(*inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem_after = measure_memory_mb()
    memory_used = max(0, mem_after - mem_before)

    return BenchmarkResult(
        name=module.__class__.__name__,
        forward_time_ms=(elapsed / num_runs) * 1000,
        memory_mb=memory_used,
        num_runs=num_runs,
    )


class TorchMHAWrapper(nn.Module):
    """Wrapper to match our API (batch_first, separate Q/K/V)."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        output, _ = self.mha(query, key, value)
        return output


def compare_numerical_accuracy(
    custom: MultiHeadAttention,
    torch_mha: nn.MultiheadAttention,
    x: torch.Tensor,
) -> dict[str, float]:
    """Compare numerical accuracy between implementations."""
    custom.eval()
    torch_mha.eval()

    with torch.no_grad():
        custom_out = custom(x, x, x)
        torch_out, _ = torch_mha(x, x, x)

    diff = (custom_out - torch_out).abs()

    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine_sim": torch.nn.functional.cosine_similarity(
            custom_out.flatten(), torch_out.flatten(), dim=0
        ).item(),
    }


def copy_weights(src: MultiHeadAttention, dst: nn.MultiheadAttention) -> None:
    """Copy weights from custom MHA to torch MHA for fair comparison."""
    with torch.no_grad():
        # PyTorch combines Q, K, V into in_proj_weight
        dst.in_proj_weight.copy_(
            torch.cat([src.q_proj.weight, src.k_proj.weight, src.v_proj.weight], dim=0)
        )
        if dst.in_proj_bias is not None:
            dst.in_proj_bias.copy_(
                torch.cat([src.q_proj.bias, src.k_proj.bias, src.v_proj.bias], dim=0)
            )
        dst.out_proj.weight.copy_(src.out_proj.weight)
        if dst.out_proj.bias is not None:
            dst.out_proj.bias.copy_(src.out_proj.bias)


def run_benchmarks() -> None:
    """Run all benchmarks and print results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    configs = [
        {"batch": 8, "seq": 128, "embed": 256, "heads": 8},
        {"batch": 8, "seq": 512, "embed": 512, "heads": 8},
        {"batch": 4, "seq": 1024, "embed": 768, "heads": 12},
        {"batch": 2, "seq": 2048, "embed": 1024, "heads": 16},
    ]

    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for cfg in configs:
        batch, seq, embed, heads = cfg["batch"], cfg["seq"], cfg["embed"], cfg["heads"]
        print(f"\n Config: batch={batch}, seq_len={seq}, embed_dim={embed}, heads={heads}")
        print("-" * 70)

        # Create models
        custom_mha = MultiHeadAttention(embed, heads).to(device)
        torch_mha = nn.MultiheadAttention(embed, heads, batch_first=True).to(device)

        # Copy weights for fair numerical comparison
        copy_weights(custom_mha, torch_mha)

        # Create input
        x = torch.randn(batch, seq, embed, device=device)

        # Benchmark custom implementation
        custom_result = benchmark_forward(custom_mha, (x, x, x))

        # Benchmark PyTorch implementation (needs wrapper for same API)
        torch_wrapper = TorchMHAWrapper(embed, heads).to(device)
        torch_wrapper.mha = torch_mha
        torch_result = benchmark_forward(torch_wrapper, (x, x, x))

        # Numerical accuracy (with same weights)
        accuracy = compare_numerical_accuracy(custom_mha, torch_mha, x)

        # Print results
        print(f"  {'Metric':<25} {'Custom MHA':>15} {'PyTorch MHA':>15} {'Ratio':>10}")
        print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")

        ratio = custom_result.forward_time_ms / torch_result.forward_time_ms
        print(
            f"  {'Forward time (ms)':<25} "
            f"{custom_result.forward_time_ms:>15.3f} "
            f"{torch_result.forward_time_ms:>15.3f} "
            f"{ratio:>10.2f}x"
        )

        if device.type == "cuda":
            print(
                f"  {'Memory (MB)':<25} "
                f"{custom_result.memory_mb:>15.2f} "
                f"{torch_result.memory_mb:>15.2f}"
            )

        print(f"\n  Numerical Accuracy (same weights):")
        print(f"    Max absolute diff:  {accuracy['max_abs_diff']:.2e}")
        print(f"    Mean absolute diff: {accuracy['mean_abs_diff']:.2e}")
        print(f"    Cosine similarity:  {accuracy['cosine_sim']:.6f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("  - Ratio > 1.0 means custom is slower than PyTorch")
    print("  - Ratio < 1.0 means custom is faster than PyTorch")
    print("  - Cosine similarity close to 1.0 indicates numerical equivalence")
    print("  - Small numerical differences are expected due to operation ordering")


if __name__ == "__main__":
    run_benchmarks()
