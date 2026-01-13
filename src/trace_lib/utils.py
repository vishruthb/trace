"""
Utility functions for trace.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn


def get_model_info(model: nn.Module) -> dict:
    """
    Extract model information for reporting.

    Args:
        model: PyTorch model

    Returns:
        Dict with model info
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get dtypes
    dtypes = set()
    for p in model.parameters():
        dtypes.add(str(p.dtype))

    # Memory estimate
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    total_memory_mb = (param_memory + buffer_memory) / (1024 * 1024)

    # Get config if available
    config_dict = {}
    if hasattr(model, "config"):
        config = model.config
        config_dict = {
            "hidden_size": getattr(config, "hidden_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "model_type": getattr(config, "model_type", None),
        }

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dtypes": list(dtypes),
        "memory_mb": total_memory_mb,
        "config": config_dict,
    }


def estimate_quantized_memory(
    model: nn.Module,
    bits: int = 8,
    group_size: Optional[int] = None,
) -> float:
    """
    Estimate memory usage after quantization.

    Args:
        model: Model to estimate for
        bits: Target bit width
        group_size: Group size for grouped quantization

    Returns:
        Estimated memory in MB
    """
    total_params = sum(p.numel() for p in model.parameters())

    # Basic estimate: params * bits / 8 bytes
    param_bytes = total_params * bits / 8

    # Add scale/zero-point overhead for grouped quantization
    if group_size:
        num_groups = math.ceil(total_params / group_size)
        # 2 bytes for scale, 1 byte for zero-point per group
        overhead = num_groups * 3
        param_bytes += overhead

    return param_bytes / (1024 * 1024)


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
) -> tuple[torch.Tensor, float, int]:
    """
    Quantize a tensor to specified bit width.

    This implements the quantization function from the design doc:
    Q(x) = clip(round(x/s) + z, 0, 2^n - 1)

    Args:
        tensor: Tensor to quantize
        bits: Target bit width
        symmetric: Whether to use symmetric quantization

    Returns:
        Tuple of (quantized_tensor, scale, zero_point)
    """
    qmin = 0
    qmax = 2**bits - 1

    if symmetric:
        # Symmetric quantization: zero_point = 2^(n-1)
        abs_max = tensor.abs().max()
        scale = abs_max / (qmax / 2)
        zero_point = qmax // 2
    else:
        # Asymmetric quantization
        t_min = tensor.min()
        t_max = tensor.max()
        scale = (t_max - t_min) / (qmax - qmin)
        zero_point = qmin - round(t_min / scale)

    # Avoid division by zero
    if scale == 0:
        scale = 1.0

    # Quantize
    quantized = torch.clamp(
        torch.round(tensor / scale) + zero_point,
        qmin,
        qmax,
    ).to(torch.uint8 if bits == 8 else torch.int32)

    return quantized, scale, zero_point


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: float,
    zero_point: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize a tensor back to floating point.

    Args:
        quantized: Quantized tensor
        scale: Quantization scale
        zero_point: Quantization zero point
        dtype: Target dtype

    Returns:
        Dequantized tensor
    """
    return ((quantized.float() - zero_point) * scale).to(dtype)


def compute_quantization_error(
    tensor: torch.Tensor,
    bits: int = 8,
) -> dict:
    """
    Compute various error metrics for quantization.

    Args:
        tensor: Original tensor
        bits: Quantization bit width

    Returns:
        Dict with error metrics
    """
    quantized, scale, zp = quantize_tensor(tensor, bits)
    reconstructed = dequantize_tensor(quantized, scale, zp, tensor.dtype)

    error = tensor - reconstructed

    return {
        "mse": (error**2).mean().item(),
        "mae": error.abs().mean().item(),
        "max_error": error.abs().max().item(),
        "relative_error": (error.abs() / (tensor.abs() + 1e-8)).mean().item(),
        "snr_db": 10 * math.log10((tensor**2).mean() / ((error**2).mean() + 1e-8)),
    }


def cosine_similarity_batched(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute cosine similarity between batched tensors.

    Args:
        x: First tensor (..., d)
        y: Second tensor (..., d)
        eps: Epsilon for numerical stability

    Returns:
        Cosine similarity tensor (...)
    """
    x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)
    y_norm = y / (y.norm(dim=-1, keepdim=True) + eps)
    return (x_norm * y_norm).sum(dim=-1)


def format_number(n: Union[int, float], precision: int = 2) -> str:
    """Format a number with appropriate suffix (K, M, B)."""
    if abs(n) >= 1e9:
        return f"{n/1e9:.{precision}f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
