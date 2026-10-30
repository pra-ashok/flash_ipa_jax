"""NVIDIA Hopper adapter for the optional ``flash_attn3_jax`` package.

IPA packing happens before this boundary. This module owns the wheel's device,
head-width, function-signature, padding, and launch conventions. Optional
imports happen only after shared dispatch accepts the input and device.
"""

import importlib

import jax.numpy as jnp

from .common import pad_qkv, require_parameters


def eligibility(width, device):
    """Return hardware/width rejection reasons and the wheel's padded width."""
    reasons = []
    if "gpu" not in device or not any(
        name in device for name in ("h100", "h200", "gh200", "sm90")
    ):
        reasons.append("FA3 adapter requires a single NVIDIA Hopper GPU")
    if width > 128:
        reasons.append(
            f"expanded IPA dimension {width} exceeds FA3 wheel dimension 128"
        )
    return reasons, 128


def load_kernel():
    """Lazily load and check the supported public FA3 function signature."""
    kernel = importlib.import_module("flash_attn3_jax").flash_mha
    return require_parameters(kernel, {"softmax_scale", "is_causal"}, "flash_attn3")


def attention(q, k, v, *, config, decision, causal):
    """Launch FA3 without hiding import, compilation, or FFI runtime failures."""
    kernel = load_kernel()
    q, k, v = pad_qkv(
        q, k, v, padded_dim=decision.padded_dim, dtype=config.kernel_dtype
    )
    # Packing includes every IPA scale; the kernel's default would change it.
    result = kernel(q, k, v, softmax_scale=1.0, is_causal=causal)
    return result.astype(jnp.float32)
