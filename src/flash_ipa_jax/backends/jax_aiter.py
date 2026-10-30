"""AMD ROCm adapter for the optional ``jax_aiter.mha`` package.

The inspected AITER contract targets gfx950/MI355X. Its packed attention call
uses zero dropout, an explicit causal flag, and no log-sum-exp output. This is
a JAX FFI adapter; it does not convert arrays to PyTorch.
"""

import importlib

import jax.numpy as jnp

from .common import pad_qkv, require_parameters


def eligibility(width, device):
    """Return hardware/width rejection reasons and the smallest supported width."""
    reasons = []
    if "gpu" not in device or not any(name in device for name in ("gfx950", "mi355")):
        reasons.append("AITER adapter targets the inspected gfx950/MI355X stack")
    padded = next((size for size in (64, 128, 256) if size >= width), None)
    if padded is None:
        reasons.append(f"expanded IPA dimension {width} exceeds 256")
    return reasons, padded


def load_kernel():
    """Lazily load and check the supported public AITER function signature."""
    kernel = importlib.import_module("jax_aiter.mha").flash_attn_func
    required = {"softmax_scale", "causal", "dropout_p", "return_lse"}
    return require_parameters(kernel, required, "jax_aiter")


def attention(q, k, v, *, config, decision, causal):
    """Launch AITER, preserving its API conventions and surfacing FFI failures."""
    kernel = load_kernel()
    q, k, v = pad_qkv(
        q, k, v, padded_dim=decision.padded_dim, dtype=config.kernel_dtype
    )
    result = kernel(
        q, k, v, dropout_p=0.0, softmax_scale=1.0, causal=causal, return_lse=False
    )
    # Inspected AITER source returns (out,), despite its tensor-only docstring.
    if isinstance(result, (tuple, list)):
        result = result[0]
    return result.astype(jnp.float32)
