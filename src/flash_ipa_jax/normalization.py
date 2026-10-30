"""Functional LayerNorm parameters and application, shared by model components.

The returned dictionaries are ordinary parameter pytrees for JAX optimizers.
"""

import jax
import jax.numpy as jnp


def init_norm(channels, dtype=jnp.float32):
    """Start with unit feature scales and zero feature offsets."""
    return {
        "weight": jnp.ones((channels,), dtype),
        "bias": jnp.zeros((channels,), dtype),
    }


def layer_norm(params, x, eps=1e-5):
    """LayerNorm with population variance, as used by the upstream model."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * params["weight"] + params[
        "bias"
    ]
