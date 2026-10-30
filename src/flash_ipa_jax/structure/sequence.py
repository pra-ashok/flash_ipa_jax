"""The dropout-free, post-normalized sequence Transformer used upstream.

IPA exchanges information using scalar features and 3D points. These ordinary
sequence-attention layers provide a second, feature-only path between residues.
They compile through XLA on any JAX device; IPA's optional fused GPU backend is
configured separately.
"""

import math

import jax
import jax.numpy as jnp

from ..linear import init_linear, linear
from ..normalization import init_norm, layer_norm


class SequenceTransformer:
    """Equivalent operations to PyTorch's default TransformerEncoderLayer.

    The feed-forward width equals ``channels`` and dropout is zero, matching the
    upstream example. Each layer has its own parameters. ``mask`` marks valid
    nodes; fully padded examples remain finite and produce zero outputs.
    """

    def __init__(self, channels, heads=4, layers=2):
        if min(channels, heads, layers) < 1 or channels % heads:
            raise ValueError(
                "channels must be divisible by heads; sizes/layers must be positive"
            )
        self.channels, self.heads, self.layers = channels, heads, layers

    def init(self, key, *, dtype=jnp.float32):
        params = []
        for layer_key in jax.random.split(key, self.layers):
            qkv_key, out_key, ff1_key, ff2_key = jax.random.split(layer_key, 4)
            c = self.channels
            output = init_linear(out_key, c, c, init="torch", dtype=dtype)
            output["bias"] = jnp.zeros((c,), dtype)
            params.append(
                {
                    "qkv": init_linear(qkv_key, c, 3 * c, init="xavier", dtype=dtype),
                    "out": output,
                    "linear1": init_linear(ff1_key, c, c, init="torch", dtype=dtype),
                    "linear2": init_linear(ff2_key, c, c, init="torch", dtype=dtype),
                    "norm1": init_norm(c, dtype),
                    "norm2": init_norm(c, dtype),
                }
            )
        return tuple(params)

    def apply(self, params, nodes, mask=None):
        if nodes.ndim != 3 or nodes.shape[-1] != self.channels:
            raise ValueError("nodes must have shape [B,N,channels]")
        batch, length, _ = nodes.shape
        if mask is not None and mask.shape != (batch, length):
            raise ValueError("mask must have shape [B,N]")
        valid = jnp.ones((batch, length), bool) if mask is None else mask != 0
        nodes = jnp.where(valid[..., None], nodes, 0)
        width = self.channels // self.heads
        for layer in params:
            qkv = linear(layer["qkv"], nodes).reshape(
                batch, length, 3, self.heads, width
            )
            query, key, value = (qkv[:, :, index] for index in range(3))
            logits = jnp.einsum(
                "bihd,bjhd->bhij", query, key, precision=jax.lax.Precision.HIGHEST
            ) / math.sqrt(width)
            # A finite sentinel and explicit zeroing define all-masked rows
            # without NaNs in either the forward pass or its derivatives.
            allowed = valid[:, None, None, :]
            logits = jnp.where(allowed, logits, jnp.finfo(logits.dtype).min)
            weights = jax.nn.softmax(logits, -1) * allowed
            weights = weights / jnp.maximum(jnp.sum(weights, -1, keepdims=True), 1e-30)
            attention = jnp.einsum(
                "bhij,bjhd->bihd", weights, value, precision=jax.lax.Precision.HIGHEST
            ).reshape(batch, length, -1)
            nodes = layer_norm(layer["norm1"], nodes + linear(layer["out"], attention))
            hidden = jax.nn.relu(linear(layer["linear1"], nodes))
            nodes = layer_norm(layer["norm2"], nodes + linear(layer["linear2"], hidden))
            nodes = jnp.where(valid[..., None], nodes, 0)
        return nodes
