"""Learned compression of an existing dense pair representation.

This is the JAX counterpart of upstream ``LinearFactorizer``. It is a learned
projection, not an SVD: converting dense edges to factors does not promise an
exact reconstruction, and still requires constructing dense edges first.
"""

import math

import jax
import jax.numpy as jnp

from .linear import init_linear, linear
from .normalization import init_norm, layer_norm


class LinearFactorizer:
    """Compress ``[B,N,N,D]`` edges into two ``[B*D_out,N,rank]`` factors.

    ``in_L`` is the largest supported sequence length. Shorter sequences are
    zero padded before the learned length projection, matching upstream.
    """

    def __init__(self, in_L, in_D, target_rank=4, target_inner_dim=8):
        if min(in_L, in_D, target_rank, target_inner_dim) < 1:
            raise ValueError("factorizer dimensions must be positive")
        self.in_L, self.in_D = in_L, in_D
        self.target_rank, self.target_inner_dim = target_rank, target_inner_dim

    def init(self, key, *, dtype=jnp.float32):
        length_key, inner_key = jax.random.split(key)
        return {
            "length_compressor": init_linear(
                length_key,
                self.in_L,
                self.target_rank,
                init="torch",
                bias=False,
                dtype=dtype,
            ),
            "inner_compressor": init_linear(
                inner_key,
                self.in_D,
                self.target_inner_dim,
                init="torch",
                bias=False,
                dtype=dtype,
            ),
            "length_norm": init_norm(self.target_rank, dtype),
            "inner_norm": init_norm(self.target_inner_dim, dtype),
        }

    def apply(self, params, x, mask=None):
        if x.ndim != 4 or x.shape[1] != x.shape[2] or x.shape[-1] != self.in_D:
            raise ValueError("x must have shape [B,N,N,in_D]")
        batch, length, _, _ = x.shape
        if length > self.in_L:
            raise ValueError("sequence length exceeds factorizer in_L")
        if mask is not None:
            if mask.shape != (batch, length):
                raise ValueError("mask must have shape [B,N]")
            # Removing padded keys before compression prevents them influencing
            # valid nodes. This also supports masks with holes, not just suffixes.
            x = jnp.where(
                (mask[:, :, None] != 0)[..., None] & (mask[:, None, :] != 0)[..., None],
                x,
                0,
            )
        x = jnp.pad(
            x, ((0, 0), (0, self.in_L - length), (0, self.in_L - length), (0, 0))
        )
        row = linear(params["length_compressor"], jnp.transpose(x, (0, 2, 3, 1)))[
            :, :length
        ]
        col = linear(params["length_compressor"], jnp.transpose(x, (0, 1, 3, 2)))[
            :, :length
        ]

        def finish(value):
            value = layer_norm(params["length_norm"], value)
            value = linear(params["inner_compressor"], jnp.swapaxes(value, -1, -2))
            value = layer_norm(params["inner_norm"], value)
            if mask is not None:
                value = jnp.where(mask[:, :, None, None] != 0, value, 0)
            # The upstream interface merges batch and channel; EdgeEmbedder
            # restores [B,N,rank,D] before passing factors to IPA.
            return jnp.transpose(value, (0, 3, 1, 2)).reshape(
                batch * self.target_inner_dim, length, self.target_rank
            ) / math.sqrt(self.target_rank)

        return finish(row), finish(col)
