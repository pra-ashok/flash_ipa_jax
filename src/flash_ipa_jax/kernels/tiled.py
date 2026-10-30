"""Portable FlashAttention-style IPA using query/key tiles and online softmax.

The running maximum, denominator, and weighted values replace a full N x N
attention matrix. This is the FlashAttention algorithm expressed in JAX/XLA;
it does not promise the SRAM placement or speed of a hand-tuned GPU kernel.
Checkpointing recomputes tiles during reverse-mode differentiation.
"""

import math

import jax
import jax.numpy as jnp

from .common import point_weights
from .types import IPAOutput


def tiled_attention(
    q,
    k,
    v,
    qp,
    kp,
    vp,
    raw,
    factors,
    pair_bias,
    pair_values,
    mask,
    pair_mask,
    causal,
    config,
):
    """Compute all IPA terms, including arbitrary dense pairs and exact masks.

    Inputs are validated by the public attention function. Sequence tails use
    clamped gathers plus an explicit mask, so tile sizes need not divide N.
    The accumulators stay in the public computation dtype (at least float32).
    """
    batch, length, heads, scalar_width = q.shape
    query_size = min(config.chunk_size, length)
    key_size = min(config.key_chunk_size, length)
    node_mask = jnp.ones((batch, length), bool) if mask is None else mask
    weights = point_weights(raw, qp.shape[-2])
    precision = jax.lax.Precision.HIGHEST
    pair_channels = (
        factors.value_q.shape[-1]
        if factors is not None
        else pair_values.shape[-1] if pair_values is not None else 0
    )

    query_tiles = jnp.arange(math.ceil(length / query_size) * query_size)
    query_tiles = query_tiles.reshape(-1, query_size)
    key_tiles = jnp.arange(math.ceil(length / key_size) * key_size)
    key_tiles = key_tiles.reshape(-1, key_size)

    def query_tile(query_ids):
        query_index = jnp.minimum(query_ids, length - 1)
        query = q[:, query_index]
        query_points = qp[:, query_index]
        row_shape = (batch, heads, query_size)

        # Each key tile updates the same row statistics and value sums.
        initial = (
            jnp.full(row_shape, -jnp.inf, q.dtype),
            jnp.zeros(row_shape, q.dtype),
            jnp.zeros((*row_shape, v.shape[-1]), q.dtype),
            jnp.zeros((*row_shape, vp.shape[-2], 3), q.dtype),
            jnp.zeros((*row_shape, pair_channels), q.dtype),
        )

        def key_tile(state, key_ids):
            maximum, denominator, scalar_sum, point_sum, pair_sum = state
            key_index = jnp.minimum(key_ids, length - 1)

            # IPA combines feature similarity, 3D distance, and pair context.
            logits = jnp.einsum(
                "bihd,bjhd->bhij", query, k[:, key_index], precision=precision
            ) / math.sqrt(3 * scalar_width)
            delta = query_points[:, :, None] - kp[:, None, key_index]
            distance = jnp.sum(delta * delta, axis=(-1, -2))
            logits -= 0.5 * jnp.einsum("bijh,h->bhij", distance, weights)
            if factors is not None:
                logits += jnp.einsum(
                    "bihr,bjhr->bhij",
                    factors.bias_q[:, query_index],
                    factors.bias_k[:, key_index],
                    precision=precision,
                )
            if pair_bias is not None:
                bias_tile = pair_bias[:, query_index[:, None], key_index[None, :]]
                logits += jnp.transpose(bias_tile, (0, 3, 1, 2))

            keep = (
                node_mask[:, query_index, None]
                & node_mask[:, None, key_index]
                & (query_ids[None, :, None] < length)
                & (key_ids[None, None, :] < length)
            )
            if pair_mask is not None:
                keep &= pair_mask[:, query_index[:, None], key_index[None, :]]
            if causal:
                keep &= key_ids[None, None, :] <= query_ids[None, :, None]

            # Rescale previous sums whenever a tile introduces a larger logit.
            masked_logits = jnp.where(keep[:, None], logits, -jnp.inf)
            next_maximum = jax.lax.stop_gradient(
                jnp.maximum(maximum, jnp.max(masked_logits, axis=-1))
            )
            safe_maximum = jnp.where(jnp.isfinite(next_maximum), next_maximum, 0)
            rescale = jnp.exp(
                jnp.where(jnp.isfinite(maximum), maximum - safe_maximum, -jnp.inf)
            )
            probabilities = jnp.exp(
                jnp.where(keep[:, None], logits - safe_maximum[..., None], -jnp.inf)
            )
            denominator = rescale * denominator + probabilities.sum(axis=-1)
            scalar_sum = rescale[..., None] * scalar_sum + jnp.einsum(
                "bhij,bjhd->bhid", probabilities, v[:, key_index], precision=precision
            )
            point_sum = rescale[..., None, None] * point_sum + jnp.einsum(
                "bhij,bjhpc->bhipc",
                probabilities,
                vp[:, key_index],
                precision=precision,
            )

            if factors is not None:
                aggregated = jnp.einsum(
                    "bhij,bjrc->bihrc",
                    probabilities,
                    factors.value_k[:, key_index],
                    precision=precision,
                )
                pair_update = jnp.einsum(
                    "birc,bihrc->bhic",
                    factors.value_q[:, query_index],
                    aggregated,
                    precision=precision,
                )
                pair_sum = rescale[..., None] * pair_sum + pair_update
            elif pair_values is not None:
                values_tile = pair_values[:, query_index[:, None], key_index[None, :]]
                pair_sum = rescale[..., None] * pair_sum + jnp.einsum(
                    "bhij,bijc->bhic", probabilities, values_tile, precision=precision
                )

            return (next_maximum, denominator, scalar_sum, point_sum, pair_sum), None

        # Rematerialize tile arithmetic instead of saving distance/logit tiles.
        final, _ = jax.lax.scan(jax.checkpoint(key_tile), initial, key_tiles)
        _, denominator, scalar_sum, point_sum, pair_sum = final
        divisor = jnp.where(denominator > 0, denominator, 1)
        return IPAOutput(
            jnp.swapaxes(scalar_sum / divisor[..., None], 1, 2),
            jnp.swapaxes(point_sum / divisor[..., None, None], 1, 2),
            (
                jnp.swapaxes(pair_sum / divisor[..., None], 1, 2)
                if pair_channels
                else None
            ),
            jnp.any(denominator > 0, axis=1),
        )

    chunks = jax.lax.map(jax.checkpoint(query_tile), query_tiles)

    def join_tiles(array):
        if array is None:
            return None
        array = jnp.swapaxes(array, 0, 1)
        return array.reshape(batch, -1, *array.shape[3:])[:, :length]

    return jax.tree.map(join_tiles, chunks)
