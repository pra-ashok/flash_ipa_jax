"""Dense and query-chunked direct-distance attention for debugging."""

import math
import jax
import jax.numpy as jnp
from .common import point_weights
from .types import IPAOutput


def reference_attention(
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
    b, n, h, _ = q.shape
    weights = point_weights(raw, qp.shape[-2])
    node_mask = jnp.ones((b, n), dtype=bool) if mask is None else mask
    key_ids = jnp.arange(n)

    def query_block(ids):
        safe_ids = jnp.minimum(ids, n - 1)
        qi, qpi = q[:, safe_ids], qp[:, safe_ids]
        scalar = jnp.einsum(
            "bihd,bjhd->bhij", qi, k, precision=jax.lax.Precision.HIGHEST
        ) / math.sqrt(3 * q.shape[-1])
        # Direct differences avoid cancellation in the native correctness path.
        distances = jnp.sum((qpi[:, :, None] - kp[:, None]) ** 2, axis=(-1, -2))
        logits = scalar - 0.5 * jnp.einsum("bijh,h->bhij", distances, weights)
        if factors is not None:
            logits += jnp.einsum(
                "bihr,bjhr->bhij",
                factors.bias_q[:, safe_ids],
                factors.bias_k,
                precision=jax.lax.Precision.HIGHEST,
            )
        if pair_bias is not None:
            logits += jnp.transpose(pair_bias[:, safe_ids], (0, 3, 1, 2))
        keep = (
            node_mask[:, safe_ids, None]
            & node_mask[:, None, :]
            & (ids[None, :, None] < n)
        )
        if pair_mask is not None:
            keep &= pair_mask[:, safe_ids]
        if causal:
            keep &= key_ids[None, None, :] <= ids[None, :, None]
        valid = jnp.any(keep, axis=-1)
        masked = jnp.where(keep[:, None], logits, -jnp.inf)
        maximum = jnp.max(masked, axis=-1, keepdims=True)
        maximum = jax.lax.stop_gradient(jnp.where(valid[:, None, :, None], maximum, 0))
        numer = jnp.exp(jnp.where(keep[:, None], logits - maximum, -jnp.inf))
        denominator = jnp.sum(numer, axis=-1, keepdims=True)
        a = numer / jnp.where(denominator > 0, denominator, 1)
        scalar_out = jnp.einsum(
            "bhij,bjhd->bihd", a, v, precision=jax.lax.Precision.HIGHEST
        )
        points_out = jnp.einsum(
            "bhij,bjhpc->bihpc", a, vp, precision=jax.lax.Precision.HIGHEST
        )
        pair_out = None
        if factors is not None:
            aggregated = jnp.einsum(
                "bhij,bjrc->bihrc",
                a,
                factors.value_k,
                precision=jax.lax.Precision.HIGHEST,
            )
            pair_out = jnp.einsum(
                "birc,bihrc->bihc",
                factors.value_q[:, safe_ids],
                aggregated,
                precision=jax.lax.Precision.HIGHEST,
            )
        elif pair_values is not None:
            pair_out = jnp.einsum(
                "bhij,bijc->bihc",
                a,
                pair_values[:, safe_ids],
                precision=jax.lax.Precision.HIGHEST,
            )
        return IPAOutput(scalar_out, points_out, pair_out, valid)

    if config.native_mode == "dense":
        return query_block(jnp.arange(n))
    size = min(config.chunk_size, n)
    ids = jnp.arange(((n + size - 1) // size) * size).reshape(-1, size)
    # Rematerialize each block in reverse mode instead of retaining its logits.
    chunks = jax.lax.map(jax.checkpoint(query_block), ids)

    def join(x):
        if x is None:
            return None
        x = jnp.swapaxes(x, 0, 1)
        return x.reshape((b, -1, *x.shape[3:]))[:, :n]

    return jax.tree.map(join, chunks)
