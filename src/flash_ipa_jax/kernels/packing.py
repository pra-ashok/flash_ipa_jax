"""Convert IPA point distances and factorized pairs to ordinary Q/K/V."""

import math
import jax.numpy as jnp
from .common import point_weights


def pack_qkv(q, k, v, q_points, k_points, v_points, raw_weights, factors=None):
    """Algebraic IPA lift. Call attention with softmax_scale=1.0, never 1/sqrt(D).

    Both quadratic norm terms are retained. Subtract a shared, differentiable
    origin before expansion to reduce cancellation from global translations.
    This is exact in real arithmetic; half-precision packing adds rounding.
    """
    origin = k_points[:, :1, :1, :1, :]
    qp, kp, vp = q_points - origin, k_points - origin, v_points - origin
    w = point_weights(raw_weights, qp.shape[-2])[None, None, :, None]
    qn = jnp.sum(qp**2, axis=-1)
    kn = jnp.sum(kp**2, axis=-1)
    q_parts = [q, qp.reshape(*q.shape[:3], -1), qn, jnp.ones_like(qn)]
    k_parts = [
        k / math.sqrt(3 * q.shape[-1]),
        kp.reshape(*k.shape[:3], -1) * w,
        jnp.ones_like(kn) * (-0.5 * w),
        kn * (-0.5 * w),
    ]
    v_parts = [v, vp.reshape(*v.shape[:3], -1)]
    if factors is not None:
        q_parts.append(factors.bias_q)
        k_parts.append(factors.bias_k)
        values = factors.value_k.reshape(*v.shape[:2], 1, -1)
        v_parts.append(jnp.broadcast_to(values, (*v.shape[:3], values.shape[-1])))
    return (
        jnp.concatenate(q_parts, -1),
        jnp.concatenate(k_parts, -1),
        jnp.concatenate(v_parts, -1),
        origin,
    )
