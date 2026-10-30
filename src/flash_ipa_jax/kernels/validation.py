"""Static shape and dtype checks, safe to run while JAX traces a function."""

import jax.numpy as jnp


def _check_shape(name, x, shape):
    if x.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {x.shape}")
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating dtype")


def validate_inputs(
    q, k, v, qp, kp, vp, raw_weights, factors, pair_bias, pair_values, mask, pair_mask
):
    if q.ndim != 4:
        raise ValueError("q must be [B,N,H,D]")
    b, n, h, d = q.shape
    if min(q.shape) < 1:
        raise ValueError("B,N,H,D must be positive")
    _check_shape("q", q, q.shape)
    _check_shape("k", k, q.shape)
    if v.ndim != 4 or v.shape[-1] < 1:
        raise ValueError("v must be [B,N,H,Dv] with Dv > 0")
    _check_shape("v", v, (b, n, h, v.shape[-1]))
    if qp.ndim != 5 or qp.shape[-2] < 1:
        raise ValueError("q_points must be [B,N,H,P,3] with P > 0")
    _check_shape("q_points", qp, (b, n, h, qp.shape[-2], 3))
    _check_shape("k_points", kp, qp.shape)
    if vp.ndim != 5 or vp.shape[-2] < 1:
        raise ValueError("v_points must be [B,N,H,Pv,3] with Pv > 0")
    _check_shape("v_points", vp, (b, n, h, vp.shape[-2], 3))
    _check_shape("head_weights", raw_weights, (h,))
    if pair_bias is not None:
        _check_shape("pair_bias", pair_bias, (b, n, n, h))
    if pair_values is not None:
        if pair_values.ndim != 4 or pair_values.shape[-1] < 1:
            raise ValueError("pair_values must be [B,N,N,C] with C > 0")
        _check_shape("pair_values", pair_values, (b, n, n, pair_values.shape[-1]))
    if factors is not None:
        if pair_values is not None:
            raise ValueError("choose factorized or dense pair values, not both")
        if factors.bias_q.ndim != 4 or factors.value_q.ndim != 4:
            raise ValueError("factors must have rank 4")
        rank = factors.bias_q.shape[-1]
        if rank < 1 or factors.value_q.shape[-1] < 1:
            raise ValueError("factor rank and value channels must be positive")
        for name, x, shape in (
            ("bias_q", factors.bias_q, (b, n, h, rank)),
            ("bias_k", factors.bias_k, (b, n, h, rank)),
            ("value_q", factors.value_q, (b, n, rank, factors.value_q.shape[-1])),
            ("value_k", factors.value_k, factors.value_q.shape),
        ):
            _check_shape(name, x, shape)
    for name, x, shape in (("mask", mask, (b, n)), ("pair_mask", pair_mask, (b, n, n))):
        if x is not None and (x.shape != shape or x.dtype != jnp.bool_):
            raise ValueError(f"{name} must be boolean with shape {shape}")
