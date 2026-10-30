"""Public projected attention API and observable execution dispatch.

Use ``layer.InvariantPointAttention`` for learned projections and local coordinate systems.
This lower-level API accepts already projected scalars and global points.
"""

import jax
import jax.numpy as jnp

from .backends import BackendConfig, resolve_backend, enforce_decision, fused_attention
from .kernels.types import IPAOutput
from .kernels.packing import pack_qkv
from .kernels.validation import validate_inputs
from .kernels.reference import reference_attention
from .kernels.tiled import tiled_attention


def backend_report(
    q,
    v,
    q_points,
    v_points,
    *,
    factors=None,
    pair_bias=None,
    pair_values=None,
    mask=None,
    pair_mask=None,
    backend=BackendConfig(),
    k=None,
    k_points=None,
    head_weights=None,
):
    """Inspect eligibility outside JIT, without executing an attention kernel.

    All supplied arrays participate in dtype promotion. Include the optional
    ``k``, ``k_points``, and ``head_weights`` for a complete mixed-dtype report;
    otherwise those omitted tensors are assumed to have a compatible dtype.
    """
    q = jnp.asarray(q)
    arrays = (
        q,
        v,
        q_points,
        v_points,
        factors,
        pair_bias,
        pair_values,
        k,
        k_points,
        head_weights,
    )
    dtype = jnp.result_type(jnp.float32, *[x.dtype for x in jax.tree.leaves(arrays)])
    qdim = q.shape[-1] + 5 * q_points.shape[-2]
    vdim = v.shape[-1] + 3 * v_points.shape[-2]
    if factors is not None:
        qdim += factors.bias_q.shape[-1]
        vdim += factors.value_k.shape[-2] * factors.value_k.shape[-1]
    return resolve_backend(
        backend,
        q,
        qdim,
        vdim,
        has_mask=mask is not None or pair_mask is not None,
        has_dense_bias=pair_bias is not None,
        has_dense_values=pair_values is not None,
        input_dtype=dtype,
    )


def ipa_attention(
    q,
    k,
    v,
    q_points,
    k_points,
    v_points,
    head_weights,
    *,
    factors=None,
    pair_bias=None,
    pair_values=None,
    mask=None,
    pair_mask=None,
    causal=False,
    backend=BackendConfig(),
):
    """Self attention on projected scalars and GLOBAL points; no dropout.

    Shapes: Q/K [B,N,H,D], V [B,N,H,Dv], points [B,N,H,P,3],
    raw head weights [H], dense bias [B,N,N,H] (already scaled),
    dense pair values [B,N,N,C]. All numeric inputs must be finite.
    Use boolean masks to remove edges. Fully masked rows have zero output and
    zero derivatives. Backend/config/causal are static closure arguments to jit.
    """
    validate_inputs(
        q,
        k,
        v,
        q_points,
        k_points,
        v_points,
        head_weights,
        factors,
        pair_bias,
        pair_values,
        mask,
        pair_mask,
    )
    arrays = (
        q,
        k,
        v,
        q_points,
        k_points,
        v_points,
        head_weights,
        factors,
        pair_bias,
        pair_values,
    )
    dtype = jnp.result_type(jnp.float32, *[x.dtype for x in jax.tree.leaves(arrays)])
    q, k, v, qp, kp, vp, raw, factors, pair_bias, pair_values = jax.tree.map(
        lambda x: jnp.asarray(x, dtype), arrays
    )
    mask = None if mask is None else jnp.asarray(mask)
    pair_mask = None if pair_mask is None else jnp.asarray(pair_mask)
    decision = backend_report(
        q,
        v,
        qp,
        vp,
        factors=factors,
        pair_bias=pair_bias,
        pair_values=pair_values,
        mask=mask,
        pair_mask=pair_mask,
        backend=backend,
    )
    enforce_decision(backend, decision)
    if decision.selected == "native":
        implementation = (
            tiled_attention if backend.native_mode == "flash" else reference_attention
        )
        return implementation(
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
            backend,
        )
    pq, pk, pv, origin = pack_qkv(q, k, v, qp, kp, vp, raw, factors)
    out = fused_attention(pq, pk, pv, config=backend, decision=decision, causal=causal)
    dv, points = v.shape[-1], vp.shape[-2]
    scalar_out = out[..., :dv]
    point_out = out[..., dv : dv + 3 * points].reshape(vp.shape) + origin
    pair_out = None
    if factors is not None:
        rank, channels = factors.value_k.shape[-2:]
        pair_part = out[..., dv + 3 * points : dv + 3 * points + rank * channels]
        pair_part = pair_part.reshape(*q.shape[:3], rank, channels)
        pair_out = jnp.einsum("birc,bihrc->bihc", factors.value_q, pair_part)
    return IPAOutput(scalar_out, point_out, pair_out, jnp.ones(q.shape[:2], dtype=bool))
