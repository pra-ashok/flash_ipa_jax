"""Array operations for rotations; quaternions use the order ``(w, x, y, z)``.

These functions are pure JAX. A rotation matrix acts on column vectors, so
``quat_to_rot(quat_multiply(a, b))`` represents applying ``b`` and then ``a``.
Adapted from the OpenFold/se3_diffusion geometry used by upstream flash_ipa.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def as_geometry_array(value):
    """Keep float64 when enabled and compute low precision geometry in float32."""
    value = jnp.asarray(value)
    return value.astype(jnp.result_type(value.dtype, jnp.float32))


def put_array(value, device=None, dtype=None):
    """Cast an array and optionally place it on a JAX device or sharding."""
    value = jnp.asarray(value, dtype=dtype)
    if isinstance(device, str):
        platform, _, index = device.partition(":")
        device = jax.devices(platform)[int(index or 0)]
    return jax.device_put(value, device) if device is not None else value


def identity_rot_mats(batch_dims, dtype=None, device=None, requires_grad=True):
    """Return identity matrices of shape ``(*batch_dims, 3, 3)``.

    ``requires_grad`` is accepted for source compatibility; JAX uses ``grad``
    transformations instead of storing a gradient flag on each array.
    """
    del requires_grad
    return put_array(
        jnp.broadcast_to(jnp.eye(3, dtype=dtype), (*batch_dims, 3, 3)), device
    )


def identity_trans(batch_dims, dtype=None, device=None, requires_grad=True):
    """Return zero translations of shape ``(*batch_dims, 3)``."""
    del requires_grad
    return put_array(jnp.zeros((*batch_dims, 3), dtype=dtype), device)


def identity_quats(batch_dims, dtype=None, device=None, requires_grad=True):
    """Return identity quaternions ``(1, 0, 0, 0)`` for each batch item."""
    del requires_grad
    value = jnp.zeros((*batch_dims, 4), dtype=dtype).at[..., 0].set(1)
    return put_array(value, device)


def normalize_quat(quat, mask=None):
    """Normalize selected quaternions; a selected zero becomes the identity."""
    norm2 = jnp.sum(quat * quat, axis=-1, keepdims=True)
    unit = quat / jnp.sqrt(jnp.maximum(norm2, jnp.finfo(quat.dtype).tiny))
    identity = jnp.zeros_like(quat).at[..., 0].set(1)
    unit = jnp.where(norm2 > 0, unit, identity)
    return (
        unit
        if mask is None
        else jnp.where(jnp.asarray(mask, bool)[..., None], unit, quat)
    )


def rot_matmul(a, b):
    """Multiply broadcastable rotation matrices without reducing precision."""
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def rot_vec_mul(r, t):
    """Rotate points: ``r [...,3,3] @ t [...,3]`` with JAX broadcasting."""
    return jnp.sum(r * t[..., None, :], axis=-1)


def quat_to_rot(quat):
    """Convert unit quaternions ``[...,4]`` to rotation matrices ``[...,3,3]``.

    This function does not normalize its input. Use ``Rotation(quats=...)``
    when normalization is wanted.
    """
    quat = jnp.asarray(quat)
    if quat.shape[-1:] != (4,):
        raise ValueError("quaternions must have shape [..., 4]")
    w, x, y, z = (quat[..., i] for i in range(4))
    rows = (
        jnp.stack(
            (w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)),
            -1,
        ),
        jnp.stack(
            (2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)),
            -1,
        ),
        jnp.stack(
            (2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z),
            -1,
        ),
    )
    return jnp.stack(rows, axis=-2)


def rot_to_quat(rot):
    """Convert proper rotation matrices to quaternions without an eigensolve.

    Selecting the best conditioned component avoids the repeated-eigenvalue
    gradient problem of an eigendecomposition at the identity rotation.
    Quaternion signs are arbitrary: ``q`` and ``-q`` describe the same rotation.
    """
    rot = as_geometry_array(rot)
    if rot.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have shape [..., 3, 3]")
    xx, xy, xz = (rot[..., 0, i] for i in range(3))
    yx, yy, yz = (rot[..., 1, i] for i in range(3))
    zx, zy, zz = (rot[..., 2, i] for i in range(3))
    squares = jnp.stack(
        (1 + xx + yy + zz, 1 + xx - yy - zz, 1 - xx + yy - zz, 1 - xx - yy + zz), -1
    )
    roots = jnp.sqrt(jnp.maximum(squares, jnp.asarray(1e-12, rot.dtype)))
    candidates = jnp.stack(
        (
            jnp.stack((squares[..., 0], zy - yz, xz - zx, yx - xy), -1),
            jnp.stack((zy - yz, squares[..., 1], xy + yx, xz + zx), -1),
            jnp.stack((xz - zx, xy + yx, squares[..., 2], yz + zy), -1),
            jnp.stack((yx - xy, xz + zx, yz + zy, squares[..., 3]), -1),
        ),
        axis=-2,
    ) / (2 * jnp.maximum(roots, 0.1)[..., :, None])
    index = jnp.argmax(squares, axis=-1)
    return jnp.take_along_axis(candidates, index[..., None, None], axis=-2)[..., 0, :]


def quat_multiply(quat1, quat2):
    """Hamilton product of broadcastable ``(w,x,y,z)`` quaternions."""
    w1, v1 = quat1[..., :1], quat1[..., 1:]
    w2, v2 = quat2[..., :1], quat2[..., 1:]
    scalar = w1 * w2 - jnp.sum(v1 * v2, axis=-1, keepdims=True)
    vector = w1 * v2 + w2 * v1 + jnp.cross(v1, v2)
    return jnp.concatenate((scalar, vector), axis=-1)


def quat_multiply_by_vec(quat, vec):
    """Multiply ``quat`` by the pure-vector quaternion ``(0, *vec)``."""
    return quat_multiply(quat, jnp.concatenate((jnp.zeros_like(vec[..., :1]), vec), -1))


def invert_rot_mat(rot_mat):
    """Invert an orthonormal rotation matrix by transposing its last axes."""
    return jnp.swapaxes(rot_mat, -1, -2)


def invert_quat(quat, mask=None):
    """Return quaternion inverses; masked missing zeros remain finite zeros.

    If ``mask`` is given, unselected values receive only conjugation, matching
    upstream's missing-coordinate convention. Zero quaternions are not rotations.
    """
    quat = as_geometry_array(quat)
    conjugate = quat * jnp.asarray((1, -1, -1, -1), dtype=quat.dtype)
    norm2 = jnp.sum(quat * quat, axis=-1, keepdims=True)
    inverse = conjugate / jnp.maximum(norm2, jnp.finfo(quat.dtype).tiny)
    return (
        inverse
        if mask is None
        else jnp.where(jnp.asarray(mask, bool)[..., None], inverse, conjugate)
    )
