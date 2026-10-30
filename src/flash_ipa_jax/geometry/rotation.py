"""A batch of 3D rotations that can pass through ``jit``, ``grad`` and ``vmap``."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._quaternion import (
    as_geometry_array,
    identity_quats,
    identity_rot_mats,
    invert_quat,
    invert_rot_mat,
    normalize_quat,
    put_array,
    quat_multiply,
    quat_multiply_by_vec,
    quat_to_rot,
    rot_matmul,
    rot_to_quat,
    rot_vec_mul,
)


@jax.tree_util.register_pytree_node_class
class Rotation:
    """Store either matrices ``[...,3,3]`` or quaternions ``[...,4]``.

    ``shape`` contains only batch axes, without the matrix/quaternion axes.
    Methods return new objects. Quaternions follow the ``(w,x,y,z)`` order.
    Matrices should be proper orthonormal rotations; they are not projected
    onto the rotation group automatically.
    """

    def __init__(
        self, rot_mats=None, quats=None, quats_mask=None, normalize_quats=True
    ):
        if (rot_mats is None) == (quats is None):
            raise ValueError("provide exactly one of rot_mats and quats")
        self._rot_mats = None if rot_mats is None else as_geometry_array(rot_mats)
        self._quats = None if quats is None else as_geometry_array(quats)
        if self._rot_mats is not None and self._rot_mats.shape[-2:] != (3, 3):
            raise ValueError("rot_mats must have shape [..., 3, 3]")
        if self._quats is not None:
            if self._quats.shape[-1:] != (4,):
                raise ValueError("quats must have shape [..., 4]")
            if normalize_quats:
                self._quats = normalize_quat(self._quats, quats_mask)

    def tree_flatten(self):
        """Expose the stored array as a differentiable JAX leaf."""
        return (self.get_cur_rot(),), self._quats is not None

    @classmethod
    def tree_unflatten(cls, quaternion_format, children):
        # Bypass validation: JAX may temporarily supply abstract placeholders.
        obj = object.__new__(cls)
        obj._quats = children[0] if quaternion_format else None
        obj._rot_mats = None if quaternion_format else children[0]
        return obj

    @staticmethod
    def identity(shape, dtype=None, device=None, requires_grad=True, fmt="quat"):
        """Construct identity rotations; ``fmt`` is ``quat`` or ``rot_mat``.

        ``requires_grad`` is accepted for migration; JAX differentiates through
        floating arrays using ``jax.grad`` without a stored gradient flag.
        """
        if fmt == "quat":
            return Rotation(
                quats=identity_quats(shape, dtype, device, requires_grad),
                normalize_quats=False,
            )
        if fmt == "rot_mat":
            return Rotation(
                rot_mats=identity_rot_mats(shape, dtype, device, requires_grad)
            )
        raise ValueError("fmt must be 'quat' or 'rot_mat'")

    @property
    def shape(self):
        """Batch dimensions shared by the rotations."""
        return self.get_cur_rot().shape[: -(1 if self._quats is not None else 2)]

    @property
    def dtype(self):
        """Floating dtype of the stored representation."""
        return self.get_cur_rot().dtype

    @property
    def device(self):
        """JAX device of a concrete array, or ``None`` for traced arrays."""
        return getattr(self.get_cur_rot(), "device", None)

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        axes = (slice(None),) * (1 if self._quats is not None else 2)
        return self._with_array(self.get_cur_rot()[index + axes])

    def __mul__(self, right):
        """Multiply components by a batch mask; zeros are missing rotations."""
        right = jnp.asarray(right)
        axes = (None,) * (1 if self._quats is not None else 2)
        return self._with_array(self.get_cur_rot() * right[(...,) + axes])

    __rmul__ = __mul__

    def _with_array(self, value):
        if self._quats is not None:
            return Rotation(quats=value, normalize_quats=False)
        return Rotation(rot_mats=value)

    def reshape(self, new_rots_shape=None):
        """Reshape the stored array, including its final representation axes.

        For example, reshape two quaternion batch axes to one with
        ``rotation.reshape((-1, 4))``. This matches upstream's API.
        """
        value = self.get_cur_rot()
        return self._with_array(
            value if new_rots_shape is None else value.reshape(new_rots_shape)
        )

    def get_cur_rot(self):
        """Return the underlying array without converting its representation."""
        return self._quats if self._quats is not None else self._rot_mats

    def get_rot_mats(self):
        """Return matrices of shape ``(*shape,3,3)``."""
        return (
            self._rot_mats if self._rot_mats is not None else quat_to_rot(self._quats)
        )

    def get_quats(self):
        """Return quaternions of shape ``(*shape,4)``."""
        return self._quats if self._quats is not None else rot_to_quat(self._rot_mats)

    def get_rotvec(self, eps=1e-6):
        """Return axis times angle (in radians), with angle in ``[0, pi]``."""
        quat = self.get_quats()
        quat = jnp.where(quat[..., :1] < 0, -quat, quat)
        norm = jnp.sqrt(jnp.maximum(jnp.sum(quat[..., 1:] ** 2, -1), eps**2))
        angle = 2 * jnp.arctan2(norm, quat[..., 0])
        angle2 = angle * angle
        series = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
        scale = jnp.where(
            angle <= 1e-3, series, angle / jnp.maximum(jnp.sin(angle / 2), eps)
        )
        return scale[..., None] * quat[..., 1:]

    def compose_q_update_vec(
        self, q_update_vec, normalize_quats=True, update_mask=None
    ):
        """Right-compose with quaternion ``(1,*q_update_vec)`` and normalize.

        ``q_update_vec`` is ``[...,3]``; an optional ``[...,1]`` mask leaves
        unselected rotations unchanged. This is AlphaFold's local coordinate system update.
        """
        quat = self.get_quats()
        update = quat_multiply_by_vec(quat, q_update_vec)
        if update_mask is not None:
            update = update * update_mask
        mask = None if update_mask is None else jnp.asarray(update_mask)[..., 0]
        return Rotation(
            quats=quat + update, quats_mask=mask, normalize_quats=normalize_quats
        )

    def compose_r(self, r):
        """Return matrix composition ``self @ r`` (apply ``r`` first)."""
        return Rotation(rot_mats=rot_matmul(self.get_rot_mats(), r.get_rot_mats()))

    def compose_q(self, r, normalize_quats=True):
        """Return quaternion composition ``self * r`` (apply ``r`` first)."""
        return Rotation(
            quats=quat_multiply(self.get_quats(), r.get_quats()),
            normalize_quats=normalize_quats,
        )

    def apply(self, pts):
        """Rotate ``[...,3]`` points; use ``unsqueeze`` for extra point axes."""
        return rot_vec_mul(self.get_rot_mats(), pts)

    def invert_apply(self, pts):
        """Apply the inverse rotation to ``[...,3]`` points."""
        return rot_vec_mul(invert_rot_mat(self.get_rot_mats()), pts)

    def invert(self, mask=None):
        """Return inverse rotations; the optional mask applies to quaternions."""
        if self._quats is not None:
            return Rotation(quats=invert_quat(self._quats, mask), normalize_quats=False)
        return Rotation(rot_mats=invert_rot_mat(self._rot_mats))

    def unsqueeze(self, dim):
        """Insert a batch axis, including at the end with ``dim=-1``."""
        axis = dim if dim >= 0 else len(self.shape) + dim + 1
        if not 0 <= axis <= len(self.shape):
            raise ValueError("dim is outside the rotation batch dimensions")
        return self._with_array(jnp.expand_dims(self.get_cur_rot(), axis))

    @staticmethod
    def cat(rs, dim):
        """Concatenate batch axes, returning matrix representation."""
        if not rs:
            raise ValueError("rs must contain at least one Rotation")
        axis = dim if dim >= 0 else len(rs[0].shape) + dim
        if not 0 <= axis < len(rs[0].shape):
            raise ValueError("dim is outside the rotation batch dimensions")
        return Rotation(
            rot_mats=jnp.concatenate([r.get_rot_mats() for r in rs], axis=axis)
        )

    def map_tensor_fn(self, fn):
        """Apply ``fn`` to each component over batch axes, e.g. summing an axis."""
        value = self.get_cur_rot()
        if self._quats is None:
            value = value.reshape(self.shape + (9,))
        mapped = jnp.stack([fn(value[..., i]) for i in range(value.shape[-1])], -1)
        if self._quats is None:
            mapped = mapped.reshape(mapped.shape[:-1] + (3, 3))
        return self._with_array(mapped)

    def to(self, device=None, dtype=None):
        """Copy to a JAX device/sharding and/or cast the stored array."""
        return self._with_array(put_array(self.get_cur_rot(), device, dtype))

    def cuda(self):
        """Compatibility shortcut for ``to(device='gpu')`` on GPU installations."""
        return self.to(device="gpu")

    def detach(self):
        """Stop gradients through the stored representation."""
        return self._with_array(jax.lax.stop_gradient(self.get_cur_rot()))
