"""Rigid coordinate systems: ``global_point = rotation @ local_point + translation``."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._quaternion import as_geometry_array, identity_trans, put_array
from .rotation import Rotation


@jax.tree_util.register_pytree_node_class
class Rigid:
    """A batch of rotations and translations, stored as a JAX pytree.

    A residue coordinate system expresses local coordinates relative to that residue.
    ``apply`` converts them to global coordinates, and ``invert_apply`` puts
    global points back in the coordinate system. ``shape`` contains only batch axes.
    """

    def __init__(self, rots=None, trans=None):
        if rots is None and trans is None:
            raise ValueError("provide at least one of rots and trans")
        if trans is not None:
            trans = as_geometry_array(trans)
            if trans.shape[-1:] != (3,):
                raise ValueError("trans must have shape [..., 3]")
        if rots is None:
            rots = Rotation.identity(trans.shape[:-1], dtype=trans.dtype)
        if not isinstance(rots, Rotation):
            raise TypeError("rots must be a Rotation; use create_rigid for matrices")
        if trans is None:
            trans = identity_trans(rots.shape, dtype=rots.dtype)
        if rots.shape != trans.shape[:-1]:
            raise ValueError("rotation and translation batch dimensions must match")
        self._rots, self._trans = rots, trans

    def tree_flatten(self):
        """Expose both rotation and translation to JAX transformations."""
        return (self._rots, self._trans), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del auxiliary
        obj = object.__new__(cls)
        obj._rots, obj._trans = children
        return obj

    @staticmethod
    def identity(shape, dtype=None, device=None, requires_grad=True, fmt="quat"):
        """Construct identity coordinate systems; JAX uses ``grad`` rather than grad flags."""
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )

    @property
    def shape(self):
        """Batch dimensions shared by rotation and translation."""
        return self._trans.shape[:-1]

    @property
    def dtype(self):
        """Translation dtype; geometry is computed in at least float32."""
        return self._trans.dtype

    @property
    def device(self):
        """Device of a concrete translation array, or ``None`` while tracing."""
        return getattr(self._trans, "device", None)

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        return Rigid(self._rots[index], self._trans[index + (slice(None),)])

    def __mul__(self, right):
        """Multiply all components by a batch mask; zeros mean missing coordinate systems."""
        right = jnp.asarray(right)
        return Rigid(self._rots * right, self._trans * right[..., None])

    __rmul__ = __mul__

    def reshape(self, new_rots_shape=None, new_trans_shape=None):
        """Reshape stored arrays, including their final representation axes."""
        rots = self._rots.reshape(new_rots_shape)
        trans = (
            self._trans
            if new_trans_shape is None
            else self._trans.reshape(new_trans_shape)
        )
        return Rigid(rots, trans)

    def get_rots(self):
        """Return the Rotation object."""
        return self._rots

    def get_trans(self):
        """Return translations of shape ``(*shape,3)``."""
        return self._trans

    def compose_q_update_vec(self, q_update_vec, update_mask=None):
        """Update coordinate systems with local quaternion/translation values ``[...,6]``.

        The first three entries define quaternion ``(1,x,y,z)``. The remaining
        three describe a translation in the old local coordinate system. An
        optional ``[...,1]`` mask leaves unselected coordinate systems unchanged.
        """
        if q_update_vec.shape[-1:] != (6,):
            raise ValueError("q_update_vec must have shape [..., 6]")
        rots = self._rots.compose_q_update_vec(
            q_update_vec[..., :3], update_mask=update_mask
        )
        trans_update = self._rots.apply(q_update_vec[..., 3:])
        if update_mask is not None:
            trans_update = trans_update * update_mask
        return Rigid(rots, self._trans + trans_update)

    def compose(self, r):
        """Compose coordinate systems: ``self.compose(r).apply(p) == self.apply(r.apply(p))``."""
        return Rigid(
            self._rots.compose_r(r._rots), self._rots.apply(r._trans) + self._trans
        )

    def compose_r(self, rot, order="right"):
        """Compose only the rotation, leaving the translation unchanged."""
        if order == "right":
            rots = self._rots.compose_r(rot)
        elif order == "left":
            rots = rot.compose_r(self._rots)
        else:
            raise ValueError("order must be 'right' or 'left'")
        return Rigid(rots, self._trans)

    def apply(self, pts):
        """Convert local ``[...,3]`` points to global coordinates.

        For points ``[B,N,H,P,3]`` with coordinate systems ``[B,N]``, first use
        ``coordinate_systems[...,None,None].apply(points)`` to broadcast
        over heads and points.
        """
        return self._rots.apply(pts) + self._trans

    def invert_apply(self, pts):
        """Convert global points to this coordinate system's local coordinates."""
        return self._rots.invert_apply(pts - self._trans)

    def invert(self):
        """Return inverse coordinate systems, including the rotated negative translation."""
        rots = self._rots.invert()
        return Rigid(rots, -rots.apply(self._trans))

    def map_tensor_fn(self, fn):
        """Apply ``fn`` independently to component arrays over batch axes."""
        trans = jnp.stack([fn(self._trans[..., i]) for i in range(3)], axis=-1)
        return Rigid(self._rots.map_tensor_fn(fn), trans)

    def to_tensor_4x4(self):
        """Return homogeneous matrices of shape ``(*shape,4,4)``."""
        value = jnp.zeros((*self.shape, 4, 4), dtype=self.dtype)
        value = value.at[..., :3, :3].set(self._rots.get_rot_mats())
        return value.at[..., :3, 3].set(self._trans).at[..., 3, 3].set(1)

    @staticmethod
    def from_tensor_4x4(t):
        """Construct coordinate systems from homogeneous matrices ``[...,4,4]``."""
        t = jnp.asarray(t)
        if t.shape[-2:] != (4, 4):
            raise ValueError("homogeneous transforms must have shape [..., 4, 4]")
        return Rigid(Rotation(rot_mats=t[..., :3, :3]), t[..., :3, 3])

    def to_tensor_7(self):
        """Return ``[...,7]`` arrays with quaternion first, then translation."""
        return jnp.concatenate((self._rots.get_quats(), self._trans), axis=-1)

    @staticmethod
    def from_tensor_7(t, normalize_quats=False):
        """Construct coordinate systems from ``(w,x,y,z,tx,ty,tz)`` arrays."""
        t = jnp.asarray(t)
        if t.shape[-1:] != (7,):
            raise ValueError("quaternion transforms must have shape [..., 7]")
        return Rigid(
            Rotation(quats=t[..., :4], normalize_quats=normalize_quats), t[..., 4:]
        )

    @staticmethod
    def from_3_points(p_neg_x_axis, origin, p_xy_plane, eps=1e-8):
        """Construct coordinate systems with Gram-Schmidt (AlphaFold Algorithm 21).

        ``origin - p_neg_x_axis`` defines positive x. The third point defines
        positive y after removing its x component; z is their cross product.
        Coincident or collinear inputs do not define a valid orthonormal coordinate system.
        """
        e0 = origin - p_neg_x_axis
        e0 = e0 / jnp.sqrt(jnp.sum(e0 * e0, -1, keepdims=True) + eps)
        e1 = p_xy_plane - origin
        e1 = e1 - e0 * jnp.sum(e0 * e1, -1, keepdims=True)
        e1 = e1 / jnp.sqrt(jnp.sum(e1 * e1, -1, keepdims=True) + eps)
        matrix = jnp.stack((e0, e1, jnp.cross(e0, e1)), axis=-1)
        return Rigid(Rotation(rot_mats=matrix), origin)

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        """Build backbone coordinate systems: CA is origin, C is +x, N is in positive xy.

        This implements the intended reference coordinate system construction without the
        upstream implementation's assignment typo in its second rotation.
        """
        return Rigid.from_3_points(2 * ca_xyz - c_xyz, ca_xyz, n_xyz, eps)

    def unsqueeze(self, dim):
        """Insert a batch axis; ``-1`` appends after existing batch axes."""
        axis = dim if dim >= 0 else len(self.shape) + dim + 1
        if not 0 <= axis <= len(self.shape):
            raise ValueError("dim is outside the rigid batch dimensions")
        return Rigid(self._rots.unsqueeze(dim), jnp.expand_dims(self._trans, axis))

    @staticmethod
    def cat(ts, dim):
        """Concatenate coordinate systems along an existing batch dimension."""
        if not ts:
            raise ValueError("ts must contain at least one Rigid")
        axis = dim if dim >= 0 else len(ts[0].shape) + dim
        rots = Rotation.cat([t._rots for t in ts], dim)
        return Rigid(rots, jnp.concatenate([t._trans for t in ts], axis=axis))

    def apply_rot_fn(self, fn):
        """Apply a ``Rotation -> Rotation`` function to the stored rotation."""
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn):
        """Apply an ``array -> array`` function to the stored translation."""
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor):
        """Scale translations, for example from Angstroms to nanometres."""
        return self.apply_trans_fn(lambda trans: trans * trans_scale_factor)

    def stop_rot_gradient(self):
        """Keep translation gradients while stopping rotation gradients."""
        return self.apply_rot_fn(lambda rots: rots.detach())

    def to(self, device=None, dtype=None):
        """Move all leaves to a JAX device/sharding, optionally changing dtype."""
        return Rigid(
            self._rots.to(device, dtype), put_array(self._trans, device, dtype)
        )

    def cuda(self):
        """Compatibility shortcut for ``to(device='gpu')``."""
        return self.to(device="gpu")


def create_rigid(rots, trans):
    """Create coordinate systems from matrix rotations ``[...,3,3]`` and ``[...,3]`` shifts."""
    return Rigid(Rotation(rot_mats=rots), trans)
