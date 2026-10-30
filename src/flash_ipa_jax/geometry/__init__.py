"""Pure JAX rigid-body geometry, using ``global = R @ local + t``."""

from ._quaternion import (
    identity_quats,
    identity_rot_mats,
    identity_trans,
    invert_quat,
    invert_rot_mat,
    quat_multiply,
    quat_multiply_by_vec,
    quat_to_rot,
    rot_matmul,
    rot_to_quat,
    rot_vec_mul,
)
from .rigid import Rigid, create_rigid
from .rotation import Rotation

__all__ = [
    "Rigid",
    "Rotation",
    "create_rigid",
    "identity_quats",
    "identity_rot_mats",
    "identity_trans",
    "invert_quat",
    "invert_rot_mat",
    "quat_multiply",
    "quat_multiply_by_vec",
    "quat_to_rot",
    "rot_matmul",
    "rot_to_quat",
    "rot_vec_mul",
]
