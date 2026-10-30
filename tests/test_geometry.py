"""Check coordinate system conventions, upstream update semantics, and JAX transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_ipa_jax.geometry import (
    Rigid,
    Rotation,
    create_rigid,
    identity_quats,
    invert_quat,
    quat_multiply,
    quat_to_rot,
    rot_to_quat,
)


def test_quaternion_convention_and_composition():
    # A quarter turn around z takes x to y; Hamilton products apply right first.
    quarter_z = jnp.array([1, 0, 0, 1], dtype=float) / jnp.sqrt(2)
    quarter_x = jnp.array([1, 1, 0, 0], dtype=float) / jnp.sqrt(2)
    rz, rx = Rotation(quats=quarter_z), Rotation(quats=quarter_x)
    np.testing.assert_allclose(rz.apply(jnp.array([1.0, 0, 0])), [0, 1, 0], atol=1e-7)
    expected = rz.get_rot_mats() @ rx.get_rot_mats()
    np.testing.assert_allclose(
        quat_to_rot(quat_multiply(quarter_z, quarter_x)), expected, atol=1e-7
    )
    np.testing.assert_allclose(rz.compose_q(rx).get_rot_mats(), expected, atol=1e-7)
    np.testing.assert_allclose(rz.compose_r(rx).get_rot_mats(), expected, atol=1e-7)


def test_rotation_matrix_quaternion_roundtrip_and_gradients():
    quat = jax.random.normal(jax.random.key(10), (32, 4), dtype=jnp.float64)
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
    matrices = quat_to_rot(quat)
    recovered = jax.jit(rot_to_quat)(matrices)
    np.testing.assert_allclose(quat_to_rot(recovered), matrices, atol=1e-10)
    # Repeated eigenvalues at identity must not create NaN gradients.
    gradient = jax.grad(lambda r: jnp.sum(rot_to_quat(r)))(jnp.eye(3))
    assert jnp.all(jnp.isfinite(gradient))
    axes = Rotation(quats=jnp.eye(4)).get_rot_mats()
    np.testing.assert_allclose(quat_to_rot(rot_to_quat(axes)), axes, atol=1e-10)


def test_rigid_composition_inversion_and_serialization():
    frames = Rigid(
        Rotation(quats=jnp.array([[1.0, 0, 0, 1], [1.0, 1, 0, 0]])),
        jnp.array([[3.0, 4, 5], [-2.0, 6, 1]]),
    )
    points = jnp.array([[1.0, 2, 3], [-3.0, 1, 4]])
    np.testing.assert_allclose(
        frames.invert_apply(frames.apply(points)), points, atol=1e-7
    )
    np.testing.assert_allclose(
        frames.invert().apply(frames.apply(points)), points, atol=1e-7
    )
    other = Rigid(None, jnp.array([[4.0, 0, 2], [2.0, 1, 3]]))
    np.testing.assert_allclose(
        frames.compose(other).apply(points),
        frames.apply(other.apply(points)),
        atol=1e-7,
    )
    homogeneous_points = jnp.concatenate((points, jnp.ones((2, 1))), -1)
    mapped = jnp.einsum("bij,bj->bi", frames.to_tensor_4x4(), homogeneous_points)
    np.testing.assert_allclose(mapped[..., :3], frames.apply(points), atol=1e-7)
    for restored in (
        Rigid.from_tensor_4x4(frames.to_tensor_4x4()),
        Rigid.from_tensor_7(frames.to_tensor_7()),
    ):
        np.testing.assert_allclose(
            restored.apply(points), frames.apply(points), atol=1e-7
        )


def test_frame_updates_use_old_local_translation_and_respect_mask():
    rotation = Rotation(quats=jnp.array([[1.0, 0, 0, 1], [1.0, 0, 0, 0]]))
    frame = Rigid(rotation, jnp.array([[5.0, 0, 0], [1.0, 2, 3]]))
    update = jnp.array([[0.0, 0, 1, 2, 0, 0], [1.0, 2, 3, 4, 5, 6]])
    mask = jnp.array([[1.0], [0.0]])
    result = jax.jit(lambda f, u, m: f.compose_q_update_vec(u, m))(frame, update, mask)
    # Translation of +2 in old local x becomes +2 in global y.
    np.testing.assert_allclose(result.get_trans(), [[5, 2, 0], [1, 2, 3]], atol=1e-7)
    np.testing.assert_allclose(
        result.get_rots().apply(jnp.array([[1.0, 0, 0], [1.0, 0, 0]])),
        [[-1, 0, 0], [1, 0, 0]],
        atol=1e-7,
    )
    unmasked = frame.compose_q_update_vec(update)
    assert jnp.all(jnp.isfinite(unmasked.to_tensor_7()))
    gradient = jax.grad(
        lambda u: jnp.sum(frame.compose_q_update_vec(u).to_tensor_4x4())
    )(jnp.zeros((2, 6)))
    assert jnp.all(jnp.isfinite(gradient))


def test_geometry_is_a_pytree_supporting_jit_grad_and_vmap():
    frame = Rigid.identity((5,), dtype=jnp.float64)
    points = jnp.arange(15, dtype=jnp.float64).reshape(5, 3)
    mapped = jax.jit(lambda f, p: f.apply(p))(frame, points)
    np.testing.assert_allclose(mapped, points)
    np.testing.assert_allclose(jax.vmap(lambda f, p: f.apply(p))(frame, points), points)
    gradients = jax.grad(lambda f: jnp.sum(f.apply(points)))(frame)
    assert isinstance(gradients, Rigid)
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(gradients)
    )
    np.testing.assert_allclose(gradients.get_trans(), jnp.ones((5, 3)))
    stopped = jax.grad(lambda f: jnp.sum(f.stop_rot_gradient().apply(points)))(frame)
    np.testing.assert_array_equal(stopped.get_rots().get_cur_rot(), 0)
    np.testing.assert_allclose(stopped.get_trans(), 1)


def test_missing_quaternions_and_small_angle_gradients_are_finite():
    quats = jnp.array([[0.0, 0, 0, 0], [2.0, 0, 0, 0]])
    mask = jnp.array([False, True])
    rotation = Rotation(quats=quats, quats_mask=mask)
    np.testing.assert_allclose(rotation.get_quats(), [[0, 0, 0, 0], [1, 0, 0, 0]])
    assert jnp.all(jnp.isfinite(invert_quat(quats, mask)))
    np.testing.assert_allclose(Rotation(quats=jnp.zeros(4)).get_quats(), [1, 0, 0, 0])
    gradient = jax.grad(lambda q: Rotation(quats=q).get_rotvec().sum())(
        jnp.array([1.0, 0, 0, 0])
    )
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(Rotation.identity(()).get_rotvec(), 0)
    np.testing.assert_allclose(
        Rotation(quats=jnp.array([1.0, 0, 0, 1])).get_rotvec(), [0, 0, np.pi / 2]
    )


def test_backbone_reference_frame_orientation():
    ca = jnp.array([2.0, 3, 4])
    n = ca + jnp.array([-0.5, 0, 1.5])
    c = ca + jnp.array([1.5, 0, 0])
    frame = Rigid.make_transform_from_reference(n, ca, c)
    np.testing.assert_allclose(frame.invert_apply(c), [1.5, 0, 0], atol=1e-7)
    np.testing.assert_allclose(frame.invert_apply(n), [-0.5, 1.5, 0], atol=1e-7)
    np.testing.assert_allclose(frame.get_trans(), ca)
    matrix = frame.get_rots().get_rot_mats()
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(jnp.linalg.det(matrix), 1)


def test_batch_indexing_reshape_concatenation_and_mapping():
    frame = Rigid.identity((2, 3)).apply_trans_fn(lambda t: t + 1)
    assert frame[:, 1].shape == (2,)
    assert frame[..., None, None].shape == (2, 3, 1, 1)
    assert frame.unsqueeze(2).shape == (2, 3, 1)
    assert frame.reshape((6, 4), (6, 3)).shape == (6,)
    joined = Rigid.cat([frame[:, :1], frame[:, 1:]], dim=-1)
    np.testing.assert_allclose(joined.to_tensor_4x4(), frame.to_tensor_4x4())
    # Summing masked components selects one coordinate system from a one-hot axis.
    selected = (frame * jnp.array([[0.0, 1, 0], [1.0, 0, 0]])).map_tensor_fn(
        lambda a: a.sum(-1)
    )
    np.testing.assert_allclose(
        selected.to_tensor_4x4(),
        Rigid.identity((2,)).to_tensor_4x4().at[..., :3, 3].set(1),
    )
    np.testing.assert_allclose(frame.scale_translation(0.1).get_trans(), 0.1)
    moved = frame.to(device="cpu", dtype=jnp.float64)
    assert moved.dtype == jnp.float64
    assert moved.get_rots().dtype == jnp.float64


def test_invalid_geometry_shapes_fail_clearly():
    for kwargs in (
        {},
        {"rot_mats": jnp.eye(3), "quats": jnp.ones(4)},
        {"quats": jnp.ones(3)},
    ):
        with pytest.raises(ValueError):
            Rotation(**kwargs)
    with pytest.raises(ValueError):
        create_rigid(jnp.eye(3), jnp.zeros((2, 3)))
    with pytest.raises(ValueError):
        Rigid.from_tensor_4x4(jnp.zeros((3, 3)))
    with pytest.raises(ValueError):
        Rigid.identity((2,)).unsqueeze(4)
    np.testing.assert_allclose(identity_quats((2,)), [[1, 0, 0, 0], [1, 0, 0, 0]])
