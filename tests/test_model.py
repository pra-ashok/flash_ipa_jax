"""Verify the complete structure trunk, coordinate system equivariance, and training gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_ipa_jax.backends import BackendConfig
from flash_ipa_jax.edge_embedder import EDGE_MODES, EdgeEmbedderConfig
from flash_ipa_jax.layer import IPAConfig
from flash_ipa_jax.model import Model, ModelConfig
from flash_ipa_jax.structure.sequence import SequenceTransformer


def _fixture(mode="flash_1d_bias", blocks=2):
    pair_mode = (
        "none"
        if "no_bias" in mode
        else "dense" if mode == "orig_2d_bias" else "factorized"
    )
    ipa = IPAConfig(
        c_s=8,
        c_z=0 if pair_mode == "none" else 4,
        c_hidden=2,
        no_heads=2,
        no_qk_points=2,
        no_v_points=2,
        pair_mode=pair_mode,
    )
    edges = EdgeEmbedderConfig(
        c_s=8, c_p=4, feat_dim=4, num_bins=5, k=2, max_len=5, mode=mode
    )
    model = Model(
        ModelConfig(
            mode=mode,
            node_embed_size=8,
            edge_embed_size=4,
            ipa=ipa,
            edge_features=edges,
            num_blocks=blocks,
            seq_tfmr_num_heads=2,
            seq_tfmr_num_layers=1,
        ),
        BackendConfig(chunk_size=2),
    )
    params = model.init(jax.random.key(0))
    points = jnp.asarray(np.random.default_rng(12).normal(size=(2, 4, 3)), jnp.float32)
    inputs = {
        "node_embeddings": jax.random.normal(jax.random.key(1), (2, 4, 8)),
        "translations": points,
        "rotations": jnp.broadcast_to(jnp.eye(3), (2, 4, 3, 3)),
        "node_mask": jnp.array([[1, 1, 0, 1], [0, 0, 0, 0]], bool),
    }
    return model, params, inputs


@pytest.mark.parametrize("mode", EDGE_MODES)
def test_full_model_zero_initialized_frames_jit_and_masks(mode):
    model, params, inputs = _fixture(mode)
    outputs = jax.jit(model.apply)(params, inputs)
    np.testing.assert_allclose(outputs["pred_trans"], inputs["translations"], atol=1e-6)
    np.testing.assert_allclose(outputs["pred_rotmats"], inputs["rotations"], atol=1e-6)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(outputs))


def test_nonzero_frame_updates_are_equivariant_and_padded_frames_stay_fixed():
    model, params, inputs = _fixture()
    # Exercise attention, sequence, and coordinate system updates, not just the identity
    # output of the intentionally zero-initialized final projections.
    for index, block in enumerate(params["trunk"]):
        for name in ("post_tfmr",):
            block[name]["weight"] = (
                jax.random.normal(jax.random.key(30 + index), (8, 8)) * 0.05
            )
        block["ipa"]["linear_out"]["weight"] = (
            jax.random.normal(
                jax.random.key(40 + index), block["ipa"]["linear_out"]["weight"].shape
            )
            * 0.03
        )
        block["bb_update"]["linear"]["weight"] = (
            jax.random.normal(jax.random.key(50 + index), (8, 6)) * 0.03
        )
    angle = 0.7
    rotation = jnp.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        jnp.float32,
    )
    shift = jnp.array([10.0, -4.0, 6.0])
    changed = dict(
        inputs,
        rotations=jnp.einsum("ij,bnjk->bnik", rotation, inputs["rotations"]),
        translations=jnp.einsum("ij,bnj->bni", rotation, inputs["translations"])
        + shift,
    )
    original, moved = model.apply(params, inputs), model.apply(params, changed)
    np.testing.assert_allclose(
        moved["pred_trans"],
        jnp.einsum("ij,bnj->bni", rotation, original["pred_trans"]) + shift,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        moved["pred_rotmats"],
        jnp.einsum("ij,bnjk->bnik", rotation, original["pred_rotmats"]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.max(np.abs(original["pred_trans"] - inputs["translations"])) > 0.01
    invalid = ~np.asarray(inputs["node_mask"])
    np.testing.assert_allclose(
        np.asarray(original["pred_trans"])[invalid],
        np.asarray(inputs["translations"])[invalid],
        atol=1e-6,
    )


def test_complete_model_training_step_lowers_loss_and_has_finite_gradients():
    model, params, inputs = _fixture(blocks=1)
    mask = inputs["node_mask"][..., None]
    target = inputs["translations"] + jnp.array([0.3, -0.4, 0.1])

    def loss(p):
        error = model.apply(p, inputs)["pred_trans"] - target
        return jnp.sum(jnp.where(mask, error**2, 0)) / jnp.sum(mask)

    before, gradients = jax.jit(jax.value_and_grad(loss))(params)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(gradients))
    assert np.linalg.norm(gradients["trunk"][0]["bb_update"]["linear"]["weight"]) > 1e-5
    updated = jax.tree.map(lambda value, grad: value - 1e-4 * grad, params, gradients)
    assert loss(updated) < before


def test_active_trunk_backpropagates_through_ipa_and_sequence_attention():
    model, params, inputs = _fixture(blocks=1)
    block = params["trunk"][0]
    for key, seed in (("post_tfmr", 61),):
        block[key]["weight"] = jax.random.normal(jax.random.key(seed), (8, 8)) * 0.05
    output = block["ipa"]["linear_out"]
    output["weight"] = (
        jax.random.normal(jax.random.key(62), output["weight"].shape) * 0.04
    )
    block["bb_update"]["linear"]["weight"] = (
        jax.random.normal(jax.random.key(63), (8, 6)) * 0.03
    )
    target = inputs["translations"] + jnp.array([0.2, 0.1, -0.4])

    def loss(p):
        error = model.apply(p, inputs)["pred_trans"] - target
        return jnp.sum(jnp.where(inputs["node_mask"][..., None], error**2, 0))

    gradients = jax.jit(jax.grad(loss))(params)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(gradients))
    active = gradients["trunk"][0]
    assert np.linalg.norm(active["ipa"]["head_weights"]) > 1e-8
    assert np.linalg.norm(active["seq_tfmr"][0]["qkv"]["weight"]) > 1e-8
    assert np.linalg.norm(gradients["edge_embedder"]["linear_s_p"]["weight"]) > 1e-8


def test_sequence_transformer_matches_independent_numpy_and_all_padding():
    module = SequenceTransformer(4, heads=2, layers=1)
    params = module.init(jax.random.key(11), dtype=jnp.float64)
    nodes = np.random.default_rng(9).normal(size=(1, 3, 4))
    mask = jnp.array([[1, 1, 0]], bool)
    layer = jax.tree.map(np.asarray, params[0])

    def project(name, x):
        return x @ layer[name]["weight"] + layer[name]["bias"]

    def norm(name, x):
        p = layer[name]
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(
            x.var(-1, keepdims=True) + 1e-5
        ) * p["weight"] + p["bias"]

    reference = np.where(np.asarray(mask)[..., None], nodes, 0)
    q, k, v = np.split(project("qkv", reference), 3, -1)
    q, k, v = (value.reshape(1, 3, 2, 2) for value in (q, k, v))
    scores = np.einsum("bihd,bjhd->bhij", q, k) / np.sqrt(2)
    scores[..., 2] = -np.inf
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    attended = np.einsum("bhij,bjhd->bihd", weights, v).reshape(1, 3, 4)
    reference = norm("norm1", reference + project("out", attended))
    reference = norm(
        "norm2",
        reference + project("linear2", np.maximum(project("linear1", reference), 0)),
    )
    reference[:, 2] = 0
    np.testing.assert_allclose(
        module.apply(params, jnp.asarray(nodes), mask),
        reference,
        atol=1e-10,
        rtol=1e-10,
    )
    empty = jax.jit(module.apply)(params, jnp.asarray(nodes), jnp.zeros_like(mask))
    np.testing.assert_array_equal(empty, 0)


def test_configuration_infers_modes_and_rejects_mismatches():
    config = ModelConfig(mode="orig_no_bias", node_embed_size=8, num_blocks=1)
    assert config.ipa.pair_mode == "none" and config.ipa.c_z == 0
    with pytest.raises(ValueError, match="pair_mode"):
        ModelConfig(mode="orig_2d_bias", ipa=IPAConfig())
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(node_embed_size=7)
