"""Check learned edge compression, feature modes, and padded-neighbor behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_ipa_jax.edge_embedder import EDGE_MODES, EdgeEmbedder, EdgeEmbedderConfig
from flash_ipa_jax.factorizer import LinearFactorizer
from flash_ipa_jax.structure.transitions import EdgeTransition


def _norm(x, p):
    return (
        (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    ) * np.asarray(p["weight"]) + np.asarray(p["bias"])


def test_factorizer_matches_independent_numpy_projection_and_is_trainable():
    module = LinearFactorizer(7, 4, target_rank=3, target_inner_dim=5)
    params = module.init(jax.random.key(0), dtype=jnp.float64)
    x = jnp.asarray(np.random.default_rng(4).normal(size=(2, 5, 5, 4)))
    first, second = jax.jit(module.apply)(params, x)
    padded = np.pad(np.asarray(x), ((0, 0), (0, 2), (0, 2), (0, 0)))
    length = np.asarray(params["length_compressor"]["weight"])
    inner = np.asarray(params["inner_compressor"]["weight"])
    references = []
    # Row/column contractions are intentionally written as einsums, separately
    # from the implementation's transpose-and-project sequence.
    for compressed in (
        np.einsum("bijd,ir->bjdr", padded, length)[:, :5],
        np.einsum("bijd,jr->bidr", padded, length)[:, :5],
    ):
        compressed = _norm(compressed, params["length_norm"])
        compressed = np.einsum("bncr,cd->bnrd", compressed, inner)
        compressed = _norm(compressed, params["inner_norm"])
        references.append(
            compressed.transpose(0, 3, 1, 2).reshape(10, 5, 3) / np.sqrt(3)
        )
    np.testing.assert_allclose(first, references[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(second, references[1], rtol=1e-10, atol=1e-10)
    gradients = jax.grad(lambda p: jnp.sum(module.apply(p, x)[0] ** 3))(params)
    assert np.linalg.norm(gradients["length_compressor"]["weight"]) > 1e-5
    with pytest.raises(ValueError, match="exceeds"):
        module.apply(params, jnp.zeros((1, 8, 8, 4)))


@pytest.mark.parametrize("mode", EDGE_MODES)
def test_edge_modes_shapes_masks_and_jit(mode):
    config = EdgeEmbedderConfig(
        c_s=8, c_p=4, feat_dim=4, num_bins=5, k=6, z_factor_rank=2, max_len=7, mode=mode
    )
    module = EdgeEmbedder(config)
    params = module.init(jax.random.key(1))
    nodes = jnp.arange(40, dtype=jnp.float32).reshape(1, 5, 8) / 40
    points = jnp.arange(15, dtype=jnp.float32).reshape(1, 5, 3) / 3
    mask = jnp.array([[1, 1, 0, 1, 0]], bool)
    dense, first, second, edge_mask = jax.jit(module.apply)(
        params, nodes, points, jnp.zeros_like(points), mask
    )
    if "no_bias" in mode:
        assert (dense, first, second, edge_mask) == (None, None, None, None)
    elif mode == "orig_2d_bias":
        assert dense.shape == (1, 5, 5, 4)
        np.testing.assert_array_equal(
            edge_mask, np.asarray(mask)[:, :, None] & np.asarray(mask)[:, None, :]
        )
        np.testing.assert_array_equal(dense[:, 2], 0)
        np.testing.assert_array_equal(dense[:, :, 4], 0)
    else:
        assert dense is None and edge_mask is None
        assert first.shape == second.shape == (1, 5, 2, 4)
        np.testing.assert_array_equal(first[:, 2], 0)
        np.testing.assert_array_equal(second[:, 4], 0)


def test_nearest_neighbor_features_ignore_padding_and_support_single_node():
    module = EdgeEmbedder(EdgeEmbedderConfig(c_s=8, c_p=4, feat_dim=4, num_bins=5, k=4))
    params = module.init(jax.random.key(2))
    nodes = jnp.ones((1, 4, 8))
    points = jnp.array(
        [[[0.0, 0.0, 0.0], [3.0, 1.0, 0.0], [5.0, 1.0, 1.0], [0.1, 0.0, 0.0]]]
    )
    mask = jnp.array([[1, 1, 1, 0]], bool)
    reference = module.apply(params, nodes, points, points, mask)
    changed = module.apply(
        params,
        nodes.at[:, 3].set(1000),
        points.at[:, 3].set(500),
        points.at[:, 3].set(-700),
        mask,
    )
    np.testing.assert_allclose(reference[1], changed[1], atol=1e-6)
    np.testing.assert_allclose(reference[2], changed[2], atol=1e-6)
    single = module.apply(
        params, nodes[:, :1], points[:, :1], points[:, :1], mask[:, :1]
    )
    assert single[1].shape == (1, 1, 2, 4)
    assert np.isfinite(single[1]).all()


@pytest.mark.parametrize("mode", ["1d", "2d"])
def test_edge_transition_final_projection_gets_gradient(mode):
    module = EdgeTransition(
        mode=mode, node_embed_size=8, edge_embed_in=4, edge_embed_out=4, z_factor_rank=2
    )
    params = module.init(jax.random.key(4))
    nodes = jax.random.normal(jax.random.key(5), (1, 3, 8))
    factors = jax.random.normal(jax.random.key(6), (1, 3, 2, 4))
    edges = jax.random.normal(jax.random.key(7), (1, 3, 3, 4))

    def output(p):
        return (
            module.apply(p, nodes, edges)
            if mode == "2d"
            else module.apply(p, nodes, None, factors, factors)[0]
        )

    np.testing.assert_array_equal(output(params), 0)
    target = jax.random.normal(jax.random.key(8), output(params).shape)
    gradient = jax.grad(lambda p: jnp.sum(output(p) * target))(params)
    assert np.linalg.norm(gradient["final_layer"]["weight"]) > 1e-5
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(gradient))
