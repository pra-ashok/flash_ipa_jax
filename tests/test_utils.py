"""Known feature examples and initialization behavior for the JAX port."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_ipa_jax.linear import Linear, he_normal_init, lecun_normal_init
from flash_ipa_jax.layer import IPAConfig
from flash_ipa_jax.utils import (
    ANG_TO_NM_SCALE,
    NM_TO_ANG_SCALE,
    calc_distogram,
    calc_distogram_knn,
    check_config_ipa,
    get_index_embedding,
    knn_indices,
)


def test_distograms_match_upstream_strict_bin_boundaries():
    positions = jnp.array([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [5.0, 0, 0]]])
    result = jax.jit(lambda p: calc_distogram(p, 0.0, 4.0, 3))(positions)
    np.testing.assert_array_equal(
        result[0, 0], [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]]
    )
    np.testing.assert_array_equal(result, jnp.swapaxes(result, -3, -2))
    np.testing.assert_allclose(ANG_TO_NM_SCALE * NM_TO_ANG_SCALE, 1)


def test_nearest_neighbors_exclude_self_and_support_padding():
    positions = jnp.array([[[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [7.0, 0, 0]]])
    distances, indices = jax.jit(lambda p: knn_indices(p, 2))(positions)
    np.testing.assert_allclose(distances[0, 0], [1, 3])
    np.testing.assert_array_equal(indices[0, 0], [1, 2])
    assert not jnp.any(indices == jnp.arange(4)[None, :, None])
    mask = jnp.array([[True, True, False, False]])
    distances, indices = knn_indices(positions, 2, mask=mask)
    assert distances[0, 0, 0] == 1
    assert jnp.all(jnp.isinf(distances[0, 2:]))
    bins, gathered = calc_distogram_knn(positions, 2, 0, 4, 3, mask=mask)
    np.testing.assert_array_equal(gathered, indices)
    np.testing.assert_array_equal(bins[0, 0, 0], [1, 0, 0])
    np.testing.assert_array_equal(bins[0, 0, 1], 0)
    np.testing.assert_array_equal(bins[0, 2:], 0)


def test_position_embeddings_support_offsets_and_arbitrary_batch_axes():
    positions = jnp.array([[0, 1, -1], [2, 3, 4]])
    result = jax.jit(lambda p: get_index_embedding(p, 4))(positions)
    assert result.shape == (2, 3, 4)
    np.testing.assert_allclose(result[0, 0], [0, 0, 1, 1])
    np.testing.assert_allclose(result[0, 1, :2], -result[0, 2, :2])
    np.testing.assert_allclose(result[0, 1, 2:], result[0, 2, 2:])
    with pytest.raises(ValueError, match="even"):
        get_index_embedding(positions, 3)
    with pytest.raises(ValueError, match="1 <= k"):
        knn_indices(jnp.zeros((1, 3, 3)), 3)


@pytest.mark.parametrize(
    "mode,c_z,rank,flash",
    [
        ("orig_no_bias", 0, 0, False),
        ("orig_2d_bias", 8, 0, False),
        ("flash_no_bias", 0, 0, True),
        ("flash_1d_bias", 8, 2, True),
        ("flash_2d_factorize_bias", 8, 2, True),
    ],
)
def test_upstream_config_modes(mode, c_z, rank, flash):
    config = dict(c_z=c_z, z_factor_rank=rank, use_flash_attn=flash)
    check_config_ipa(config, mode)
    check_config_ipa(SimpleNamespace(**config), mode)
    with pytest.raises(ValueError, match="incompatible"):
        check_config_ipa(dict(config, use_flash_attn=not flash), mode)


def test_jax_config_validation_uses_pair_mode_and_separate_backend():
    no_pairs = IPAConfig(c_z=0, pair_mode="none")
    check_config_ipa(no_pairs, "orig_no_bias")
    check_config_ipa(no_pairs, "flash_no_bias")
    check_config_ipa(IPAConfig(pair_mode="dense"), "orig_2d_bias")
    check_config_ipa(IPAConfig(pair_mode="factorized"), "flash_1d_bias")
    with pytest.raises(ValueError, match="incompatible"):
        check_config_ipa(no_pairs, "orig_2d_bias")


def test_linear_layers_are_trainable_and_use_requested_initializers():
    inputs = jnp.ones((3, 4))
    gating = Linear(4, 6, init="gating")
    np.testing.assert_allclose(gating.apply(gating.init(jax.random.key(1)), inputs), 1)
    final = Linear(4, 6, init="final")
    params = final.init(jax.random.key(2))
    np.testing.assert_allclose(final.apply(params, inputs), 0)
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(final.apply(p, inputs))))(params)
    # A zero initialized residual output must still receive a learning signal.
    assert jnp.linalg.norm(gradient["weight"]) > 0
    assert jnp.linalg.norm(gradient["bias"]) > 0
    no_bias = Linear(4, 6, bias=False)
    params = no_bias.init(jax.random.key(3))
    assert "bias" not in params
    assert no_bias.apply(params, inputs).shape == (3, 6)


def test_truncated_initializers_have_requested_variance():
    shape = (128, 512)
    for initializer, variance in (
        (lecun_normal_init, 1 / 128),
        (he_normal_init, 2 / 128),
    ):
        weights = initializer(jax.random.key(4), shape)
        np.testing.assert_allclose(np.var(weights), variance, rtol=0.05)
        assert abs(float(jnp.mean(weights))) < 0.005
