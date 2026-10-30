"""Compare tiled online softmax to dense IPA, including training derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_ipa_jax import BackendConfig, ipa_attention
from helpers import inputs, numpy_reference


def assert_close(actual, expected, rtol=1e-9, atol=1e-10):
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)


@pytest.mark.parametrize("pairs", ["none", "dense", "factorized"])
@pytest.mark.parametrize("causal", [False, True])
def test_online_softmax_matches_independent_oracle_with_masks(pairs, causal):
    args, factors = inputs()
    rng = np.random.default_rng(81)
    options = {"causal": causal}
    if pairs == "factorized":
        options["factors"] = factors
    elif pairs == "dense":
        options["pair_bias"] = jnp.asarray(rng.normal(size=(2, 5, 5, 2)))
        options["pair_values"] = jnp.asarray(rng.normal(size=(2, 5, 5, 3)))

    # Empty batch item, isolated query, and holes exercise finite empty rows.
    options["mask"] = jnp.array([[1, 0, 1, 1, 1], [0, 0, 0, 0, 0]], bool)
    options["pair_mask"] = jnp.ones((2, 5, 5), bool).at[:, 3, :].set(False)
    config = BackendConfig(native_mode="flash", chunk_size=2, key_chunk_size=3)
    actual = jax.jit(lambda a: ipa_attention(*a, backend=config, **options))(args)
    assert_close(actual, numpy_reference(args, **options))


@pytest.mark.parametrize("dense_pairs", [False, True])
def test_tiled_all_input_gradients_match_dense(dense_pairs):
    args, factors = inputs(n=3)
    rng = np.random.default_rng(3)
    state = (
        (
            args,
            jnp.asarray(rng.normal(size=(2, 3, 3, 2))),
            jnp.asarray(rng.normal(size=(2, 3, 3, 3))),
        )
        if dense_pairs
        else (args, factors)
    )
    mask = jnp.array([[1, 0, 1], [0, 0, 0]], bool)

    def loss(state, mode):
        options = (
            {"pair_bias": state[1], "pair_values": state[2]}
            if dense_pairs
            else {"factors": state[1]}
        )
        output = ipa_attention(
            *state[0],
            mask=mask,
            causal=True,
            **options,
            backend=BackendConfig(native_mode=mode, chunk_size=2, key_chunk_size=2)
        )
        return sum(
            jnp.sum(value**2 + 0.13 * value)
            for value in output[:3]
            if value is not None
        )

    dense = jax.jit(jax.value_and_grad(lambda s: loss(s, "dense")))(state)
    tiled = jax.jit(jax.value_and_grad(lambda s: loss(s, "flash")))(state)
    assert_close(tiled, dense)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(tiled))


@pytest.mark.parametrize("tile", [1, 2, 5, 9])
def test_tail_tiles_and_float32(tile):
    args, factors = inputs(dtype=jnp.float32)
    expected = ipa_attention(
        *args, factors=factors, backend=BackendConfig(native_mode="dense")
    )
    config = BackendConfig(native_mode="flash", chunk_size=3, key_chunk_size=tile)
    actual = jax.jit(lambda a: ipa_attention(*a, factors=factors, backend=config))(args)
    assert_close(actual, expected, rtol=3e-6, atol=1e-7)


def test_extreme_logits_are_stable_and_fully_masked_gradients_are_zero():
    args, factors = inputs(n=3)
    args = (args[0] * 1000, args[1] * 1000, *args[2:])
    config = BackendConfig(chunk_size=2, key_chunk_size=1)

    def loss(a, f):
        result = ipa_attention(
            *a, factors=f, mask=jnp.zeros((2, 3), bool), backend=config
        )
        return sum(jnp.sum(x) for x in result[:3])

    result = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(args, factors)
    for value in jax.tree.leaves(result):
        np.testing.assert_array_equal(value, jnp.zeros_like(value))

    actual = ipa_attention(*args, factors=factors, backend=config)
    expected = ipa_attention(
        *args, factors=factors, backend=BackendConfig(native_mode="dense")
    )
    assert_close(actual, expected)


def test_empty_dense_pair_channels_are_rejected_consistently():
    args, _ = inputs(n=3)
    for mode in ("dense", "chunked", "flash"):
        with pytest.raises(ValueError, match="C > 0"):
            ipa_attention(
                *args,
                pair_values=jnp.zeros((2, 3, 3, 0)),
                backend=BackendConfig(native_mode=mode)
            )
