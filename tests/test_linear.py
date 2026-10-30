"""Protect initialization and projection behavior while sharing implementations."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_ipa_jax.linear import (
    Linear,
    final_init,
    glorot_uniform_init,
    he_normal_init,
    init_linear,
    lecun_normal_init,
    linear,
)


@pytest.mark.parametrize(
    "mode,initialize",
    [
        ("default", lecun_normal_init),
        ("relu", he_normal_init),
        ("xavier", glorot_uniform_init),
        ("final", final_init),
    ],
)
def test_functional_initialization_uses_split_weight_key(mode, initialize):
    key = jax.random.key(21)
    weight_key, _ = jax.random.split(key)
    params = init_linear(key, 7, 5, init=mode, dtype=jnp.float64)
    np.testing.assert_array_equal(
        params["weight"], initialize(weight_key, (7, 5), jnp.float64)
    )
    np.testing.assert_array_equal(params["bias"], jnp.zeros(5))
    assert params["weight"].dtype == params["bias"].dtype == jnp.float64
    without_bias = init_linear(key, 7, 5, init=mode, bias=False)
    assert set(without_bias) == {"weight"}


def test_torch_initialization_preserves_independent_uniform_weights_and_bias():
    key = jax.random.key(22)
    weight_key, bias_key = jax.random.split(key)
    params = init_linear(key, 7, 5, init="torch", dtype=jnp.float64)
    expected_weight = jax.random.uniform(
        weight_key, (7, 5), jnp.float64, minval=-1, maxval=1
    ) / math.sqrt(7)
    expected_bias = jax.random.uniform(
        bias_key, (5,), jnp.float64, minval=-1, maxval=1
    ) / math.sqrt(7)
    np.testing.assert_array_equal(params["weight"], expected_weight)
    np.testing.assert_array_equal(params["bias"], expected_bias)
    assert jnp.any(params["bias"] != 0)
    assert jnp.all(jnp.abs(params["weight"]) <= 1 / math.sqrt(7))
    without_bias = init_linear(key, 7, 5, init="torch", bias=False, dtype=jnp.float64)
    np.testing.assert_array_equal(without_bias["weight"], expected_weight)
    assert "bias" not in without_bias


def test_shared_projection_preserves_leading_axes_and_gradients():
    params = {
        "weight": jnp.arange(12, dtype=jnp.float64).reshape(3, 4),
        "bias": jnp.arange(4, dtype=jnp.float64),
    }
    inputs = jnp.arange(36, dtype=jnp.float64).reshape(2, 2, 3, 3)
    result = jax.jit(linear)(params, inputs)
    np.testing.assert_allclose(
        result, np.asarray(inputs) @ params["weight"] + params["bias"]
    )
    assert result.shape == (2, 2, 3, 4)
    assert result.dtype == jnp.float64
    gradients = jax.jit(jax.grad(lambda p: jnp.sum(linear(p, inputs))))(params)
    np.testing.assert_allclose(
        gradients["weight"], inputs.sum((0, 1, 2))[:, None].repeat(4, 1)
    )
    np.testing.assert_allclose(gradients["bias"], 12)
    np.testing.assert_allclose(
        linear({"weight": params["weight"]}, inputs),
        np.asarray(inputs) @ params["weight"],
    )


def test_linear_class_keeps_direct_keys_custom_initialization_and_bias_choice():
    key = jax.random.key(23)
    layer = Linear(3, 4)
    params = layer.init(key, dtype=jnp.float64)
    np.testing.assert_array_equal(
        params["weight"], lecun_normal_init(key, (3, 4), jnp.float64)
    )

    def custom_initializer(key, shape, dtype):
        return jax.random.normal(key, shape, dtype)

    custom = Linear(3, 4, init="gating", init_fn=custom_initializer)
    custom_params = custom.init(key, dtype=jnp.float64)
    np.testing.assert_array_equal(
        custom_params["weight"], custom_initializer(key, (3, 4), jnp.float64)
    )
    np.testing.assert_array_equal(custom_params["bias"], 0)
    gating = Linear(3, 4, init="gating")
    np.testing.assert_array_equal(gating.apply(gating.init(key), jnp.ones((2, 3))), 1)

    # An explicitly biasless layer ignores a bias supplied by another checkpoint.
    without_bias = Linear(3, 4, bias=False)
    np.testing.assert_allclose(
        without_bias.apply(params, jnp.ones((2, 3))),
        jnp.ones((2, 3)) @ params["weight"],
    )
