"""Opt-in real GPU checks: FLASH_IPA_GPU_BACKEND=flash_attn3 or jax_aiter."""

import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flash_ipa_jax import BackendConfig, ipa_attention
from helpers import inputs


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["flash_attn3", "jax_aiter"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("causal", [False, True])
def test_real_gpu_forward_and_vjp(name, dtype, causal):
    if os.environ.get("FLASH_IPA_GPU_BACKEND") != name:
        pytest.skip("set FLASH_IPA_GPU_BACKEND to run a real kernel; no emulation")
    # Strict mode prevents a missing GPU/package from turning this into a CPU test.
    config = BackendConfig(name=name, kernel_dtype=dtype, fallback="error")
    args, factors = inputs(n=64, dtype=jnp.float32)

    def fused(a, f):
        return ipa_attention(*a, factors=f, backend=config, causal=causal)

    def native(a, f):
        return ipa_attention(*a, factors=f, causal=causal)

    out, reference = jax.jit(fused)(args, factors), jax.jit(native)(args, factors)
    tol = 0.025 if dtype == "bfloat16" else 0.004
    for x, y in zip(jax.tree.leaves(out), jax.tree.leaves(reference)):
        np.testing.assert_allclose(x, y, rtol=tol, atol=tol / 10)

    def loss(fn, a, f):
        return sum(jnp.mean(x * x) + 0.2 * jnp.mean(x) for x in fn(a, f)[:3])

    actual = jax.jit(jax.grad(lambda a, f: loss(fused, a, f), argnums=(0, 1)))(
        args, factors
    )
    expected = jax.jit(jax.grad(lambda a, f: loss(native, a, f), argnums=(0, 1)))(
        args, factors
    )
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(x).all()
        scale = max(float(jnp.max(jnp.abs(y))), 1e-5)
        np.testing.assert_allclose(x, y, rtol=tol * 3, atol=scale * tol * 3)
    # Global points are equivariant; scalar and pair outputs are invariant.
    rotation, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))
    rotation = jnp.asarray(rotation * np.linalg.det(rotation), jnp.float32)
    shift = jnp.array([2.0, -3.0, 4.0], jnp.float32)
    transformed = list(args)
    for index in (3, 4, 5):
        transformed[index] = jnp.einsum("ij,...j->...i", rotation, args[index]) + shift
    moved = jax.jit(fused)(tuple(transformed), factors)
    np.testing.assert_allclose(moved.scalar, out.scalar, rtol=tol * 2, atol=tol / 5)
    np.testing.assert_allclose(moved.pair, out.pair, rtol=tol * 2, atol=tol / 5)
    expected_points = jnp.einsum("ij,...j->...i", rotation, out.points) + shift
    np.testing.assert_allclose(
        moved.points, expected_points, rtol=tol * 2, atol=tol / 5
    )
