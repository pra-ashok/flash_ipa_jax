import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flash_ipa_jax import (
    IPAConfig,
    InvariantPointAttention,
    BackendConfig,
    PairFactors,
    from_torch_state_dict,
)
from helpers import numpy_reference


def fixture(mode):
    c = IPAConfig(
        c_s=7,
        c_z=8 if mode != "none" else 0,
        c_hidden=3,
        no_heads=2,
        no_qk_points=2,
        no_v_points=3,
        z_factor_rank=2,
        pair_mode=mode,
    )
    layer = InvariantPointAttention(c, BackendConfig(chunk_size=2))
    params = layer.init(jax.random.key(1), zero_init_output=False, dtype=jnp.float64)
    rng = np.random.default_rng(8)
    # Nonzero projection biases exercise the upstream factor semantics.
    for value in params.values():
        if isinstance(value, dict):
            value["bias"] = jnp.array(rng.normal(size=value["bias"].shape) * 0.1)
    s = jnp.array(rng.normal(size=(2, 5, 7)) * 0.2)
    r, _ = np.linalg.qr(rng.normal(size=(2, 5, 3, 3)))
    r = r * np.linalg.det(r)[..., None, None]
    r = jnp.array(r)
    t = jnp.array(rng.normal(size=(2, 5, 3)))
    kwargs = {}
    if mode == "dense":
        kwargs["z"] = jnp.array(rng.normal(size=(2, 5, 5, 8)))
    elif mode == "factorized":
        kwargs.update(
            z_factor_1=jnp.array(rng.normal(size=(2, 5, 2, 8))),
            z_factor_2=jnp.array(rng.normal(size=(2, 5, 2, 8))),
        )
    return layer, params, s, r, t, kwargs


def numpy_layer(c, params, s, r, t, kwargs):
    def linear(name, x):
        return np.asarray(x) @ np.asarray(params[name]["weight"]) + np.asarray(
            params[name]["bias"]
        )

    b, n, _ = s.shape
    q = linear("linear_q", s).reshape(b, n, c.no_heads, c.c_hidden)
    k, v = np.split(
        linear("linear_kv", s).reshape(b, n, c.no_heads, 2 * c.c_hidden), 2, -1
    )

    def points(name, num):
        x = linear(name, s)
        x = np.stack(np.split(x, 3, axis=-1), axis=-1).reshape(b, n, c.no_heads, num, 3)
        return np.einsum("bnij,bnhpj->bnhpi", r, x) + t[:, :, None, None, :]

    qp = points("linear_q_points", c.no_qk_points)
    kp, vp = np.split(
        points("linear_kv_points", c.no_qk_points + c.no_v_points),
        [c.no_qk_points],
        axis=-2,
    )
    pairs = {}
    if c.pair_mode == "dense":
        pairs.update(
            pair_bias=linear("linear_b", kwargs["z"]) / np.sqrt(3),
            pair_values=linear("down_z", kwargs["z"]),
        )
    elif c.pair_mode == "factorized":
        z1, z2 = kwargs["z_factor_1"], kwargs["z_factor_2"]
        pairs["factors"] = PairFactors(
            linear("linear_b", z1).swapaxes(-1, -2),
            linear("linear_b", z2).swapaxes(-1, -2),
            linear("down_z", z1),
            linear("down_z", z2),
        )
    o, pt, pair, valid = numpy_reference(
        (q, k, v, qp, kp, vp, params["head_weights"]), mask=kwargs.get("mask"), **pairs
    )
    local = np.einsum("bnji,bnhpj->bnhpi", r, pt - t[:, :, None, None, :])
    norm = np.sqrt(np.sum(local**2, -1) + c.eps)
    feats = [
        o.reshape(b, n, -1),
        *(local[..., a].reshape(b, n, -1) for a in range(3)),
        norm.reshape(b, n, -1),
    ]
    if pair is not None:
        feats.append(pair.reshape(b, n, -1))
    return np.where(
        valid[..., None], linear("linear_out", np.concatenate(feats, -1)), 0
    )


@pytest.mark.parametrize("mode", ["none", "dense", "factorized"])
def test_layer_against_numpy_and_checkpoint_layout(mode):
    layer, p, s, r, t, kw = fixture(mode)
    state = {"ipa.head_weights": np.array(p["head_weights"])}
    for name, x in p.items():
        if isinstance(x, dict):
            state["ipa." + name + ".weight"] = np.array(x["weight"].T)
            state["ipa." + name + ".bias"] = np.array(x["bias"])
    converted = from_torch_state_dict(state, layer.config, prefix="ipa.")
    kw["mask"] = jnp.array([[1, 0, 1, 1, 1], [1, 1, 0, 1, 0]], bool)
    result = jax.jit(lambda p, s, r, t: layer.apply(p, s, r, t, **kw))(
        converted, s, r, t
    )
    np.testing.assert_allclose(
        result, numpy_layer(layer.config, p, s, r, t, kw), rtol=1e-9, atol=1e-9
    )
    assert np.max(np.abs(result)) > 0.01


@pytest.mark.parametrize("mode", ["none", "dense", "factorized"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_se3_invariance_nontrivial_output(mode, dtype):
    layer, p, s, r, t, kw = fixture(mode)
    p, s, r, t, kw = jax.tree.map(lambda x: x.astype(dtype), (p, s, r, t, kw))
    kw["mask"] = jnp.array([[1, 0, 1, 1, 1], [0, 0, 0, 0, 0]], bool)
    g, _ = np.linalg.qr(np.random.default_rng(93).normal(size=(3, 3)))
    g = jnp.array(g * np.linalg.det(g), dtype)
    shift = jnp.array([12, -8, 31], dtype)
    r2 = jnp.einsum("ij,bnjk->bnik", g, r)
    t2 = jnp.einsum("ij,bnj->bni", g, t) + shift
    a = layer.apply(p, s, r, t, **kw)
    b = layer.apply(p, s, r2, t2, **kw)
    np.testing.assert_allclose(
        a,
        b,
        rtol=5e-5 if dtype == jnp.float32 else 1e-10,
        atol=2e-5 if dtype == jnp.float32 else 1e-10,
    )
    np.testing.assert_array_equal(a[1], 0)


def test_parameter_and_frame_gradients_nonzero_and_finite_difference():
    layer, p, s, r, t, kw = fixture("factorized")

    def loss(p, s, r, t):
        return jnp.sum(layer.apply(p, s, r, t, **kw) ** 2)

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(p, s, r, t)
    rng = np.random.default_rng(17)
    state = (p, s, r, t)
    tangent = jax.tree.map(lambda x: jnp.array(rng.normal(size=x.shape)), state)
    for g in jax.tree.leaves(grads):
        assert np.isfinite(g).all()
    assert np.linalg.norm(grads[0]["head_weights"]) > 1e-6
    expected = sum(
        jnp.sum(x * y) for x, y in zip(jax.tree.leaves(grads), jax.tree.leaves(tangent))
    )
    plus = jax.tree.map(lambda x, d: x + 1e-5 * d, state, tangent)
    minus = jax.tree.map(lambda x, d: x - 1e-5 * d, state, tangent)
    actual = (loss(*plus) - loss(*minus)) / 2e-5
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_initialization_zero_and_validation():
    layer, _, s, r, t, kw = fixture("factorized")
    p = layer.init(jax.random.key(12))
    np.testing.assert_array_equal(layer.apply(p, s, r, t, **kw), 0)
    with pytest.raises(ValueError, match="both z factors"):
        layer.apply(p, s, r, t)
    with pytest.raises(ValueError):
        IPAConfig(pair_mode="none", c_z=8)
