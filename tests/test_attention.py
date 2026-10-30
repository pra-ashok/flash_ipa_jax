import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flash_ipa_jax import BackendConfig, PairFactors, ipa_attention, pack_qkv
from helpers import inputs, numpy_reference


def assert_tree_close(a, b, rtol=1e-9, atol=1e-10):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


@pytest.mark.parametrize("mode", ["dense", "chunked"])
@pytest.mark.parametrize("pairs", ["none", "dense", "factors"])
@pytest.mark.parametrize("causal", [False, True])
def test_forward_numpy(mode, pairs, causal):
    args, factors = inputs()
    rng = np.random.default_rng(3)
    kwargs = {}
    if pairs == "dense":
        kwargs.update(
            pair_bias=jnp.array(rng.normal(size=(2, 5, 5, 2))),
            pair_values=jnp.array(rng.normal(size=(2, 5, 5, 3))),
        )
    if pairs == "factors":
        kwargs["factors"] = factors
    mask = jnp.array([[1, 0, 1, 1, 1], [0, 0, 0, 0, 0]], bool)
    pm = jnp.ones((2, 5, 5), bool).at[:, 3, :].set(False).at[:, 0, 2].set(False)
    kwargs.update(mask=mask, pair_mask=pm, causal=causal)
    expected = numpy_reference(args, **kwargs)
    fun = jax.jit(
        lambda *xs: ipa_attention(
            *xs, backend=BackendConfig(native_mode=mode, chunk_size=2), **kwargs
        )
    )
    assert_tree_close(fun(*args), expected)


def packed_reference(args, factors, causal=False):
    q, k, v, origin = pack_qkv(*args, factors)
    logits = jnp.einsum("bihd,bjhd->bhij", q, k)
    if causal:
        logits = jnp.where(
            jnp.arange(q.shape[1])[None, :] <= jnp.arange(q.shape[1])[:, None],
            logits,
            -jnp.inf,
        )
    a = jax.nn.softmax(logits, -1)
    o = jnp.einsum("bhij,bjhd->bihd", a, v)
    dv, p = args[2].shape[-1], args[5].shape[-2]
    scalar = o[..., :dv]
    points = o[..., dv : dv + 3 * p].reshape(args[5].shape) + origin
    pair = jnp.einsum(
        "birc,bihrc->bihc",
        factors.value_q,
        o[..., dv + 3 * p :].reshape(*o.shape[:3], 2, 3),
    )
    return scalar, points, pair


def loss_outputs(result):
    return sum(jnp.sum(x * x) + 0.13 * jnp.sum(x) for x in result[:3] if x is not None)


@pytest.mark.parametrize("causal", [False, True])
def test_lift_outputs_and_all_gradients(causal):
    args, factors = inputs()
    native = lambda a, f: ipa_attention(
        *a, factors=f, causal=causal, backend=BackendConfig(chunk_size=2)
    )
    packed = lambda a, f: packed_reference(a, f, causal)
    assert_tree_close(native(args, factors)[:3], packed(args, factors))
    g1 = jax.jit(jax.grad(lambda a, f: loss_outputs(native(a, f)), argnums=(0, 1)))(
        args, factors
    )
    g2 = jax.jit(jax.grad(lambda a, f: loss_outputs(packed(a, f)), argnums=(0, 1)))(
        args, factors
    )
    assert_tree_close(g1, g2)


def test_all_input_gradients_against_independent_finite_difference():
    args, factors = inputs(n=3)
    state = (args, factors)
    loss = lambda state: loss_outputs(
        ipa_attention(*state[0], factors=state[1], backend=BackendConfig(chunk_size=2))
    )
    grad = jax.jit(jax.grad(loss))(state)
    leaves, treedef = jax.tree.flatten(state)
    gradients = jax.tree.leaves(grad)
    rng = np.random.default_rng(31)
    for i, x in enumerate(leaves):
        direction = rng.normal(size=x.shape)
        eps = 1e-5
        losses = []
        for sign in (-1, 1):
            perturbed = list(leaves)
            perturbed[i] = np.array(x) + sign * eps * direction
            aa, ff = jax.tree.unflatten(treedef, perturbed)
            oracle = numpy_reference(aa, ff)
            losses.append(sum(np.sum(y * y) + 0.13 * np.sum(y) for y in oracle[:3]))
        expected = (losses[1] - losses[0]) / (2 * eps)
        np.testing.assert_allclose(
            np.sum(gradients[i] * direction), expected, rtol=2e-6, atol=1e-8
        )


def test_dense_pair_bias_and_value_gradients():
    args, _ = inputs(n=3)
    rng = np.random.default_rng(5)
    bias = jnp.array(rng.normal(size=(2, 3, 3, 2)))
    values = jnp.array(rng.normal(size=(2, 3, 3, 3)))
    loss = lambda b, v: loss_outputs(
        ipa_attention(
            *args, pair_bias=b, pair_values=v, backend=BackendConfig(chunk_size=2)
        )
    )
    grads = jax.jit(jax.grad(loss, argnums=(0, 1)))(bias, values)
    for i, x in enumerate((bias, values)):
        direction = rng.normal(size=x.shape)
        vals = []
        for sign in (-1, 1):
            xs = [bias, values]
            xs[i] = x + sign * 1e-5 * direction
            oracle = numpy_reference(args, pair_bias=xs[0], pair_values=xs[1])
            vals.append(sum(np.sum(y * y) + 0.13 * np.sum(y) for y in oracle[:3]))
        np.testing.assert_allclose(
            np.sum(grads[i] * direction),
            (vals[1] - vals[0]) / 2e-5,
            rtol=2e-6,
            atol=1e-8,
        )


def test_fully_masked_zero_outputs_and_gradients():
    args, factors = inputs()

    def run(a, f):
        return ipa_attention(
            *a,
            factors=f,
            mask=jnp.zeros((2, 5), bool),
            backend=BackendConfig(chunk_size=2),
        )

    out = jax.jit(run)(args, factors)
    for x in jax.tree.leaves(out):
        assert np.all(np.asarray(x) == 0)
    gradients = jax.jit(jax.grad(lambda a, f: loss_outputs(run(a, f)), argnums=(0, 1)))(
        args, factors
    )
    for x in jax.tree.leaves(gradients):
        assert np.isfinite(x).all()
        np.testing.assert_array_equal(x, 0)


def test_masked_tokens_have_no_influence():
    args, factors = inputs()
    mask = jnp.ones((2, 5), bool).at[:, 1].set(False)
    out = ipa_attention(*args, factors=factors, mask=mask)
    changed = tuple(x.at[:, 1].set(1000) if x.ndim > 1 else x for x in args)
    ff = PairFactors(*(x.at[:, 1].set(-200) for x in factors))
    assert_tree_close(out, ipa_attention(*changed, factors=ff, mask=mask))


def test_chunk_sizes_and_fp32_precision():
    args, factors = inputs(dtype=jnp.float32)
    expected = ipa_attention(
        *args, factors=factors, backend=BackendConfig(native_mode="dense")
    )
    for size in (1, 3, 5, 9):
        out = jax.jit(
            lambda *a: ipa_attention(
                *a, factors=factors, backend=BackendConfig(chunk_size=size)
            )
        )(*args)
        assert_tree_close(out, expected, rtol=3e-6, atol=1e-7)


def test_invalid_shapes_masks_and_modes():
    args, factors = inputs()
    with pytest.raises(ValueError, match="boolean"):
        ipa_attention(*args, mask=jnp.ones((2, 5)))
    with pytest.raises(ValueError, match="not both"):
        ipa_attention(*args, factors=factors, pair_values=jnp.zeros((2, 5, 5, 3)))
    with pytest.raises(ValueError, match="shape"):
        ipa_attention(args[0], args[1][:, :-1], *args[2:])
    with pytest.raises(ValueError):
        BackendConfig(name="cuda_magic")


def test_dynamic_masks_change_without_recompilation():
    args, factors = inputs()
    run = jax.jit(
        lambda m: ipa_attention(
            *args, factors=factors, mask=m, backend=BackendConfig(chunk_size=2)
        )
    )
    for mask in (
        jnp.ones((2, 5), bool),
        jnp.zeros((2, 5), bool),
        jnp.array([[1, 0, 0, 1, 0], [0, 1, 1, 0, 1]], bool),
    ):
        assert_tree_close(run(mask), numpy_reference(args, factors, mask=mask))
