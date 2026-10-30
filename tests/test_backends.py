import importlib
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flash_ipa_jax import (
    BackendConfig,
    BackendUnavailableError,
    BackendFallbackWarning,
    ipa_attention,
    backend_report,
)
import flash_ipa_jax.backends as backends
from helpers import inputs


@pytest.mark.parametrize("name", ["flash_attn3", "jax_aiter"])
def test_cpu_fallback_is_correct_and_strict_raises(name, monkeypatch):
    monkeypatch.setattr(backends, "_device_description", lambda q: "cpu")
    args, factors = inputs(dtype=jnp.float32)
    with pytest.warns(BackendFallbackWarning, match="native"):
        out = ipa_attention(*args, factors=factors, backend=BackendConfig(name=name))
    expected = ipa_attention(*args, factors=factors)
    for x, y in zip(jax.tree.leaves(out), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(BackendUnavailableError):
        ipa_attention(*args, backend=BackendConfig(name=name, fallback="error"))


@pytest.mark.parametrize(
    "name,device",
    [("flash_attn3", "gpu nvidia h100"), ("jax_aiter", "gpu amd mi355x gfx950")],
)
def test_eligibility_reasons(name, device, monkeypatch):
    monkeypatch.setattr(backends, "_device_description", lambda q: device)
    monkeypatch.setattr(backends, "_load_kernel", lambda name: object())
    args, factors = inputs(dtype=jnp.float32)
    config = BackendConfig(name=name, kernel_dtype="float16")

    def report(**kw):
        return backend_report(args[0], args[2], args[3], args[5], backend=config, **kw)

    assert report(factors=factors).selected == name
    assert "explicit masks" in report(mask=jnp.ones((2, 5), bool)).reason
    assert "explicit masks" in report(pair_mask=jnp.ones((2, 5, 5), bool)).reason
    assert "dense pair bias" in report(pair_bias=jnp.zeros((2, 5, 5, 2))).reason
    assert "dense pair values" in report(pair_values=jnp.zeros((2, 5, 5, 3))).reason
    decision = backends.resolve_backend(config, args[0], 300, 30)
    assert decision.selected == "native" and "exceeds" in decision.reason

    def missing(name):
        raise ImportError("test missing kernel")

    monkeypatch.setattr(backends, "_load_kernel", missing)
    assert "test missing kernel" in report().reason


@pytest.mark.parametrize("name", ["flash_attn3", "jax_aiter"])
def test_adapter_contract_with_emulated_kernel(name, monkeypatch):
    """Contract test only; this does not exercise a real GPU kernel."""
    args, factors = inputs(dtype=jnp.float32)
    device = "gpu h100" if name == "flash_attn3" else "gpu gfx950"
    monkeypatch.setattr(backends, "_device_description", lambda q: device)
    calls = []

    def kernel(q, k, v, **kw):
        calls.append(kw)
        assert kw["softmax_scale"] == 1.0
        assert q.dtype == jnp.float16
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        logits = jnp.einsum(
            "bihd,bjhd->bhij", q.astype(jnp.float32), k.astype(jnp.float32)
        )
        causal = kw.get("causal", kw.get("is_causal"))
        if causal:
            logits = jnp.where(
                jnp.arange(q.shape[1])[None, :] <= jnp.arange(q.shape[1])[:, None],
                logits,
                -jnp.inf,
            )
        out = jnp.einsum(
            "bhij,bjhd->bihd", jax.nn.softmax(logits, -1), v.astype(jnp.float32)
        ).astype(q.dtype)
        return (out,) if name == "jax_aiter" else out

    monkeypatch.setattr(backends, "_load_kernel", lambda name: kernel)
    config = BackendConfig(name=name, kernel_dtype="float16", fallback="error")
    out = jax.jit(
        lambda *xs: ipa_attention(*xs, factors=factors, backend=config, causal=True)
    )(*args)
    expected = ipa_attention(*args, factors=factors, causal=True)
    for x, y in zip(jax.tree.leaves(out), jax.tree.leaves(expected)):
        np.testing.assert_allclose(x, y, rtol=0.01, atol=0.001)
    assert calls


def test_native_never_imports_optional_packages(monkeypatch):
    def fail(name):
        raise AssertionError("unexpected optional import")

    monkeypatch.setattr(backends, "_load_kernel", fail)
    args, _ = inputs()
    ipa_attention(*args)


def test_api_drift_is_rejected(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(flash_mha=lambda q, k, v: q),
    )
    with pytest.raises(ImportError, match="incompatible"):
        backends._load_kernel("flash_attn3")


def test_kernel_runtime_failure_not_hidden(monkeypatch):
    args, _ = inputs(dtype=jnp.float32)
    monkeypatch.setattr(backends, "_device_description", lambda q: "gpu h100")

    def fail(*a, **kw):
        raise RuntimeError("FFI failure")

    monkeypatch.setattr(backends, "_load_kernel", lambda name: fail)
    with pytest.raises(RuntimeError, match="FFI failure"):
        ipa_attention(
            *args, backend=BackendConfig(name="flash_attn3", kernel_dtype="float16")
        )


def test_mixed_float64_inputs_require_native(monkeypatch):
    monkeypatch.setattr(backends, "_device_description", lambda q: "gpu h100")
    monkeypatch.setattr(backends, "_load_kernel", lambda name: object())
    args, _ = inputs(dtype=jnp.float32)
    args = (*args[:-1], args[-1].astype(jnp.float64))
    with pytest.raises(BackendUnavailableError, match="float64"):
        ipa_attention(
            *args,
            backend=BackendConfig(
                name="flash_attn3", kernel_dtype="float16", fallback="error"
            )
        )


def test_report_promotes_every_supplied_tensor(monkeypatch):
    monkeypatch.setattr(backends, "_device_description", lambda q: "gpu h100")
    monkeypatch.setattr(backends, "_load_kernel", lambda name: object())
    args, factors = inputs(dtype=jnp.float32)
    config = BackendConfig(name="flash_attn3", kernel_dtype="float16")
    for mixed in (
        dict(v=args[2].astype(jnp.float64)),
        dict(head_weights=args[-1].astype(jnp.float64)),
        dict(factors=jax.tree.map(lambda x: x.astype(jnp.float64), factors)),
    ):
        options = dict(
            q=args[0], v=args[2], q_points=args[3], v_points=args[5], backend=config
        )
        report = backend_report(**(options | mixed))
        assert report.selected == "native"
        assert "float64" in report.reason
