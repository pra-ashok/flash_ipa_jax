"""Adapter boundaries: dependency isolation, hardware limits, and API details."""

import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from flash_ipa_jax import BackendConfig, BackendDecision
from flash_ipa_jax.backends import common, flash_attn3, jax_aiter
from flash_ipa_jax.backends.dispatch import resolve_backend

ADAPTERS = {"flash_attn3": flash_attn3, "jax_aiter": jax_aiter}


def test_package_import_does_not_import_optional_gpu_libraries():
    """A fresh CPU process can import the public API without GPU packages."""
    code = """
import importlib.abc
import sys
class NoGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'flash_attn3_jax', 'jax_aiter'}:
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, NoGPUImports())
from flash_ipa_jax import BackendConfig, ipa_attention
from flash_ipa_jax.backends import native, flash_attn3, jax_aiter
assert BackendConfig().name == 'native'
"""
    env = os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


@pytest.mark.parametrize("name", ["flash_attn3", "jax_aiter"])
def test_ineligible_device_does_not_load_optional_kernel(name, monkeypatch):
    def fail():
        raise AssertionError("CPU fallback must not import a GPU binding")

    monkeypatch.setattr(ADAPTERS[name], "load_kernel", fail)
    monkeypatch.setattr(common, "device_description", lambda q: "cpu")
    decision = resolve_backend(
        BackendConfig(name=name, kernel_dtype="float16"), jnp.zeros((1,)), 32, 32
    )
    assert decision.selected == "native"
    assert "GPU" in decision.reason or "gfx950" in decision.reason


@pytest.mark.parametrize(
    "name,device,width,padded",
    [
        ("flash_attn3", "gpu nvidia h100", 128, 128),
        ("jax_aiter", "gpu amd gfx950", 63, 64),
        ("jax_aiter", "gpu amd gfx950", 65, 128),
        ("jax_aiter", "gpu amd gfx950", 129, 256),
    ],
)
def test_adapter_hardware_width_contract(name, device, width, padded):
    reasons, actual_width = ADAPTERS[name].eligibility(width, device)
    assert not reasons
    assert actual_width == padded


@pytest.mark.parametrize("name", ["flash_attn3", "jax_aiter"])
@pytest.mark.parametrize("causal", [False, True])
def test_adapter_padding_and_launch_arguments(name, causal, monkeypatch):
    """Padding preserves every original channel and uses exact launch flags."""
    width = 128 if name == "flash_attn3" else 64
    q = jnp.arange(12, dtype=jnp.float32).reshape(1, 2, 1, 6)
    k, v = q + 1, q[..., :3] + 2
    calls = []

    def kernel(actual_q, actual_k, actual_v, **kwargs):
        for actual, original in zip((actual_q, actual_k, actual_v), (q, k, v)):
            assert actual.shape[-1] == width
            assert actual.dtype == jnp.bfloat16
            np.testing.assert_array_equal(actual[..., : original.shape[-1]], original)
            np.testing.assert_array_equal(actual[..., original.shape[-1] :], 0)
        expected = {"softmax_scale": 1.0}
        if name == "flash_attn3":
            expected["is_causal"] = causal
        else:
            expected |= {"causal": causal, "dropout_p": 0.0, "return_lse": False}
        assert kwargs == expected
        calls.append(kwargs)
        # Both tuple and list results occur in compatible AITER wrappers.
        return [actual_v] if name == "jax_aiter" else actual_v

    monkeypatch.setattr(ADAPTERS[name], "load_kernel", lambda: kernel)
    decision = BackendDecision(name, name, "test launch", width)
    output = ADAPTERS[name].attention(
        q,
        k,
        v,
        config=BackendConfig(name=name, kernel_dtype="bfloat16"),
        decision=decision,
        causal=causal,
    )
    assert output.dtype == jnp.float32
    assert calls
