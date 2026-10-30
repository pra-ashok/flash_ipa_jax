"""The documented optimizer loop must actually learn a nonzero target."""

import importlib.util
from pathlib import Path
import sys

import jax
import pytest

from flash_ipa_jax import BackendFallbackWarning, BackendUnavailableError


def load_example():
    """Load the executable without making examples an installed runtime package."""
    directory = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "ipa_training_example", directory / "train.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("backend", ["native", "flash_attn3", "jax_aiter"])
def test_training_loop_reduces_nonzero_teacher_loss(backend):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU smoke test; strict real GPU tests are separate")
    example = load_example()
    kwargs = dict(backend=backend, length=4, steps=5, log_every=5)
    if backend == "native":
        result = example.run_training(**kwargs)
    else:
        with pytest.warns(BackendFallbackWarning):
            result = example.run_training(**kwargs)
    assert result["selected_backend"] == "native"
    assert result["initial_loss"] > 0.01
    assert result["final_loss"] < 0.8 * result["initial_loss"]


@pytest.mark.parametrize("backend", ["flash_attn3", "jax_aiter"])
def test_strict_training_does_not_disguise_cpu_as_gpu(backend):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU guard test")
    with pytest.raises(BackendUnavailableError):
        load_example().run_training(backend=backend, strict=True, length=4, steps=1)
