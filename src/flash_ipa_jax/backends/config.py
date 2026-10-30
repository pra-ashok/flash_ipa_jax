"""Backend settings, errors, and inspectable selection results."""

from dataclasses import dataclass


class BackendUnavailableError(RuntimeError):
    """A requested fused kernel cannot execute under the chosen strict policy."""


class BackendFallbackWarning(UserWarning):
    """A fused request selected the portable JAX implementation instead."""


@dataclass(frozen=True)
class BackendConfig:
    """Select execution separately from the model's pair representation.

    ``flash`` tiles both sequence axes with online softmax. ``chunked`` tiles
    only queries, and ``dense`` is a small-input reference. Optional GPU kernels
    require explicit reduced-precision opt-in and compatible hardware.
    """

    name: str = "native"
    fallback: str = "native"
    kernel_dtype: str = "float32"
    native_mode: str = "flash"
    chunk_size: int = 64
    warn_on_fallback: bool = True
    key_chunk_size: int = 64

    def __post_init__(self):
        if self.name not in ("native", "flash_attn3", "jax_aiter"):
            raise ValueError("name must be native, flash_attn3, or jax_aiter")
        if self.fallback not in ("native", "error"):
            raise ValueError("fallback must be native or error")
        if self.kernel_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("kernel_dtype must be float32, float16, or bfloat16")
        if self.native_mode not in ("dense", "chunked", "flash"):
            raise ValueError("native_mode must be dense, chunked, or flash")
        if min(self.chunk_size, self.key_chunk_size) <= 0:
            raise ValueError("chunk_size and key_chunk_size must be positive")


@dataclass(frozen=True)
class BackendDecision:
    """Requested adapter, selected adapter, and an explanation of the choice."""

    requested: str
    selected: str
    reason: str
    padded_dim: int | None = None
