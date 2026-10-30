"""Conservative, observable dispatch. Native execution never imports GPU packages."""

from dataclasses import dataclass
import importlib
import inspect
import warnings

import jax
import jax.numpy as jnp


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
    requested: str
    selected: str
    reason: str
    padded_dim: int | None = None


def _device_description(q):
    # Traced calls use the configured default device. See README for jit placement.
    if isinstance(q, jax.core.Tracer):
        default = jax.config.jax_default_device
        devices = [default] if default is not None else jax.local_devices()
    else:
        devices = list(q.devices())
    if len(devices) != 1:
        return "unsupported multi-device placement"
    d = devices[0]
    return f"{d.platform} {d.device_kind} {getattr(d.client, 'platform_version', '')}".lower()


def _load_kernel(name):
    if name == "flash_attn3":
        fn = importlib.import_module("flash_attn3_jax").flash_mha
        required = {"softmax_scale", "is_causal"}
    else:
        fn = importlib.import_module("jax_aiter.mha").flash_attn_func
        required = {"softmax_scale", "causal", "dropout_p", "return_lse"}
    if not required.issubset(inspect.signature(fn).parameters):
        raise ImportError(
            f"{name} public API is incompatible with the inspected adapter"
        )
    return fn


def resolve_backend(
    config,
    q,
    qk_dim,
    value_dim,
    *,
    has_mask=False,
    has_dense_bias=False,
    has_dense_values=False,
    input_dtype=None,
):
    """Read-only eligibility report; unsupported features are never silently dropped."""
    if config.name == "native":
        return BackendDecision(
            "native", "native", f"portable JAX/XLA ({config.native_mode})"
        )
    reasons = []
    if has_mask:
        reasons.append("explicit masks require native execution (even all-true masks)")
    if has_dense_bias:
        reasons.append("arbitrary dense pair bias is not supported by this adapter")
    if has_dense_values:
        reasons.append(
            "query-dependent dense pair values cannot be ordinary attention V"
        )
    if config.kernel_dtype == "float32":
        reasons.append("fused kernels require explicit float16 or bfloat16 opt-in")
    if (q.dtype if input_dtype is None else input_dtype) == jnp.float64:
        reasons.append("float64 inputs require native execution")
    width = max(qk_dim, value_dim)
    device = _device_description(q)
    if config.name == "flash_attn3":
        padded = 128
        if "gpu" not in device or not any(
            x in device for x in ("h100", "h200", "gh200", "sm90")
        ):
            reasons.append("FA3 adapter requires a single NVIDIA Hopper GPU")
        if width > 128:
            reasons.append(
                f"expanded IPA dimension {width} exceeds FA3 wheel dimension 128"
            )
    else:
        padded = next((d for d in (64, 128, 256) if d >= width), None)
        if "gpu" not in device or not any(x in device for x in ("gfx950", "mi355")):
            reasons.append("AITER adapter targets the inspected gfx950/MI355X stack")
        if padded is None:
            reasons.append(f"expanded IPA dimension {width} exceeds 256")
    if not reasons:
        try:
            _load_kernel(config.name)
        except (ImportError, OSError, AttributeError, ValueError) as exc:
            reasons.append(f"optional kernel unavailable: {type(exc).__name__}: {exc}")
    if reasons:
        return BackendDecision(config.name, "native", "; ".join(reasons))
    return BackendDecision(
        config.name, config.name, "eligible expanded IPA attention", padded
    )


def enforce_decision(config, decision):
    if decision.selected != decision.requested:
        if config.fallback == "error":
            raise BackendUnavailableError(decision.reason)
        if config.warn_on_fallback:
            warnings.warn(
                f"{decision.requested} -> native: {decision.reason}",
                BackendFallbackWarning,
                stacklevel=3,
            )


def fused_attention(q, k, v, *, config, decision, causal):
    """No runtime exception swallowing: failed FFI launches must remain visible."""
    dim = decision.padded_dim

    def pad(x):
        return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, dim - x.shape[-1]))).astype(
            config.kernel_dtype
        )

    fn = _load_kernel(decision.selected)
    q, k, v = map(pad, (q, k, v))
    if decision.selected == "flash_attn3":
        result = fn(q, k, v, softmax_scale=1.0, is_causal=causal)
    else:
        result = fn(
            q, k, v, dropout_p=0.0, softmax_scale=1.0, causal=causal, return_lse=False
        )
        # Inspected AITER source returns (out,), despite its docstring.
        if isinstance(result, (tuple, list)):
            result = result[0]
    return result.astype(jnp.float32)
