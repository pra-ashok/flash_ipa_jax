"""Shared semantic checks and visible fallback policy for the three adapters."""

import warnings

import jax.numpy as jnp

from . import common, flash_attn3, jax_aiter
from .config import BackendDecision, BackendFallbackWarning, BackendUnavailableError


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
    # Ordinary fused Q/K/V APIs cannot represent these IPA semantics.
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
    adapter = flash_attn3 if config.name == "flash_attn3" else jax_aiter
    hardware_reasons, padded = adapter.eligibility(
        max(qk_dim, value_dim), common.device_description(q)
    )
    reasons.extend(hardware_reasons)
    if not reasons:
        try:
            adapter.load_kernel()
        except (ImportError, OSError, AttributeError, ValueError) as exc:
            reasons.append(f"optional kernel unavailable: {type(exc).__name__}: {exc}")
    if reasons:
        return BackendDecision(config.name, "native", "; ".join(reasons))
    return BackendDecision(
        config.name, config.name, "eligible expanded IPA attention", padded
    )


def enforce_decision(config, decision):
    """Warn about a native fallback, or raise when strict execution was requested."""
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
    """Call the selected GPU adapter; launch failures remain visible to callers."""
    adapter = flash_attn3 if decision.selected == "flash_attn3" else jax_aiter
    return adapter.attention(q, k, v, config=config, decision=decision, causal=causal)
