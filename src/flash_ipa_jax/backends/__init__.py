"""Three attention adapters with a shared, observable selection policy.

``native`` runs portable JAX/XLA; ``flash_attn3`` adapts the NVIDIA FA3
package; ``jax_aiter`` adapts the AMD AITER package. Importing this package
never imports either optional GPU dependency.
"""

from .config import (
    BackendConfig,
    BackendDecision,
    BackendFallbackWarning,
    BackendUnavailableError,
)
from .dispatch import enforce_decision, fused_attention, resolve_backend

__all__ = [
    "BackendConfig",
    "BackendDecision",
    "BackendFallbackWarning",
    "BackendUnavailableError",
    "enforce_decision",
    "fused_attention",
    "resolve_backend",
]
