"""Portable JAX/XLA adapter for CPU checks and accelerator execution.

The default online-softmax implementation tiles both sequence axes. Reference
modes are useful for debugging. All modes support masks, dense pairs, and
factorized pairs; no external attention package is needed.
"""

from ..kernels.reference import reference_attention
from ..kernels.tiled import tiled_attention


def attention(
    q,
    k,
    v,
    qp,
    kp,
    vp,
    raw,
    factors,
    pair_bias,
    pair_values,
    mask,
    pair_mask,
    causal,
    config,
):
    """Run the selected native IPA algorithm on already validated inputs."""
    kernel = tiled_attention if config.native_mode == "flash" else reference_attention
    return kernel(
        q,
        k,
        v,
        qp,
        kp,
        vp,
        raw,
        factors,
        pair_bias,
        pair_values,
        mask,
        pair_mask,
        causal,
        config,
    )
