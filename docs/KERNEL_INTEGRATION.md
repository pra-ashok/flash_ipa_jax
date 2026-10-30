# Attention equations and kernel integration

This page is for contributors extending execution paths. Start with
[Concepts](CONCEPTS.md) for the plain-language explanation.

## Preserve the complete IPA score

Let `D` be scalar width per head, `P` the number of query/key points, and
`w[h] = softplus(raw[h]) / sqrt(3 * P * 9/2)`. Then the implemented logit is:

```text
L[b,h,i,j] = dot(q[b,i,h], k[b,j,h]) / sqrt(3D)
             - 0.5 * w[h] * sum_p ||Q[b,i,h,p] - K[b,j,h,p]||²
             + pair_bias[b,i,j,h]
```

Dense IPA uses `linear_b(z)/sqrt(3)` as pair bias, and aggregates
`down_z(z[i,j])` with attention weights. Factorized FlashIPA applies the same
biased `linear_b` separately to each factor, then takes the rank dot product.
The inspected upstream flash branch has **no additional sqrt(1/3) scale**.
Pair values use separately projected factors `u1=down_z(z1)`, `u2=down_z(z2)`:

```text
pair_bias[i,j,h] = sum_r b1[i,h,r] * b2[j,h,r]
pair_out[i,h,c] = sum_r u1[i,r,c] * sum_j A[i,j,h] * u2[j,r,c]
```

These biased factor projections are not equivalent to applying a single
projection to a reconstructed dense pair tensor. Backend switching never
silently reconstructs, approximates or drops pair contributions.

## Packing for ordinary fused attention

```text
Q' = [q,                 flatten(Q),   ||Q_p||², 1_p,           b1]
K' = [k/sqrt(3D),        w*flatten(K), -w/2,     -w*||K_p||²/2, b2]
V' = [v, flatten(V_points), flatten(u2)]
```

`Q' @ K'^T` equals the complete factorized IPA logit in exact arithmetic.
Both norm terms are necessary. The fused kernel's **softmax scale is 1.0**,
including after padding. Its ordinary default inverse-square-root scale would
change the model. After aggregation, query-side value factors are contracted
and global point values return to each query residue's coordinate system. Final features use
upstream's scalar, all x, all y, all z, point-norm, pair ordering.

A common differentiable origin is subtracted before geometric expansion and
restored after point aggregation. Distances stay the same mathematically;
centering reduces numerical cancellation under global translations. FP16/BF16
still round, and extreme feature/coordinate ranges can overflow. Direct native
float32, or float64 with `jax_enable_x64=True`, remains the accuracy reference.

Expanded widths are `D+5P+rank` and `Dv+3Pv+rank*C_pair`; omit factor terms
without pairs. The kernel's width contract must cover both, not just scalar D.
The public `pack_qkv` and native reference tests exercise this algebra.

## What native flash now supports

`kernels/tiled.py` tiles both query and key axes, computes direct geometric
scores, applies exact boolean masks, and maintains online-softmax accumulators
for scalar values, global point values and pair values. It supports **arbitrary
dense pair bias and query-dependent dense pair values**, in addition to factors.
There is no need to drop those terms to run the native tiled algorithm.

Backward uses JAX autodiff through scans and rematerialization. It is a
portable differentiable FlashAttention algorithm, not a target-specific fused
SRAM kernel or a custom streaming backward implementation. Autodiff can retain
scan state; actual training-memory use must be profiled. Dense pair inputs
remain quadratic allocations even when attention probabilities are tiled.

## What remains for specialized GPU kernels

The ordinary FA3/AITER adapters cannot represent a general query-dependent
dense pair value in V. They also do not implement arbitrary batch/head pair
bias or explicit edge masks. These cases intentionally select native execution.
A dedicated IPA-aware fused kernel would need to:

1. Load scalar, point and pair tiles, preserving every scale and projection.
2. Apply token, pair and causal masks with stable online row statistics.
3. Accumulate scalars, points and query-dependent dense pair values.
4. Return gradients for scalar Q/K/V, all points, raw head weights, pair bias
   and pair values, with the full chain through rigid transformations and projections.
5. Produce finite zero outputs and VJPs for entirely masked rows.

XLA FFI or target-specific Pallas kernels are possible extension routes. Neither
an arbitrary-mask fused GPU path nor a dedicated dense-pair fused GPU kernel is
implemented here. An API that accepts a bias is not proof that its gradients
or shape contract match IPA requirements.

## Masks, build capabilities and validation

Variable-length APIs can help with ragged sequences, but copying PyTorch's
runtime `nonzero()` unpadding into JIT introduces dynamic shapes. An extension
needs static buffers or a supported runtime-length contract, arbitrary holes,
empty samples, output repadding and gradients. Lengths alone cannot encode
arbitrary pair masks. Until verified, explicit masks use native execution.

Larger FA3 builds require capability checks and strict forward/backward tests;
changing a maximum width constant alone is insufficient. Preserve scale 1.0.
Third-party compilation/launch errors must remain visible.

Before enabling a new fused path, compare outputs and every input VJP against
native float32/float64. Include trained weights, point ranges, non-multiple
lengths, empty rows and common SE(3) transforms with nonzero output projections.
Then measure synchronized end-to-end forward/backward and real peak device
memory. Emulated API tests prove adapter conventions, not hardware execution.
See [VALIDATION.md](../VALIDATION.md) for the current evidence.
