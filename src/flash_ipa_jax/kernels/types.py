"""Array containers shared by every attention implementation."""

from typing import NamedTuple
import jax


class PairFactors(NamedTuple):
    """Projected factors, not an implicit factorization of arbitrary dense pairs.

    bias_q/bias_k: [B,N,H,R]; value_q/value_k: [B,N,R,C].
    Bias is sum_r bias_q[i,h,r]*bias_k[j,h,r], without extra scaling.
    Pair values are sum_r value_q[i,r,c]*value_k[j,r,c].
    """

    bias_q: jax.Array
    bias_k: jax.Array
    value_q: jax.Array
    value_k: jax.Array


class IPAOutput(NamedTuple):
    scalar: jax.Array
    points: jax.Array  # global coordinates; equivariant for valid rows
    pair: jax.Array | None
    valid: jax.Array  # [B,N], False for entirely masked rows
