import numpy as np
import jax.numpy as jnp
from flash_ipa_jax import PairFactors


def inputs(n=5, dtype=jnp.float64):
    rng = np.random.default_rng(149)

    def rand(shape):
        return jnp.asarray(rng.normal(size=shape) * 0.3, dtype)

    q, k, v = [rand((2, n, 2, d)) for d in (3, 3, 4)]
    qp, kp, vp = [rand((2, n, 2, p, 3)) for p in (2, 2, 3)]
    raw = rand((2,))
    factors = PairFactors(
        rand((2, n, 2, 2)), rand((2, n, 2, 2)), rand((2, n, 2, 3)), rand((2, n, 2, 3))
    )
    return (q, k, v, qp, kp, vp, raw), factors


def numpy_reference(
    args,
    factors=None,
    pair_bias=None,
    pair_values=None,
    mask=None,
    pair_mask=None,
    causal=False,
):
    """Independent explicit-loop oracle, following upstream IPA equations."""
    q, k, v, qp, kp, vp, raw = map(np.asarray, args)
    b, n, h, d = q.shape
    weights = np.logaddexp(0, raw) / np.sqrt(3 * qp.shape[-2] * 9 / 2)
    scalar, points = np.zeros_like(v), np.zeros_like(vp)
    pair = None
    if factors is not None:
        bq, bk, vq, vk = map(np.asarray, factors)
        pair = np.zeros((b, n, h, vq.shape[-1]))
    elif pair_values is not None:
        pair = np.zeros((b, n, h, pair_values.shape[-1]))
    valid = np.zeros((b, n), bool)
    for bi in range(b):
        for i in range(n):
            js = [
                j
                for j in range(n)
                if (mask is None or (mask[bi, i] and mask[bi, j]))
                and (pair_mask is None or pair_mask[bi, i, j])
                and (not causal or j <= i)
            ]
            if not js:
                continue
            valid[bi, i] = True
            for hi in range(h):
                logits = []
                for j in js:
                    value = q[bi, i, hi] @ k[bi, j, hi] / np.sqrt(3 * d)
                    value -= (
                        0.5 * weights[hi] * np.sum((qp[bi, i, hi] - kp[bi, j, hi]) ** 2)
                    )
                    if factors is not None:
                        value += bq[bi, i, hi] @ bk[bi, j, hi]
                    if pair_bias is not None:
                        value += pair_bias[bi, i, j, hi]
                    logits.append(value)
                a = np.exp(np.array(logits) - max(logits))
                a /= a.sum()
                for aj, j in zip(a, js):
                    scalar[bi, i, hi] += aj * v[bi, j, hi]
                    points[bi, i, hi] += aj * vp[bi, j, hi]
                    if factors is not None:
                        pair[bi, i, hi] += aj * np.sum(vq[bi, i] * vk[bi, j], axis=0)
                    elif pair_values is not None:
                        pair[bi, i, hi] += aj * pair_values[bi, i, j]
    return scalar, points, pair, valid
