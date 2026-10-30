"""Framework-free linear layers with AlphaFold/OpenFold initializers.

Use ``layer.init(key)`` to create a parameter pytree, then ``layer.apply(params,
x)`` inside ``jax.jit`` or ``jax.grad``. Weights have shape ``[in_dim,out_dim]``
as in the rest of this JAX package; transpose PyTorch ``[out_dim,in_dim]`` weights
when importing a checkpoint. Initializers return arrays and never mutate them.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

# Standard deviation of a standard normal truncated to [-2, 2]. Dividing by
# this value restores the requested variance after truncation.
_TRUNCATED_NORMAL_STD = 0.87962566103423978


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    """Get the fan for JAX's ``[in_dim,out_dim]`` weight layout."""
    if len(linear_weight_shape) != 2:
        raise ValueError("linear weights must have two dimensions [in_dim, out_dim]")
    fan_in, fan_out = linear_weight_shape
    choices = {"fan_in": fan_in, "fan_out": fan_out, "fan_avg": (fan_in + fan_out) / 2}
    if fan not in choices:
        raise ValueError("fan must be 'fan_in', 'fan_out' or 'fan_avg'")
    return choices[fan]


def trunc_normal_init(key, shape, dtype=jnp.float32, *, scale=1.0, fan="fan_in"):
    """Draw a truncated normal with variance ``scale / fan``."""
    if scale < 0:
        raise ValueError("scale must be nonnegative")
    std = math.sqrt(scale / max(1, _calculate_fan(shape, fan))) / _TRUNCATED_NORMAL_STD
    return jax.random.truncated_normal(key, -2, 2, tuple(shape), dtype) * std


def lecun_normal_init(key, shape, dtype=jnp.float32):
    """LeCun truncated normal: variance ``1 / fan_in``."""
    return trunc_normal_init(key, shape, dtype)


def he_normal_init(key, shape, dtype=jnp.float32):
    """He truncated normal for ReLU layers: variance ``2 / fan_in``."""
    return trunc_normal_init(key, shape, dtype, scale=2.0)


def glorot_uniform_init(key, shape, dtype=jnp.float32):
    """Glorot uniform: variance ``2 / (fan_in + fan_out)``."""
    bound = math.sqrt(3 / max(1, _calculate_fan(shape, "fan_avg")))
    return jax.random.uniform(key, tuple(shape), dtype, minval=-bound, maxval=bound)


def final_init(key, shape, dtype=jnp.float32):
    """Zero weights for residual outputs; the branch initially changes nothing."""
    del key
    return jnp.zeros(shape, dtype=dtype)


gating_init = final_init


def normal_init(key, shape, dtype=jnp.float32):
    """Untruncated normal with standard deviation ``1 / sqrt(fan_in)``."""
    return jax.random.normal(key, tuple(shape), dtype) / math.sqrt(
        max(1, _calculate_fan(shape))
    )


def init_linear(key, inputs, outputs, *, init="default", bias=True, dtype=jnp.float32):
    """Create a projection parameter dictionary for the structure components.

    ``default`` uses LeCun initialization, ``relu`` uses He initialization,
    ``xavier`` uses Glorot uniform, and ``final`` gives zero residual updates.
    ``torch`` reproduces ``torch.nn.Linear``'s uniform weights and biases.
    A split key initializes the weight, preserving the structure model's random
    key convention; :meth:`Linear.init` uses its caller's key directly instead.
    """
    weight_key, bias_key = jax.random.split(key)
    shape = (inputs, outputs)
    if init == "torch":
        weight = jax.random.uniform(
            weight_key, shape, dtype, minval=-1, maxval=1
        ) / math.sqrt(inputs)
    else:
        initialize = {
            "final": final_init,
            "xavier": glorot_uniform_init,
            "relu": he_normal_init,
        }.get(init, lecun_normal_init)
        weight = initialize(weight_key, shape, dtype)
    params = {"weight": weight}
    if bias:
        params["bias"] = (
            jax.random.uniform(bias_key, (outputs,), dtype, minval=-1, maxval=1)
            / math.sqrt(inputs)
            if init == "torch"
            else jnp.zeros((outputs,), dtype)
        )
    return params


def linear(params, x):
    """Project the final axis, keeping any batch, length, or rank axes."""
    output = jnp.matmul(x, params["weight"], precision=jax.lax.Precision.HIGHEST)
    return output + params["bias"] if "bias" in params else output


# Migration helpers retain upstream names but return a replacement array.
# Usage: weight = lecun_normal_init_(weight, key), rather than in-place mutation.
def trunc_normal_init_(weights, key, scale=1.0, fan="fan_in"):
    """Functional replacement for upstream's mutating initializer."""
    return trunc_normal_init(key, weights.shape, weights.dtype, scale=scale, fan=fan)


def lecun_normal_init_(weights, key):
    """Return initialized weights; an explicit JAX random key is required."""
    return lecun_normal_init(key, weights.shape, weights.dtype)


def he_normal_init_(weights, key):
    """Return initialized weights; an explicit JAX random key is required."""
    return he_normal_init(key, weights.shape, weights.dtype)


def glorot_uniform_init_(weights, key):
    """Return initialized weights; an explicit JAX random key is required."""
    return glorot_uniform_init(key, weights.shape, weights.dtype)


def final_init_(weights, key=None):
    """Return zero weights; no random key is needed."""
    return final_init(key, weights.shape, weights.dtype)


gating_init_ = final_init_


def normal_init_(weights, key):
    """Return initialized weights; an explicit JAX random key is required."""
    return normal_init(key, weights.shape, weights.dtype)


class Linear:
    """A pure JAX dense layer with an explicit parameter initialization step.

    ``init`` selects ``default`` (LeCun), ``relu`` (He), ``glorot``, ``normal``,
    ``gating`` (zero weights, bias one), or ``final`` (all zeros). A custom
    ``init_fn(key, shape, dtype)`` may return the weight array instead.
    """

    def __init__(self, in_dim, out_dim, bias=True, init="default", init_fn=None):
        if in_dim < 1 or out_dim < 1:
            raise ValueError("in_dim and out_dim must be positive")
        if init not in ("default", "relu", "glorot", "normal", "gating", "final"):
            raise ValueError(f"unknown linear initializer {init!r}")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.use_bias, self.initializer, self.init_fn = bias, init, init_fn

    def init(self, key, dtype=jnp.float32):
        """Return a ``{'weight': ..., 'bias': ...}`` parameter pytree."""
        initializers = {
            "default": lecun_normal_init,
            "relu": he_normal_init,
            "glorot": glorot_uniform_init,
            "normal": normal_init,
            "gating": gating_init,
            "final": final_init,
        }
        initialize = self.init_fn or initializers[self.initializer]
        params = {"weight": initialize(key, (self.in_dim, self.out_dim), dtype)}
        if params["weight"].shape != (self.in_dim, self.out_dim):
            raise ValueError("custom initializer returned an incorrect weight shape")
        if self.use_bias:
            value = 1 if self.initializer == "gating" and self.init_fn is None else 0
            params["bias"] = jnp.full((self.out_dim,), value, dtype=dtype)
        return params

    def apply(self, params, x):
        """Apply a linear projection over the final input dimension."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected {self.in_dim} input features; got {x.shape[-1]}"
            )
        projection = {"weight": params["weight"]}
        if self.use_bias:
            projection["bias"] = params["bias"]
        return linear(projection, x)

    __call__ = apply


__all__ = [
    "Linear",
    "init_linear",
    "linear",
    "trunc_normal_init",
    "lecun_normal_init",
    "he_normal_init",
    "glorot_uniform_init",
    "normal_init",
    "final_init",
    "gating_init",
    "trunc_normal_init_",
    "lecun_normal_init_",
    "he_normal_init_",
    "glorot_uniform_init_",
    "normal_init_",
    "final_init_",
    "gating_init_",
]
