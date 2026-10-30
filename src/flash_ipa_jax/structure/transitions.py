"""Residual node, edge, and local coordinate system updates from the upstream IPA model."""

import math

import jax
import jax.numpy as jnp

from ..linear import init_linear, linear
from ..normalization import init_norm, layer_norm


class StructureModuleTransition:
    """Three projections, a residual connection, and a final LayerNorm."""

    def __init__(self, c):
        if c < 1:
            raise ValueError("channel count must be positive")
        self.c = c

    def init(self, key, *, dtype=jnp.float32):
        keys = jax.random.split(key, 3)
        return {
            **{
                f"linear_{i+1}": init_linear(
                    keys[i],
                    self.c,
                    self.c,
                    init="final" if i == 2 else "relu",
                    dtype=dtype,
                )
                for i in range(3)
            },
            "ln": init_norm(self.c, dtype),
        }

    def apply(self, params, s):
        hidden = jax.nn.relu(linear(params["linear_1"], s))
        hidden = jax.nn.relu(linear(params["linear_2"], hidden))
        return layer_norm(params["ln"], s + linear(params["linear_3"], hidden))


class BackboneUpdate:
    """Predict a local quaternion-vector/translation update (or translation only)."""

    def __init__(self, c_s, use_rot_updates=True):
        if c_s < 1:
            raise ValueError("channel count must be positive")
        self.c_s, self.use_rot_updates = c_s, use_rot_updates

    def init(self, key, *, dtype=jnp.float32):
        return {
            "linear": init_linear(
                key,
                self.c_s,
                6 if self.use_rot_updates else 3,
                init="final",
                dtype=dtype,
            )
        }

    def apply(self, params, s):
        return linear(params["linear"], s)


class EdgeTransition:
    """Update dense edges (``2d``) or two linear-size edge factors (``1d``).

    In the factorized path, node features become separate row/column biases.
    The same MLP then updates both factors without creating an N by N tensor.
    """

    def __init__(
        self,
        *,
        mode,
        node_embed_size,
        edge_embed_in,
        edge_embed_out,
        z_factor_rank=0,
        num_layers=2,
        node_dilation=2,
    ):
        if mode not in ("1d", "2d"):
            raise ValueError("mode must be '1d' or '2d'")
        if min(node_embed_size, edge_embed_in, edge_embed_out, node_dilation) < 1:
            raise ValueError("edge transition dimensions must be positive")
        if num_layers < 0 or (mode == "1d" and z_factor_rank < 1):
            raise ValueError(
                "num_layers must be nonnegative; factor mode needs a positive rank"
            )
        self.mode, self.z_factor_rank = mode, z_factor_rank
        self.node_embed_size = node_embed_size
        self.edge_embed_in, self.edge_embed_out = edge_embed_in, edge_embed_out
        self.bias_size = node_embed_size // node_dilation
        if self.bias_size < 1:
            raise ValueError("node_dilation is larger than node_embed_size")
        self.hidden_size, self.num_layers = (
            2 * self.bias_size + edge_embed_in,
            num_layers,
        )

    def init(self, key, *, dtype=jnp.float32):
        keys = iter(jax.random.split(key, self.num_layers + 3))
        params = {
            "initial_embed": init_linear(
                next(keys),
                self.node_embed_size,
                self.bias_size,
                init="relu",
                dtype=dtype,
            )
        }
        if self.mode == "1d":
            params["edge_bias_linear"] = init_linear(
                next(keys),
                self.bias_size,
                4 * self.z_factor_rank * self.bias_size,
                init="final",
                dtype=dtype,
            )
        params["trunk"] = tuple(
            init_linear(
                next(keys), self.hidden_size, self.hidden_size, init="relu", dtype=dtype
            )
            for _ in range(self.num_layers)
        )
        params["final_layer"] = init_linear(
            next(keys), self.hidden_size, self.edge_embed_out, init="final", dtype=dtype
        )
        params["layer_norm"] = init_norm(self.edge_embed_out, dtype)
        return params

    def apply(
        self, params, node_embed, edge_embed=None, z_factor_1=None, z_factor_2=None
    ):
        nodes = linear(params["initial_embed"], node_embed)
        batch, length, _ = nodes.shape
        if self.mode == "2d":
            if edge_embed is None or z_factor_1 is not None or z_factor_2 is not None:
                raise ValueError("2d transition requires dense edges and no factors")
            expected = (batch, length, length, self.edge_embed_in)
            if edge_embed.shape != expected:
                raise ValueError(f"edge_embed must have shape {expected}")
            row = jnp.broadcast_to(
                nodes[:, :, None, :], (batch, length, length, self.bias_size)
            )
            col = jnp.broadcast_to(nodes[:, None, :, :], row.shape)
            features = jnp.concatenate((edge_embed, row, col), axis=-1)
        else:
            expected = (batch, length, self.z_factor_rank, self.edge_embed_in)
            if edge_embed is not None or z_factor_1 is None or z_factor_2 is None:
                raise ValueError(
                    "1d transition requires both factors and no dense edges"
                )
            if z_factor_1.shape != expected or z_factor_2.shape != expected:
                raise ValueError(f"factors must have shape {expected}")
            bias = linear(params["edge_bias_linear"], nodes)
            bias = jnp.transpose(
                bias.reshape(batch, length, 2, self.z_factor_rank, 2 * self.bias_size),
                (0, 2, 1, 3, 4),
            )
            features = jnp.concatenate(
                (jnp.stack((z_factor_1, z_factor_2), axis=1), bias), -1
            )
            features = features / math.sqrt(2)
        hidden = features
        for projection in params["trunk"]:
            hidden = jax.nn.relu(linear(projection, hidden))
        output = linear(params["final_layer"], hidden + features)
        output = layer_norm(params["layer_norm"], output)
        return output if self.mode == "2d" else (output[:, 0], output[:, 1])
