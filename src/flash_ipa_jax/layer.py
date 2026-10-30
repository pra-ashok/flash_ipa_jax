"""Functional IPA layer: project features, attend in 3D, return local features.

Parameters are ordinary JAX pytrees. A layer returns a single-feature update;
use a residual connection (``s + update``) when building a structure block.
"""

import math
import jax
import jax.numpy as jnp

from .attention import ipa_attention
from .backends import BackendConfig
from .config import IPAConfig
from .linear import linear
from .projections import parameter_shapes, project_pair_factors


def apply_rigid(rotation, translation, points):
    """Column-vector convention: global = R @ local + t."""
    return (
        jnp.einsum("bnij,bnhpj->bnhpi", rotation, points)
        + translation[:, :, None, None, :]
    )


def invert_rigid(rotation, translation, points):
    """Map attended global points back into each query residue's local coordinate system."""
    return jnp.einsum(
        "bnji,bnhpj->bnhpi", rotation, points - translation[:, :, None, None, :]
    )


class InvariantPointAttention:
    """No framework dependency: init(key) and apply(params, ...) are pure JAX.

    Rotations must be proper orthonormal matrices. No automatic unit conversion
    is performed; upstream examples convert Angstrom translations to nm first.
    """

    def __init__(self, config=IPAConfig(), backend=BackendConfig()):
        self.config, self.backend = config, backend

    def init(self, key, *, zero_init_output=True, dtype=jnp.float32):
        """Create trainable weights; zero output initialization stabilizes residuals.

        Set ``zero_init_output=False`` when inspecting geometric effects before
        training. With the default, the first update is zero and gradients first
        train the output projection before reaching the earlier projections.
        """
        params = {}
        for name, (inputs, outputs) in parameter_shapes(self.config).items():
            key, subkey = jax.random.split(key)
            weight = jax.random.truncated_normal(
                subkey, -2, 2, (inputs, outputs), dtype
            )
            weight = weight / (math.sqrt(inputs) * 0.87962566103423978)
            if name == "linear_out" and zero_init_output:
                weight = jnp.zeros_like(weight)
            params[name] = {"weight": weight, "bias": jnp.zeros((outputs,), dtype)}
        params["head_weights"] = jnp.full(
            (self.config.no_heads,), 0.541324854612918, dtype
        )
        return params

    def apply(
        self,
        params,
        s,
        rotations,
        translations,
        *,
        z=None,
        z_factor_1=None,
        z_factor_2=None,
        mask=None,
        pair_mask=None,
        causal=False,
    ):
        """Return an update of shape [batch, residues, c_s].

        ``rotations`` and ``translations`` describe each residue's local coordinate system.
        Supply either dense ``z`` or two factors according to ``pair_mode``.
        Masks are boolean; invalid query rows always return zero.
        """
        c = self.config
        if s.ndim != 3 or s.shape[-1] != c.c_s:
            raise ValueError("s must be [B,N,c_s]")
        b, n, _ = s.shape
        if rotations.shape != (b, n, 3, 3) or translations.shape != (b, n, 3):
            raise ValueError("rotations/translations must be [B,N,3,3] and [B,N,3]")
        # 1. Project ordinary features into per-head query/key/value scalars.
        q = linear(params["linear_q"], s).reshape(b, n, c.no_heads, c.c_hidden)
        kv = linear(params["linear_kv"], s).reshape(b, n, c.no_heads, 2 * c.c_hidden)
        k, v = jnp.split(kv, 2, axis=-1)

        # 2. Predict local points, then place them in the shared global coordinate system.
        def project_points(name, count):
            # Upstream emits all x coordinates, then all y, then all z.
            x = linear(params[name], s).reshape(b, n, 3, c.no_heads, count)
            return apply_rigid(
                rotations, translations, jnp.transpose(x, (0, 1, 3, 4, 2))
            )

        qp = project_points("linear_q_points", c.no_qk_points)
        kvp = project_points("linear_kv_points", c.no_qk_points + c.no_v_points)
        kp, vp = jnp.split(kvp, [c.no_qk_points], axis=-2)
        # 3. Prepare pair context independently of the execution backend.
        factors = pair_bias = pair_values = None
        if c.pair_mode == "factorized":
            expected = (b, n, c.z_factor_rank, c.c_z)
            if z is not None or z_factor_1 is None or z_factor_2 is None:
                raise ValueError(
                    "factorized mode requires both z factors and no dense z"
                )
            if z_factor_1.shape != expected or z_factor_2.shape != expected:
                raise ValueError(f"z factors must have shape {expected}")
            factors = project_pair_factors(params, z_factor_1, z_factor_2)
        elif c.pair_mode == "dense":
            if (
                z is None
                or z.shape != (b, n, n, c.c_z)
                or z_factor_1 is not None
                or z_factor_2 is not None
            ):
                raise ValueError("dense mode requires z [B,N,N,c_z] and no factors")
            pair_bias = linear(params["linear_b"], z) / math.sqrt(3)
            pair_values = linear(params["down_z"], z)
        elif z is not None or z_factor_1 is not None or z_factor_2 is not None:
            raise ValueError("pair_mode='none' does not accept pair features")
        # 4. Attend using scalar similarity, point distance, and pair context.
        result = ipa_attention(
            q,
            k,
            v,
            qp,
            kp,
            vp,
            params["head_weights"],
            factors=factors,
            pair_bias=pair_bias,
            pair_values=pair_values,
            mask=mask,
            pair_mask=pair_mask,
            causal=causal,
            backend=self.backend,
        )
        # 5. Express attended points locally to produce invariant output features.
        local = invert_rigid(rotations, translations, result.points)
        local = jnp.where(result.valid[:, :, None, None, None], local, 0)
        norms = jnp.sqrt(jnp.sum(local**2, -1) + c.eps)
        features = [result.scalar.reshape(b, n, -1)]
        features.extend(local[..., axis].reshape(b, n, -1) for axis in range(3))
        features.append(norms.reshape(b, n, -1))
        if result.pair is not None:
            features.append(result.pair.reshape(b, n, -1))
        update = linear(params["linear_out"], jnp.concatenate(features, axis=-1))
        # Explicitly define padded and fully masked query outputs, including bias.
        return jnp.where(result.valid[:, :, None], update, 0)
