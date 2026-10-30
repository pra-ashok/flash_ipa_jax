"""Learned scalar, point, and pair projections with upstream parameter names."""

import jax.numpy as jnp
from .kernels.types import PairFactors
from .linear import linear


def parameter_shapes(config):
    """Return projection dimensions under the upstream checkpoint names."""
    c = config
    result = {
        "linear_q": (c.c_s, c.no_heads * c.c_hidden),
        "linear_kv": (c.c_s, 2 * c.no_heads * c.c_hidden),
        "linear_q_points": (c.c_s, 3 * c.no_heads * c.no_qk_points),
        "linear_kv_points": (c.c_s, 3 * c.no_heads * (c.no_qk_points + c.no_v_points)),
    }
    if c.c_z:
        result.update(linear_b=(c.c_z, c.no_heads), down_z=(c.c_z, c.c_z // 4))
    result["linear_out"] = (
        c.no_heads * (c.c_hidden + 4 * c.no_v_points + c.c_z // 4),
        c.c_s,
    )
    return result


def project_pair_factors(params, z_factor_1, z_factor_2):
    """Match upstream flash_ipa_fwd, including biased projections and no sqrt(1/3)."""
    return PairFactors(
        jnp.swapaxes(linear(params["linear_b"], z_factor_1), -1, -2),
        jnp.swapaxes(linear(params["linear_b"], z_factor_2), -1, -2),
        linear(params["down_z"], z_factor_1),
        linear(params["down_z"], z_factor_2),
    )
