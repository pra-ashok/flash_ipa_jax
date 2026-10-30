"""Shared argument parsing and tiny, valid geometric inputs for the examples."""

import argparse

import jax
import jax.numpy as jnp

from flash_ipa_jax import (
    BackendConfig,
    IPAConfig,
    InvariantPointAttention,
    backend_report,
    project_pair_factors,
)


def argument_parser(description):
    """Use the same backend switches in every executable example."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--backend", choices=("native", "flash_attn3", "jax_aiter"), default="native"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if the requested GPU kernel is unavailable.",
    )
    parser.add_argument(
        "--length", type=int, default=32, help="Number of residues per sample."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def example_config():
    """Small learning configuration: both expanded kernel widths fit into 128."""
    return IPAConfig(
        c_s=64,
        c_z=32,
        c_hidden=32,
        no_heads=4,
        no_qk_points=4,
        no_v_points=8,
        z_factor_rank=2,
        pair_mode="factorized",
    )


def make_backend(name="native", strict=False):
    return BackendConfig(
        name=name,
        native_mode="flash",  # Tile both queries and keys using online softmax.
        chunk_size=16,
        key_chunk_size=16,
        kernel_dtype="bfloat16",  # Only the optional GPU packing uses this cast.
        fallback="error" if strict else "native",
    )


def make_batch(config, key, batch_size=1, length=32):
    """Make fixed synthetic inputs; these are shapes to learn from, not a dataset.

    Each residue has a local coordinate system. Identity matrices are valid
    rotations; arbitrary random 3x3 matrices would not be valid rotations.
    Translations and learned points use one consistent unit throughout.
    """
    if min(batch_size, length) < 1:
        raise ValueError("batch size and length must be positive")
    single_key, translation_key, first_key, second_key = jax.random.split(key, 4)
    factor_shape = (batch_size, length, config.z_factor_rank, config.c_z)
    return {
        "s": jax.random.normal(
            single_key, (batch_size, length, config.c_s), dtype=jnp.float32
        ),
        "rotations": jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float32), (batch_size, length, 3, 3)
        ),
        "translations": jax.random.normal(
            translation_key, (batch_size, length, 3), dtype=jnp.float32
        ),
        "z_factor_1": jax.random.normal(first_key, factor_shape, dtype=jnp.float32),
        "z_factor_2": jax.random.normal(second_key, factor_shape, dtype=jnp.float32),
    }


def apply_layer(layer, params, batch):
    """A layer returns a single-feature update, with shape [batch, length, c_s]."""
    return layer.apply(
        params,
        batch["s"],
        batch["rotations"],
        batch["translations"],
        z_factor_1=batch["z_factor_1"],
        z_factor_2=batch["z_factor_2"],
        # Omit masks because this synthetic batch contains no padded residues.
        # Real padded batches must pass a mask; fused adapters then use native.
    )


def describe_backend(layer, params, batch):
    """Report eligibility from the same projected shapes as this layer uses.

    Dispatch depends on shapes, dtype, device, masks and library availability,
    not the numeric values. Zero placeholders avoid duplicating projections.
    """
    config = layer.config
    batch_size, length = batch["s"].shape[:2]
    dtype = batch["s"].dtype
    scalar = jnp.zeros(
        (batch_size, length, config.no_heads, config.c_hidden), dtype=dtype
    )
    query_points = jnp.zeros(
        (batch_size, length, config.no_heads, config.no_qk_points, 3), dtype=dtype
    )
    value_points = jnp.zeros(
        (batch_size, length, config.no_heads, config.no_v_points, 3), dtype=dtype
    )
    factors = project_pair_factors(params, batch["z_factor_1"], batch["z_factor_2"])
    return backend_report(
        scalar,
        scalar,
        query_points,
        value_points,
        factors=factors,
        backend=layer.backend,
    )


def make_example(name="native", strict=False, seed=0, batch_size=1, length=32):
    config = example_config()
    layer = InvariantPointAttention(config, make_backend(name, strict))
    parameter_key, batch_key = jax.random.split(jax.random.key(seed))
    params = layer.init(parameter_key)
    batch = make_batch(config, batch_key, batch_size, length)
    return layer, params, batch
