"""JAX feature helpers corresponding to upstream ``flash_ipa.utils``.

Distances use the same units as the input positions. Use the conversion
constants explicitly when a model expects nanometres rather than Angstroms.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral

import jax
import jax.numpy as jnp

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


def _positions(pos):
    pos = jnp.asarray(pos)
    if pos.ndim < 2 or pos.shape[-1] != 3:
        raise ValueError("pos must have shape [..., N, 3]")
    if not jnp.issubdtype(pos.dtype, jnp.floating):
        raise ValueError("pos must contain floating point coordinates")
    return pos


def _distance_bins(distances, min_bin, max_bin, num_bins, dtype):
    if not isinstance(num_bins, Integral) or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")
    if min_bin >= max_bin:
        raise ValueError("min_bin must be smaller than max_bin")
    lower = jnp.linspace(min_bin, max_bin, num_bins, dtype=dtype)
    upper = jnp.concatenate((lower[1:], jnp.asarray([1e8], dtype=dtype)))
    # Strict edges reproduce upstream: distances exactly on an edge get zeros.
    bins = (distances[..., None] > lower) & (distances[..., None] < upper)
    return (bins & jnp.isfinite(distances[..., None])).astype(dtype)


def calc_distogram(pos, min_bin, max_bin, num_bins):
    """Encode all pair distances as bins: ``[...,N,N,num_bins]``.

    Bin lower bounds are evenly spaced from ``min_bin`` to ``max_bin``;
    the final upper bound is ``1e8``. Following upstream, edges are strict,
    and values equal to an edge have no active bin. This is a hard geometric
    feature, so its bin assignments are not differentiable in the positions.
    """
    pos = _positions(pos)
    distances = jnp.linalg.norm(pos[..., :, None, :] - pos[..., None, :, :], axis=-1)
    return _distance_bins(distances, min_bin, max_bin, num_bins, pos.dtype)


def knn_indices(pos, k, *, mask=None):
    """Return nearest-neighbor distances and indices, each ``[...,N,k]``.

    Self matches are excluded. ``k`` must be static during JIT and satisfy
    ``1 <= k < N``. An optional boolean ``[...,N]`` mask excludes padded nodes.
    If fewer than k valid neighbors exist, distances are infinity; the matching
    indices are placeholders that remain in bounds. Ignore these infinite slots.
    This computes all pair distances and has quadratic memory cost.
    """
    pos = _positions(pos)
    length = pos.shape[-2]
    if not isinstance(k, Integral) or not 1 <= k < length:
        raise ValueError("k must be an integer with 1 <= k < number of nodes")
    distances = jnp.linalg.norm(pos[..., :, None, :] - pos[..., None, :, :], axis=-1)
    valid = ~jnp.eye(length, dtype=bool)
    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)
        if mask.shape != pos.shape[:-1]:
            raise ValueError("mask must match the position batch and node dimensions")
        valid = valid & mask[..., :, None] & mask[..., None, :]
    distances = jnp.where(valid, distances, jnp.inf)
    negative_distances, indices = jax.lax.top_k(-distances, k)
    return -negative_distances, indices


def calc_distogram_knn(pos, k, min_bin, max_bin, num_bins, *, mask=None):
    """Return neighbor distance bins ``[...,N,k,num_bins]`` and indices.

    Missing/padded neighbors receive all-zero bins when ``mask`` is provided.
    See ``knn_indices`` for the fixed-size neighbor contract.
    """
    pos = _positions(pos)
    distances, indices = knn_indices(pos, k, mask=mask)
    return _distance_bins(distances, min_bin, max_bin, num_bins, pos.dtype), indices


def get_index_embedding(indices, embed_size, max_len=2056):
    """Sine/cosine embeddings of integer positions or relative offsets.

    Returns ``(*indices.shape, embed_size)``. The embedding width must be a
    positive even integer, and the scale ``max_len`` must be positive. Frequencies
    and sine-then-cosine feature ordering follow the upstream implementation.
    """
    if not isinstance(embed_size, Integral) or embed_size <= 0 or embed_size % 2:
        raise ValueError("embed_size must be a positive even integer")
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    indices = jnp.asarray(indices)
    frequencies = jnp.arange(embed_size // 2, dtype=jnp.float32)
    phase = indices[..., None] * math.pi / (max_len ** (2 * frequencies / embed_size))
    return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)


def check_config_ipa(ipa_conf, mode):
    """Validate an IPA mode for a JAX config or legacy upstream config.

    JAX ``IPAConfig`` uses ``pair_mode``; backend selection is configured
    separately, so this validates pair-feature compatibility only. For legacy
    configs without ``pair_mode``, validate ``c_z``, ``z_factor_rank``, and
    ``use_flash_attn`` as upstream did. Returns ``None`` on success.
    """
    modes = {
        "orig_no_bias": (False, False, False),
        "orig_2d_bias": (True, False, False),
        "flash_no_bias": (False, False, True),
        "flash_1d_bias": (True, True, True),
        "flash_2d_factorize_bias": (True, True, True),
    }
    if mode not in modes:
        raise ValueError(f"unknown IPA mode {mode!r}; choose from {tuple(modes)}")

    def field(name):
        try:
            return (
                ipa_conf[name]
                if isinstance(ipa_conf, Mapping)
                else getattr(ipa_conf, name)
            )
        except (KeyError, AttributeError) as error:
            raise ValueError(f"IPA configuration is missing {name!r}") from error

    c_z, rank = field("c_z"), field("z_factor_rank")
    has_pair_mode = (
        "pair_mode" in ipa_conf
        if isinstance(ipa_conf, Mapping)
        else hasattr(ipa_conf, "pair_mode")
    )
    if has_pair_mode:
        pair_mode = field("pair_mode")
        expected_pair_mode = {
            "orig_no_bias": "none",
            "flash_no_bias": "none",
            "orig_2d_bias": "dense",
            "flash_1d_bias": "factorized",
            "flash_2d_factorize_bias": "factorized",
        }[mode]
        valid_width = c_z == 0 if pair_mode == "none" else c_z > 0
        valid_rank = pair_mode != "factorized" or rank > 0
        if pair_mode != expected_pair_mode or not valid_width or not valid_rank:
            raise ValueError(
                f"{mode} is incompatible with pair_mode={pair_mode!r}, c_z={c_z}, z_factor_rank={rank}"
            )
        return
    flash = field("use_flash_attn")
    if c_z < 0 or rank < 0 or (c_z > 0, rank > 0, bool(flash)) != modes[mode]:
        raise ValueError(
            f"{mode} is incompatible with c_z={c_z}, z_factor_rank={rank}, use_flash_attn={flash}"
        )
