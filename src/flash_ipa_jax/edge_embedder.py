"""Turn node features and distances into dense or factorized IPA pair features.

Adapted from the Flash IPA example model and Protein Frame Flow. The five
upstream mode names are kept so experiments can be compared directly. Only
``orig_2d_bias`` creates dense pair features for the IPA layers; the factorized
mode ``flash_2d_factorize_bias`` creates them temporarily before compression.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .factorizer import LinearFactorizer
from .linear import init_linear, linear
from .normalization import init_norm, layer_norm
from .utils import calc_distogram, calc_distogram_knn, get_index_embedding

EDGE_MODES = (
    "orig_no_bias",
    "orig_2d_bias",
    "flash_no_bias",
    "flash_1d_bias",
    "flash_2d_factorize_bias",
)


@dataclass(frozen=True)
class EdgeEmbedderConfig:
    """Feature sizes; the defaults match the upstream example architecture.

    ``k`` counts non-self neighbors. For short/padded proteins, missing neighbors
    are represented by zeros. The legacy options marked below were unused
    upstream and are accepted only to make old configuration files readable.
    """

    z_factor_rank: int = 2
    c_s: int = 256
    c_p: int = 128
    feat_dim: int = 64
    num_bins: int = 22
    mode: str = "flash_1d_bias"
    k: int = 10
    max_len: int | None = None
    single_bias_transition_n: int = 2  # Legacy, unused.
    relpos_k: int = 64  # Legacy, unused.
    use_rbf: bool = True  # Legacy, unused.
    num_rbf: int = 32  # Legacy, unused.
    self_condition: bool = True  # Legacy, unused.

    def __post_init__(self):
        if self.mode not in EDGE_MODES:
            raise ValueError(f"mode must be one of {EDGE_MODES}")
        if min(self.c_s, self.feat_dim, self.num_bins, self.k) < 1 or self.feat_dim % 2:
            raise ValueError(
                "feature sizes and k must be positive; feat_dim must be even"
            )
        if self.c_p < 0 or ("no_bias" not in self.mode and self.c_p < 1):
            raise ValueError("pair-feature modes require positive c_p")
        if (
            self.mode.startswith("flash_")
            and "no_bias" not in self.mode
            and self.z_factor_rank < 1
        ):
            raise ValueError("factorized modes require a positive z_factor_rank")
        if self.mode == "flash_2d_factorize_bias" and (
            self.max_len is None or self.max_len < 1
        ):
            raise ValueError("flash_2d_factorize_bias requires a positive max_len")


class EdgeEmbedder:
    """Functional edge feature generator: ``init(key)`` then ``apply(params,...)``.

    Positions and self-conditioning positions use the same units as input model
    translations (Angstroms). The distogram range is 0.001 to 20 Angstroms, as in
    upstream. This module creates features; the IPA backend is chosen separately.
    """

    def __init__(self, config=EdgeEmbedderConfig()):
        self.config = config
        self.factorizer = None
        if config.mode == "flash_2d_factorize_bias":
            self.factorizer = LinearFactorizer(
                config.max_len, config.c_p, config.z_factor_rank, config.c_p
            )

    def init(self, key, *, dtype=jnp.float32):
        c = self.config
        if "no_bias" in c.mode:
            return {}  # No unused parameters in a model without pair features.
        keys = iter(jax.random.split(key, 8))
        rank = c.z_factor_rank
        one_d = c.mode == "flash_1d_bias"
        specs = {
            "linear_s_p": (c.c_s, c.feat_dim * (2 * rank if one_d else 1)),
            "linear_relpos": (c.feat_dim, c.feat_dim * (4 * rank if one_d else 1)),
        }
        if one_d:
            for name in ("linear_t_distogram", "linear_sc_distogram"):
                specs[name] = (c.k * (c.num_bins + c.feat_dim), 2 * rank * c.num_bins)
        params = {
            name: init_linear(next(keys), inputs, outputs, init="torch", dtype=dtype)
            for name, (inputs, outputs) in specs.items()
        }
        params["edge_embedder"] = tuple(
            init_linear(next(keys), inputs, c.c_p, init="torch", dtype=dtype)
            for inputs in (3 * c.feat_dim + 2 * c.num_bins, c.c_p, c.c_p)
        )
        params["edge_norm"] = init_norm(c.c_p, dtype)
        if self.factorizer is not None:
            params["factorizer"] = self.factorizer.init(next(keys), dtype=dtype)
        return params

    def _mlp(self, params, features):
        for projection in params["edge_embedder"][:-1]:
            features = jax.nn.relu(linear(projection, features))
        return layer_norm(
            params["edge_norm"], linear(params["edge_embedder"][-1], features)
        )

    def embed_relpos(self, params, positions):
        relative = positions[:, :, None] - positions[:, None, :]
        return linear(
            params["linear_relpos"],
            get_index_embedding(relative, self.config.feat_dim, max_len=2056),
        )

    def _knn_features(self, positions, mask):
        """Keep the projection width fixed even when the protein has fewer than k neighbors."""
        c = self.config
        batch, length, _ = positions.shape
        count = min(c.k, length - 1)
        if count:
            bins, indices = calc_distogram_knn(
                positions,
                k=count,
                min_bin=1e-3,
                max_bin=20.0,
                num_bins=c.num_bins,
                mask=mask,
            )
            index_features = get_index_embedding(indices, c.feat_dim, max_len=2056)
            # A valid zero-distance neighbor can have no active distance bin,
            # so determine availability from node masks rather than bin values.
            neighbors = jnp.take_along_axis(mask[:, None, :], indices, axis=-1)
            available = (mask[:, :, None] != 0) & (neighbors != 0)
            available = available & (indices != jnp.arange(length)[None, :, None])
            index_features = jnp.where(available[..., None], index_features, 0)
            features = jnp.concatenate((bins, index_features), -1)
            features = jnp.pad(features, ((0, 0), (0, 0), (0, c.k - count), (0, 0)))
        else:
            features = jnp.zeros(
                (batch, length, c.k, c.num_bins + c.feat_dim), positions.dtype
            )
        return features.reshape(batch, length, -1)

    def fwd_1d(self, params, s, t, sc_t, p_mask):
        """Build factors of shape ``[B,N,rank,c_p]`` without dense edge features."""
        c = self.config
        batch, length, _ = s.shape

        def split_factors(features):
            return jnp.transpose(
                features.reshape(batch, length, 2, c.z_factor_rank, -1), (0, 2, 1, 3, 4)
            )

        node_features = split_factors(linear(params["linear_s_p"], s))
        indices = jnp.broadcast_to(jnp.arange(length), (batch, length))
        index_features = get_index_embedding(indices, c.feat_dim, max_len=2056)
        position_features = split_factors(
            linear(params["linear_relpos"], index_features)
        )
        distance_features = split_factors(
            linear(params["linear_t_distogram"], self._knn_features(t, p_mask))
        )
        condition_features = split_factors(
            linear(params["linear_sc_distogram"], self._knn_features(sc_t, p_mask))
        )
        factors = self._mlp(
            params,
            jnp.concatenate(
                (
                    node_features,
                    position_features,
                    distance_features,
                    condition_features,
                ),
                -1,
            ),
        )
        factors = jnp.where(p_mask[:, None, :, None, None] != 0, factors, 0)
        return factors[:, 0], factors[:, 1]

    def fwd_2d(self, params, s, t, sc_t, p_mask):
        """Build explicit ``[B,N,N,c_p]`` features for the dense/factorize modes."""
        c = self.config
        batch, length, _ = s.shape
        node = linear(params["linear_s_p"], s)
        shape = (batch, length, length, c.feat_dim)
        row, col = jnp.broadcast_to(node[:, :, None], shape), jnp.broadcast_to(
            node[:, None], shape
        )
        positions = jnp.broadcast_to(jnp.arange(length), (batch, length))
        kwargs = dict(min_bin=1e-3, max_bin=20.0, num_bins=c.num_bins)
        features = jnp.concatenate(
            (
                row,
                col,
                self.embed_relpos(params, positions),
                calc_distogram(t, **kwargs),
                calc_distogram(sc_t, **kwargs),
            ),
            -1,
        )
        return jnp.where(p_mask[..., None] != 0, self._mlp(params, features), 0)

    def apply(
        self,
        params,
        node_embed,
        translations,
        trans_sc,
        node_mask,
        edge_embed=None,
        edge_mask=None,
    ):
        """Return ``(dense_edges, factor_1, factor_2, dense_edge_mask)``.

        Dense inputs are accepted only by the two dense-input modes. ``edge_mask``
        masks edge features, matching upstream; ``node_mask`` also masks IPA
        attention when these features are used by :class:`~flash_ipa_jax.model.Model`.
        """
        c = self.config
        if node_embed.ndim != 3 or node_embed.shape[-1] != c.c_s:
            raise ValueError("node_embed must have shape [B,N,c_s]")
        batch, length, _ = node_embed.shape
        if (
            length < 1
            or translations.shape != (batch, length, 3)
            or trans_sc.shape != translations.shape
        ):
            raise ValueError("positions must have shape [B,N,3] with N > 0")
        if node_mask.shape != (batch, length):
            raise ValueError("node_mask must have shape [B,N]")
        if c.mode in ("orig_no_bias", "flash_no_bias", "flash_1d_bias"):
            if edge_embed is not None or edge_mask is not None:
                raise ValueError(f"{c.mode} does not accept dense edge inputs")
            if c.mode == "flash_1d_bias":
                first, second = self.fwd_1d(
                    params, node_embed, translations, trans_sc, node_mask
                )
                return None, first, second, None
            return None, None, None, None
        valid = (node_mask[:, :, None] != 0) & (node_mask[:, None, :] != 0)
        if edge_mask is not None:
            if edge_mask.shape != (batch, length, length):
                raise ValueError("edge_mask must have shape [B,N,N]")
            valid = valid & (edge_mask != 0)
        if edge_embed is None:
            edge_embed = self.fwd_2d(params, node_embed, translations, trans_sc, valid)
        elif edge_embed.shape != (batch, length, length, c.c_p):
            raise ValueError("edge_embed must have shape [B,N,N,c_p]")
        edge_embed = jnp.where(valid[..., None], edge_embed, 0)
        if c.mode == "orig_2d_bias":
            return edge_embed, None, None, valid
        first, second = self.factorizer.apply(
            params["factorizer"], edge_embed, node_mask
        )

        def unflatten(factor):
            return jnp.transpose(
                factor.reshape(batch, c.c_p, length, c.z_factor_rank), (0, 2, 3, 1)
            )

        return None, unflatten(first), unflatten(second), None
