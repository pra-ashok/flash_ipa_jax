"""Functional JAX port of the upstream example protein structure network.

The model refines residue coordinate systems from existing node features. Its
trunk contains edge embedding, IPA, sequence attention, node/edge transitions
and coordinate updates. It is a building block, not a pretrained folding system: users supply their own input embeddings, targets, and losses.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .backends import BackendConfig
from .edge_embedder import EDGE_MODES, EdgeEmbedder, EdgeEmbedderConfig
from .config import IPAConfig
from .geometry import create_rigid
from .layer import InvariantPointAttention
from .linear import init_linear, linear
from .normalization import init_norm, layer_norm
from .structure.sequence import SequenceTransformer
from .structure.transitions import (
    BackboneUpdate,
    EdgeTransition,
    StructureModuleTransition,
)


def _pair_mode(mode):
    if "no_bias" in mode:
        return "none"
    return "dense" if mode == "orig_2d_bias" else "factorized"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings; defaults match the upstream example model.

    The mode selects the pair representation, while :class:`BackendConfig`
    selects the attention implementation. Unlike upstream's mixed IPA config,
    block count and sequence-Transformer settings live at model level here.
    Omitted IPA/edge configs are inferred; explicit configs are checked.
    """

    mode: str = "flash_1d_bias"
    node_embed_size: int = 256
    edge_embed_size: int = 128
    ipa: IPAConfig | None = None
    edge_features: EdgeEmbedderConfig | None = None
    num_blocks: int = 6
    seq_tfmr_num_heads: int = 4
    seq_tfmr_num_layers: int = 2

    def __post_init__(self):
        if self.mode not in EDGE_MODES:
            raise ValueError(f"mode must be one of {EDGE_MODES}")
        if (
            min(
                self.node_embed_size,
                self.num_blocks,
                self.seq_tfmr_num_heads,
                self.seq_tfmr_num_layers,
            )
            < 1
            or self.edge_embed_size < 0
        ):
            raise ValueError("model dimensions and block/layer counts must be positive")
        if self.node_embed_size % self.seq_tfmr_num_heads:
            raise ValueError("node_embed_size must be divisible by seq_tfmr_num_heads")
        pair_mode = _pair_mode(self.mode)
        pair_channels = 0 if pair_mode == "none" else self.edge_embed_size
        if self.ipa is None:
            object.__setattr__(
                self,
                "ipa",
                IPAConfig(
                    c_s=self.node_embed_size, c_z=pair_channels, pair_mode=pair_mode
                ),
            )
        if self.edge_features is None:
            object.__setattr__(
                self,
                "edge_features",
                EdgeEmbedderConfig(
                    mode=self.mode,
                    c_s=self.node_embed_size,
                    c_p=self.edge_embed_size,
                    z_factor_rank=self.ipa.z_factor_rank,
                ),
            )
        if (self.ipa.c_s, self.ipa.c_z, self.ipa.pair_mode) != (
            self.node_embed_size,
            pair_channels,
            pair_mode,
        ):
            raise ValueError("IPA c_s/c_z/pair_mode must match model sizes and mode")
        edge = self.edge_features
        if (edge.c_s, edge.c_p, edge.mode) != (
            self.node_embed_size,
            self.edge_embed_size,
            self.mode,
        ):
            raise ValueError("edge c_s/c_p/mode must match model sizes and mode")
        if pair_mode == "factorized" and edge.z_factor_rank != self.ipa.z_factor_rank:
            raise ValueError("edge and IPA factor ranks must match")


class Model:
    """Refine coordinate systems with ``init(key)`` and ``apply(params, input_feats)``.

    Required inputs are ``node_embeddings [B,N,c_s]``, ``translations [B,N,3]``
    in Angstroms, and proper ``rotations [B,N,3,3]``. Optional inputs are
    ``node_mask [B,N]``, ``trans_sc [B,N,3]`` (self-conditioning coordinates),
    ``edge_embeddings [B,N,N,c_z]``, and ``edge_mask [B,N,N]``. Dense edge inputs
    are supported only by the two dense-input modes. Padded coordinate systems stay fixed.

    Zero-initialized coordinate update projections make the initial model preserve
    input coordinate systems. These projections still receive gradients during training.
    """

    def __init__(self, config=ModelConfig(), backend=BackendConfig()):
        self.config, self.backend = config, backend
        self.edge_embedder = EdgeEmbedder(config.edge_features)
        self.ipa = InvariantPointAttention(config.ipa, backend)
        self.sequence = SequenceTransformer(
            config.node_embed_size,
            config.seq_tfmr_num_heads,
            config.seq_tfmr_num_layers,
        )
        self.node_transition = StructureModuleTransition(config.node_embed_size)
        self.backbone_update = BackboneUpdate(config.node_embed_size)
        self.edge_transition = None
        if config.ipa.pair_mode != "none":
            self.edge_transition = EdgeTransition(
                mode="2d" if config.ipa.pair_mode == "dense" else "1d",
                node_embed_size=config.node_embed_size,
                edge_embed_in=config.edge_embed_size,
                edge_embed_out=config.edge_embed_size,
                z_factor_rank=config.ipa.z_factor_rank,
            )

    def init(self, key, *, dtype=jnp.float32):
        keys = iter(jax.random.split(key, self.config.num_blocks + 1))
        params = {"edge_embedder": self.edge_embedder.init(next(keys), dtype=dtype)}
        blocks = []
        for index in range(self.config.num_blocks):
            block_keys = iter(jax.random.split(next(keys), 6))
            channels = self.config.node_embed_size
            block = {
                "ipa": self.ipa.init(next(block_keys), dtype=dtype),
                "ipa_ln": init_norm(channels, dtype),
                "seq_tfmr": self.sequence.init(next(block_keys), dtype=dtype),
                "post_tfmr": init_linear(
                    next(block_keys), channels, channels, init="final", dtype=dtype
                ),
                "node_transition": self.node_transition.init(
                    next(block_keys), dtype=dtype
                ),
                "bb_update": self.backbone_update.init(next(block_keys), dtype=dtype),
            }
            if self.edge_transition is not None and index + 1 < self.config.num_blocks:
                block["edge_transition"] = self.edge_transition.init(
                    next(block_keys), dtype=dtype
                )
            blocks.append(block)
        params["trunk"] = tuple(blocks)
        return params

    def apply(self, params, input_feats):
        nodes = input_feats["node_embeddings"]
        translations, rotations = input_feats["translations"], input_feats["rotations"]
        if nodes.ndim != 3 or nodes.shape[-1] != self.config.node_embed_size:
            raise ValueError("node_embeddings must have shape [B,N,node_embed_size]")
        shape = nodes.shape[:2]
        if translations.shape != shape + (3,) or rotations.shape != shape + (3, 3):
            raise ValueError(
                "translations/rotations must have shape [B,N,3] / [B,N,3,3]"
            )
        explicit_mask = input_feats.get("node_mask")
        mask = jnp.ones(shape, bool) if explicit_mask is None else explicit_mask != 0
        if mask.shape != shape:
            raise ValueError("node_mask must have shape [B,N]")
        nodes = jnp.where(mask[..., None], nodes, 0)
        condition = input_feats.get("trans_sc", jnp.zeros_like(translations))
        edges, first, second, edge_mask = self.edge_embedder.apply(
            params["edge_embedder"],
            nodes,
            translations,
            condition,
            mask,
            input_feats.get("edge_embeddings"),
            input_feats.get("edge_mask"),
        )

        # Edge distances use Angstroms; IPA and coordinate updates use nm.
        # Predictions convert back to Angstroms, matching upstream preprocessing.
        frames = create_rigid(rotations, translations * 0.1)
        for index, block in enumerate(params["trunk"]):
            update = self.ipa.apply(
                block["ipa"],
                nodes,
                frames.get_rots().get_rot_mats(),
                frames.get_trans(),
                z=edges,
                z_factor_1=first,
                z_factor_2=second,
                mask=None if explicit_mask is None else mask,
            )
            nodes = layer_norm(block["ipa_ln"], nodes + update)
            sequence = self.sequence.apply(block["seq_tfmr"], nodes, mask)
            nodes = nodes + linear(block["post_tfmr"], sequence)
            nodes = self.node_transition.apply(block["node_transition"], nodes)
            nodes = jnp.where(mask[..., None], nodes, 0)
            frame_update = self.backbone_update.apply(block["bb_update"], nodes)
            frames = frames.compose_q_update_vec(frame_update, mask[..., None])
            if index + 1 < self.config.num_blocks and self.edge_transition is not None:
                updated = self.edge_transition.apply(
                    block["edge_transition"], nodes, edges, first, second
                )
                if self.config.ipa.pair_mode == "dense":
                    edges = jnp.where(edge_mask[..., None], updated, 0)
                else:
                    first, second = (
                        jnp.where(mask[:, :, None, None], factor, 0)
                        for factor in updated
                    )
        return {
            "pred_trans": frames.get_trans() * 10.0,
            "pred_rotmats": frames.get_rots().get_rot_mats(),
        }
