"""JAX IPA layers, protein structure model components, and rigid-body geometry.

Start with ``InvariantPointAttention`` for a single layer, or ``Model`` for the
upstream example structure trunk. All learned components use ``init``/``apply``
with ordinary JAX parameter pytrees. Optional GPU libraries load only on demand.
"""

from .attention import backend_report, ipa_attention
from .kernels.packing import pack_qkv
from .kernels.types import IPAOutput, PairFactors
from .backends import (
    BackendConfig,
    BackendDecision,
    BackendFallbackWarning,
    BackendUnavailableError,
)
from .checkpoint import from_torch_state_dict
from .config import IPAConfig
from .edge_embedder import EdgeEmbedder, EdgeEmbedderConfig
from .factorizer import LinearFactorizer
from .geometry import Rigid, Rotation, create_rigid
from .layer import InvariantPointAttention, apply_rigid, invert_rigid
from .linear import Linear
from .model import Model, ModelConfig
from .projections import project_pair_factors
from .structure import BackboneUpdate, EdgeTransition, StructureModuleTransition

__version__ = "0.2.0"

__all__ = [
    "IPAOutput",
    "PairFactors",
    "backend_report",
    "ipa_attention",
    "pack_qkv",
    "BackendConfig",
    "BackendDecision",
    "BackendFallbackWarning",
    "BackendUnavailableError",
    "IPAConfig",
    "InvariantPointAttention",
    "apply_rigid",
    "invert_rigid",
    "from_torch_state_dict",
    "project_pair_factors",
    "Rigid",
    "Rotation",
    "create_rigid",
    "Linear",
    "EdgeEmbedder",
    "EdgeEmbedderConfig",
    "LinearFactorizer",
    "Model",
    "ModelConfig",
    "BackboneUpdate",
    "EdgeTransition",
    "StructureModuleTransition",
]
