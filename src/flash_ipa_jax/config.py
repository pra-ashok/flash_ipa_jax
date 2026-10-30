"""Model dimensions and named presets; backend settings live in backends.py."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IPAConfig:
    """Dimensions of one IPA layer, using the upstream configuration names.

    ``c_s`` is the width of each residue's single representation; ``c_z`` is
    the pair feature width; ``c_hidden`` is the scalar width *per head*.
    Point counts control geometric features, and ``z_factor_rank`` controls
    factorized pair capacity. Defaults follow the upstream FlashIPA example.
    """

    c_s: int = 256
    c_z: int = 128
    c_hidden: int = 128
    no_heads: int = 8
    no_qk_points: int = 8
    no_v_points: int = 12
    z_factor_rank: int = 2
    pair_mode: str = "factorized"
    eps: float = 1e-8

    @classmethod
    def alphafold2(cls, **overrides):
        """AlphaFold2 monomer scalar/point dimensions with dense input pairs.

        Pair outputs still use FlashIPA's c_z//4 down-projection, rather than
        AlphaFold2's full-width pair aggregation. This presets one layer's
        dimensions, not the complete AlphaFold2 architecture or training recipe.
        See docs/TRAINING.md for sources and distinctions.
        """
        dimensions = dict(
            c_s=384,
            c_z=128,
            c_hidden=16,
            no_heads=12,
            no_qk_points=4,
            no_v_points=8,
            pair_mode="dense",
        )
        return cls(**(dimensions | overrides))

    def __post_init__(self):
        if (
            min(
                self.c_s,
                self.c_hidden,
                self.no_heads,
                self.no_qk_points,
                self.no_v_points,
            )
            <= 0
        ):
            raise ValueError(
                "single, hidden, head and point dimensions must be positive"
            )
        if self.pair_mode not in ("none", "dense", "factorized"):
            raise ValueError("pair_mode must be none, dense, or factorized")
        if self.c_z < 0 or (self.pair_mode != "none" and self.c_z < 4):
            raise ValueError("pair modes require c_z >= 4")
        if self.pair_mode == "none" and self.c_z != 0:
            raise ValueError("pair_mode='none' requires c_z=0")
        if self.pair_mode == "factorized" and self.z_factor_rank < 1:
            raise ValueError("factorized mode requires a positive rank")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
