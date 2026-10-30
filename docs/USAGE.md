# Using layers and the structure model

## Pick the right entry point

- `InvariantPointAttention`: learned feature projections and one IPA update.
- `Model`: upstream-style edge embedding, stacked IPA/sequence blocks and orientation/position updates.
- `ipa_attention`: already projected scalar features and global points.
- `Rigid` and `Rotation`: differentiable rigid transformations and rotation conversions.

Import these classes from `flash_ipa_jax`. Geometry helper functions also live
in `flash_ipa_jax.geometry`. The old `flash_ipa_jax.rigid` and
`flash_ipa_jax.ipa` forwarding modules have been removed; the actual rigid-transformation
implementation remains in `geometry/rigid.py`.

Every learned component uses `params = component.init(key)` followed by
`component.apply(params, ...)`. Parameters are ordinary JAX pytrees. Configuration
objects contain static choices and dimensions; they are not trainable arrays.

## Layer inputs and outputs

Use `B` for batch size, `N` for residues, `H` for heads, and `r` for factor rank.

| Input | Shape | Meaning |
| --- | --- | --- |
| `s` | `[B,N,c_s]` | One feature vector per residue |
| `rotations` | `[B,N,3,3]` | Proper orthonormal local-to-global rotations |
| `translations` | `[B,N,3]` | Residue positions (origins of their coordinate systems) |
| Dense `z` | `[B,N,N,c_z]` | Features for every ordered residue pair |
| `z_factor_1`, `z_factor_2` | Each `[B,N,r,c_z]` | Learned pair factors |
| `mask` | Boolean `[B,N]` | True for valid residues |
| `pair_mask` | Boolean `[B,N,N]` | True for allowed attention edges |
| Output | `[B,N,c_s]` | Feature update, usually added to `s` |

The coordinate convention is `global_point = R @ local_point + t`. The layer does
not fix invalid rotation matrices or convert units. Use consistent units for
translations and learned points. Upstream models pass IPA translations in
nanometres. The higher-level `Model` handles its Angstrom-to-nanometre conversion.

A new layer uses a zero output projection. `layer.init(key, zero_init_output=False)`
is useful for diagnostics with visible point/attention gradients. Parameters
and head weights remain differentiable with either initialization; some inner
weights initially receive zero gradients behind the zero output projection.

## Choose a pair representation

### No pair features

Use `IPAConfig(c_z=0, pair_mode="none", ...)` and pass neither `z` nor factors.
The [README example](../README.md#a-small-ipa-layer) is executable as written.

### Factorized pairs

```python
config = IPAConfig(c_s=64, c_z=32, c_hidden=32, no_heads=4,
                   no_qk_points=4, no_v_points=8, z_factor_rank=2)
layer = InvariantPointAttention(config, BackendConfig(name="native"))
params = layer.init(jax.random.key(0))
# s, rotations and translations have the shapes in the table above.
# Each factor has shape [B,N,2,32]. EdgeEmbedder can learn these factors.
update = layer.apply(params, s, rotations, translations,
                     z_factor_1=first_factor, z_factor_2=second_factor)
```

Both factors are required. They are learned features, not an automatic lossless
factorization of arbitrary dense pairs. The complete runnable factor example is
[examples/minimal.py](../examples/minimal.py).

### Dense pairs

Use `IPAConfig(pair_mode="dense", ...)` and pass `z=z` instead of factors.
Dense pairs and arbitrary pair masks use native execution. For original
AlphaFold2 dimensions, use `IPAConfig.alphafold2()`; this presets dimensions,
not the whole AlphaFold2 architecture. See [Training](TRAINING.md).

## Generate edge features

```python
from flash_ipa_jax import EdgeEmbedder, EdgeEmbedderConfig

config = EdgeEmbedderConfig(c_s=64, c_p=32, z_factor_rank=2,
                            mode="flash_1d_bias", k=10)
embedder = EdgeEmbedder(config)
params = embedder.init(jax.random.key(0))
# Here translations and self_conditioning_translations are in Angstroms.
z, first, second, edge_mask = embedder.apply(
    params, single_features, translations, self_conditioning_translations,
    node_mask,
)
```

| Edge mode | Produced representation | Important cost |
| --- | --- | --- |
| `orig_no_bias` | No pair features | IPA uses scalar and point terms |
| `orig_2d_bias` | Dense pair features | Quadratic pair tensor |
| `flash_no_bias` | No pair features | Can use either native or a supported fused backend |
| `flash_1d_bias` | Learned per-residue pair factors | Avoids a dense pair feature tensor; neighbor search still computes dense distances |
| `flash_2d_factorize_bias` | Learned compression of dense features into factors | Still creates/accepts dense pairs; requires `max_len` |

The two dense-input modes accept optional `edge_embed=[B,N,N,c_p]` and
`edge_mask=[B,N,N]`. `LinearFactorizer` is the learned compression module used
by the last mode. It is not SVD and does not guarantee exact reconstruction.
Its trained length projection fixes a maximum supported length.

An edge embedder's `edge_mask` zeros edge **features**, matching upstream.
It does not forbid the corresponding attention probability. To disallow an
attention edge, pass `pair_mask` directly to the IPA layer; the full `Model`
currently exposes feature masks rather than a separate attention pair mask.

## Run the complete structure model

```python
from flash_ipa_jax import Model, ModelConfig

# Missing IPA/edge configs are inferred from the selected mode and sizes.
model = Model(ModelConfig(mode="flash_1d_bias"), BackendConfig(name="native"))
params = model.init(jax.random.key(0))
prediction = jax.jit(model.apply)(params, {
    "node_embeddings": single_features,  # [B,N,256] for the defaults
    "rotations": rotations,             # [B,N,3,3], proper rotations
    "translations": translations,       # [B,N,3], Angstroms
    "node_mask": node_mask,             # optional [B,N]
})
# prediction["pred_trans"]: [B,N,3], Angstroms
# prediction["pred_rotmats"]: [B,N,3,3]
```

Start with [examples/structure_model.py](../examples/structure_model.py) for a
smaller executable configuration. Model options `num_blocks`,
`seq_tfmr_num_heads` and `seq_tfmr_num_layers` live in `ModelConfig`.
Explicit IPA/edge configs must agree with its mode and embedding sizes.
Optional `trans_sc` provides self-conditioning translations; otherwise zeros
are used. Dense edge inputs use `edge_embeddings` and `edge_mask` in the batch.

Rotation/translation update projections start at zero, so the initial model
preserves input orientations and positions. This is expected initialization;
training updates those projections first. Padded residues keep their orientation
and position.
Omit `node_mask` only when every residue really is valid; any explicit mask
selects native IPA, even if all entries are true.

Sequence attention and the neighbor search can still allocate quadratic
intermediates. This model is a training building block, not a pretrained
folding predictor or a full reproduction of AlphaFold2.

## Coordinate systems and geometry helpers

```python
from flash_ipa_jax import Rigid, create_rigid

transforms = create_rigid(rotations, translations)
global_points = transforms.apply(local_points)       # e.g. points [B,N,3]
recovered = transforms.invert_apply(global_points)
matrix_transforms = transforms.to_tensor_4x4()
restored_transforms = Rigid.from_tensor_4x4(matrix_transforms)
```

`Rotation` and `Rigid` are pytrees: their array contents can pass through JIT
and autodiff. Points with extra head/point dimensions need matching broadcast
transformation dimensions, for example `transforms[..., None, None].apply(points)`.
Geometry uses JAX arrays throughout; it has no hidden PyTorch conversion.

## Low-level projected attention

`ipa_attention(q,k,v,q_points,k_points,v_points,head_weights, ...)` accepts:

| Argument | Shape / convention |
| --- | --- |
| `q`, `k` | `[B,N,H,D]` |
| `v` | `[B,N,H,Dv]` |
| `q_points`, `k_points` | `[B,N,H,P,3]`, global coordinates |
| `v_points` | `[B,N,H,Pv,3]`, global coordinates |
| `head_weights` | `[H]`, raw weights before softplus |
| `pair_bias` | Optional `[B,N,N,H]`, already scaled |
| `pair_values` | Optional `[B,N,N,C]` |
| `factors` | Optional `PairFactors(bias_q,bias_k,value_q,value_k)` |

The result contains `scalar`, global `points`, optional `pair`, and row-validity
flag `valid`. Invalid rows are exactly zero sentinels, with finite zero gradients.
All numeric inputs must be finite. Use boolean masks rather than NaN/infinite
biases. Self attention is supported; dropout, cross attention, GQA and KV caches
are not part of this API. Native supports JAX differentiation; optional fused
libraries determine their supported derivative orders.

## Load upstream IPA weights

```python
from flash_ipa_jax import from_torch_state_dict
params = from_torch_state_dict(state_dict, config, prefix="ipa.")
```

The converter validates shapes and transposes linear weights from upstream's
`[out,in]` to JAX's `[in,out]`. Arrays or detached PyTorch tensors are accepted
without making PyTorch a runtime dependency. It converts one IPA module,
not the entire structure model. Match pair mode and coordinate preprocessing.
No actual trained upstream checkpoint has been supplied for end-to-end parity
validation; synthetic layout and mathematical checks are recorded separately.
