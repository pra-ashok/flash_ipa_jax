# Upstream coverage and deliberate differences

Source inspection date: 2026-10-30. The reference FlashIPA revision is
`fdeca12cedda766c5dc8ac484cc9eb4b78738500`.
The original repository is PyTorch; this port implements numerical work in JAX.
It does not import PyTorch to run layers, geometry, edge features or the model.

## Module map

| Upstream file | JAX counterpart | Covered behavior |
| --- | --- | --- |
| [ipa.py](https://github.com/flagshippioneering/flash_ipa/blob/fdeca12cedda766c5dc8ac484cc9eb4b78738500/src/flash_ipa/ipa.py) | `layer.py`, `attention.py`, `kernels/`, `structure/transitions.py` | Scalar/point/pair projections, IPA, node/edge transitions and backbone updates |
| [model.py](https://github.com/flagshippioneering/flash_ipa/blob/fdeca12cedda766c5dc8ac484cc9eb4b78738500/src/flash_ipa/model.py) | `model.py`, `structure/sequence.py` | Edge embedding, stacked IPA, sequence Transformer, transitions and iterative structure refinement |
| [edge_embedder.py](https://github.com/flagshippioneering/flash_ipa/blob/fdeca12cedda766c5dc8ac484cc9eb4b78738500/src/flash_ipa/edge_embedder.py) | `edge_embedder.py` | All five mode names, learned factors, dense features and local-neighbor features |
| [factorizer.py](https://github.com/flagshippioneering/flash_ipa/blob/fdeca12cedda766c5dc8ac484cc9eb4b78738500/src/flash_ipa/factorizer.py) | `factorizer.py` | Learned length/channel compression and normalized pair factors |
| [rigid.py](https://github.com/flagshippioneering/flash_ipa/blob/fdeca12cedda766c5dc8ac484cc9eb4b78738500/src/flash_ipa/rigid.py) | `geometry/` | Rotation/rigid composition, inversion, quaternion/matrix conversion and coordinate-system helpers |
| [linear.py](https://github.com/flagshippioneering/flash_ipa/blob/fdeca12cedda766c5dc8ac484cc9eb4b78738500/src/flash_ipa/linear.py) | `linear.py`, `normalization.py`, `projections.py` | Initializers and learned projections with JAX-oriented weight layout |
| [utils.py](https://github.com/flagshippioneering/flash_ipa/blob/fdeca12cedda766c5dc8ac484cc9eb4b78738500/src/flash_ipa/utils.py) | `utils.py` | Unit constants, distance bins, neighbor lookup, index embeddings and config checks |

The public JAX API uses explicit `init(key)` / `apply(params, ...)` rather than
stateful PyTorch modules. Configuration separates architecture (`IPAConfig`,
`ModelConfig`) from execution (`BackendConfig`). Import the main classes directly from
`flash_ipa_jax`, and geometry helpers from `flash_ipa_jax.geometry`. The former
`flash_ipa_jax.ipa` and `flash_ipa_jax.rigid` compatibility wrappers were removed;
the implementations live in the modules shown above.

## Numerical and API differences

- Linear weights use `[in,out]`. `from_torch_state_dict` converts one IPA's
  projection layouts; automatic whole-model checkpoint conversion is not supplied.
- Initialization distributions follow their upstream roles, but PRNG samples
  differ. Sequence Transformer layers initialize independently; PyTorch's
  TransformerEncoder initially clones identical layer parameters.
- Boolean masks remove edges exactly. Invalid IPA rows have zero outputs and
  finite zero gradients, including after output bias. This replaces upstream's
  finite masking sentinel and artifacts from invalid residue coordinate systems.
- Geometric packing uses a common origin for better translation stability.
  Native tiled attention is an additional execution option.
- Neighbor features exclude padded neighbors. Short sequences pad missing
  neighbors to a fixed width. Dense compression masks padding before projection.
  These correct upstream edge cases rather than reproduce padding-dependent output.
- Equal-distance KNN ordering can differ. Hard distance-bin boundaries and
  discrete neighbor choices are not smooth coordinate operations, as upstream.
- No-pair modes omit unused edge-network parameters. Legacy configuration fields
  documented as unused do not activate hidden features.
- Explicit dense edge masks zero features, following upstream semantics. They
  do not prohibit attention edges; use the layer's `pair_mask` for that purpose.

The full model takes/returns Angstrom translations and converts to nanometres
inside IPA. The standalone layer performs no unit conversion. Projection/output
ordering and the dense-versus-factorized score conventions follow the pinned
source; [Kernel integration](KERNEL_INTEGRATION.md) spells out those equations.

This is not a complete AlphaFold2 implementation. It has no bundled dataset,
trained weights, task-specific loss or released training pipeline. Upstream's
FFPA path is not included. Dedicated dense-pair fused GPU kernels, dynamic mask
unpadding and validated distributed fused execution remain extension work.
The complete model still includes quadratic sequence-attention and neighbor
search intermediates; no full-model linear-memory guarantee is made.

## NVIDIA binding inspected

Reference revision `4af680283658f467ced12a0811894a672e8d87d1`:

- [Public API and custom VJP](https://github.com/kyutai-labs/flash-attn3-jax/blob/4af680283658f467ced12a0811894a672e8d87d1/src/flash_attn3_jax/flash.py).
- [README/build instructions](https://github.com/kyutai-labs/flash-attn3-jax/blob/4af680283658f467ced12a0811894a672e8d87d1/README.md).

The adapter uses `flash_mha(q,k,v,softmax_scale=1.0,is_causal=...)`, BSHD layout,
and FP16/BF16. The scale is a Python float. The public API does not accept our
arbitrary pair-mask/bias contract. We choose the conservative published-wheel
Hopper/128-width contract rather than infer every build's capabilities.

## AMD binding inspected

Reference revision `c290cdd5437936b702402a7dca7ef52b0acad624`:

- [Public attention API](https://github.com/ROCm/jax-aiter/blob/c290cdd5437936b702402a7dca7ef52b0acad624/jax_aiter/mha/mha.py).
- [README/build instructions](https://github.com/ROCm/jax-aiter/blob/c290cdd5437936b702402a7dca7ef52b0acad624/README.md).

The adapter uses `flash_attn_func(q,k,v,dropout_p=0.0,softmax_scale=1.0,
causal=...,return_lse=False)` and accepts its array or tuple return convention.
The inspected bias contract does not establish arbitrary batch/head IPA bias
and gradient support. Optional MHA binary import failures trigger visible native
fallback unless strict mode is requested.

Pinned source references make inspection reproducible; they do not establish
real-device execution or trained-checkpoint parity. See the current
[validation record](../VALIDATION.md). Kernel packages are not vendored, and no
upstream repository has been changed. Attribution remains in [NOTICE](../NOTICE)
and [LICENSE](../LICENSE).
