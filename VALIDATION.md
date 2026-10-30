# Validation record

Executed locally on 2026-10-30. These results describe CPU correctness and
optimization checks, not GPU/TPU performance or biomolecular prediction quality.

## Environment and test results

| Environment | Device | Result |
| --- | --- | --- |
| Python 3.12.3, JAX/jaxlib 0.12.0.1 | CPU | 109 passed, 8 skipped |
| Python 3.12.3, JAX/jaxlib 0.7.1 | CPU | 109 passed, 8 skipped |

The eight skipped cases require explicitly selected real NVIDIA or AMD kernels.
Emulated adapter tests are separate from those hardware tests. The supplied
GitHub Actions matrix is configuration, not a claim of remote CI execution.

## What the tests establish

- Dense, query-chunked and query/key-tiled IPA preserve scalar similarity,
  point distances, dense/factorized pair terms and causal/token/edge masks.
  Independent NumPy equations check forward values in float64. Tests cover
  tail tiles, empty rows, extreme logits and float32 rounding.
- The online-softmax implementation matches dense outputs and gradients for
  scalar Q/K/V, point Q/K/V, raw head weights and pair features. Fully masked
  rows have finite zero outputs and gradients. Directional finite differences
  independently check the attention and layer derivatives.
- Learned projections preserve upstream coordinate/feature ordering and biased
  factor conventions. Nonzero output projections exercise SE(3) invariance.
  Synthetic checkpoint conversion verifies IPA parameter layouts.
- Rotation/Rigid pytrees support JIT, gradients, vmap, composition, inversion,
  quaternion conversions and masked rotation/translation updates. Geometry and feature tests
  also cover distance bins, neighbor exclusion, index embeddings and initializers.
- All five upstream edge/model modes compile and execute. Independent NumPy
  references check factor compression and sequence attention. Nonzero geometric
  updates are equivariant, padded residue coordinate systems remain fixed, and gradients propagate
  through active IPA, sequence attention and learned edge features.
- Both layer and complete-model training steps reduce a nonzero target loss.
  Example regression tests run all three requested backend names on CPU and
  verify visible native fallback. Strict GPU requests correctly fail on CPU.
- Adapter tests check scale 1.0, padding, precision, causal flags, AITER tuple
  returns, mixed-dtype reporting and API drift. Binary launch failures remain
  visible. Native execution does not load optional GPU packages.
- A repository check enforces the 300-line limit across source, examples,
  benchmarks and tests. There are 48 Python modules including the figure generator; the longest has
  276 lines. The limit also covers Python files under `docs/`.

## Cleanup and architecture checks

The unused top-level `rigid.py` and `ipa.py` forwarding modules and the duplicate
`structure/common.py` helpers were removed. The essential rigid-transformation implementation
remains in `geometry/rigid.py`. Shared projection operations now live in
`linear.py`, and LayerNorm helpers live in `normalization.py`. Focused tests
verify weight-key splitting, independent uniform biases, custom and gating
initializers, biasless projections, leading array axes and gradients.

The [architecture guide](docs/architecture.md) includes full-model and IPA/backend
diagrams. SVG and PNG outputs are rendered from the checked-in DOT definitions
with the Python/Graphviz script. Both PNGs were visually checked; SVG XML and
PNG signatures were validated. Local documentation/image links resolve, and the repository
README indexes all guides, figures, examples, source groups and development tools.
Source-archive checks include the guide and diagram assets; wheel checks confirm
that removed wrappers are absent and the canonical geometry/normalization modules
are present. The terminology revision also preserved the runtime syntax trees
of all 47 numerical, example and test Python files after excluding docstrings.
Only explanatory text changed; the figure generator and DOT definitions were
redesigned separately.

## Executed example results

All results below used CPU and float32 synthetic inputs. These tasks demonstrate
optimization mechanics, not real protein/RNA training or benchmark accuracy.

| Command | Observed result |
| --- | --- |
| `python examples/train.py --backend native --steps 30` | Loss 0.466577 to 0.017045 |
| `python examples/train.py --backend native --length 8 --steps 12` | Loss 0.728565 to 0.050365 |
| `python examples/train.py --backend flash_attn3 --length 8 --steps 5` | Selected native; loss 0.728565 to 0.227214 |
| `python examples/train.py --backend jax_aiter --length 8 --steps 5` | Selected native; loss 0.728565 to 0.227214 |
| `python examples/structure_model.py --backend native --steps 5` | Loss 0.053222 to 0.003942; translations `(1,8,3)`, rotations `(1,8,3,3)` |

Both `examples/minimal.py` and its parameter-gradient calculation also executed.
The optional GPU kernels were not run by these fallback commands.

## Benchmark and packaging checks

[benchmarks/results/cpu-tiled-smoke.json](benchmarks/results/cpu-tiled-smoke.json)
records synchronized forward/backward CPU runs for lengths 16 and 64 with
16-by-16 native attention tiles. All three requested backend names selected
native. Maximum absolute output difference against the native reference was
about `6e-8`, reflecting tile reduction order. The JSON includes compiler
memory estimates, not measured accelerator peak memory. These short smoke
measurements do not establish comparative performance or a fused-kernel speedup.

The older `cpu-smoke.json` is a historical result from the earlier layer-only
implementation; it does not describe the new complete model.

Version 0.2.0 wheel and source archive builds succeeded. A wheel installed into
an isolated target imported all public components and executed a JIT-compiled
IPA layer with finite output. Local relative documentation links were checked.

## Remaining validation and implementation boundaries

- No NVIDIA/AMD fused forward/backward, TPU execution, accelerator throughput,
  actual peak device memory or distributed/sharded fused training was tested.
- Native tiled attention uses generic JAX autodiff and rematerialization. It
  has no handcrafted streaming backward; scan carries may be retained. No
  target-independent training-memory bound or SRAM fusion guarantee is claimed.
- The complete model still contains quadratic sequence attention and neighbor
  distance calculations. Dense pair inputs remain quadratic allocations.
- No direct execution comparison with the PyTorch upstream model or an actual
  trained checkpoint was performed. Equation/layout checks are separate evidence.
  Initialization samples and documented padding fixes differ from upstream.
- Only IPA checkpoint conversion is supplied; a whole-model converter and
  production data/loss/training pipeline are not included.
- Optional GPU adapters support their documented packed factorized/no-pair
  contracts. Explicit masks and general dense-pair requests use native execution.
  Dedicated dense-pair/mask GPU kernels and the upstream FFPA binding are not
  included. Fused derivative orders depend on the external libraries.

See [upstream coverage](docs/UPSTREAM.md), [backends](docs/BACKENDS.md) and
[kernel integration](docs/KERNEL_INTEGRATION.md) for the exact boundaries.

## Reproduce

After installing the package with test dependencies:

```bash
python -m pytest -q
python examples/minimal.py
python examples/train.py --backend native --steps 30
python examples/structure_model.py --backend native --steps 5
python benchmarks/benchmark.py --backend all --native-mode flash \
  --lengths 16 64 --chunk-size 16 --key-chunk-size 16 --repeats 3
```

On the matching GPU stack, these opt-in tests require actual kernel execution:

```bash
FLASH_IPA_GPU_BACKEND=flash_attn3 python -m pytest tests/test_gpu.py -q
FLASH_IPA_GPU_BACKEND=jax_aiter python -m pytest tests/test_gpu.py -q
python examples/train.py --backend flash_attn3 --strict --steps 30
python examples/train.py --backend jax_aiter --strict --steps 30
```

Each GPU-test command skips the other backend's cases. Strict execution cannot
be counted as a GPU success when only native fallback is available.
