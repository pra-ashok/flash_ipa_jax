# Executable examples

Install the repository first: `python -m pip install -e '.[test]'`.
Run commands below from the repository root.

| Script | What it demonstrates |
| --- | --- |
| `minimal.py` | One IPA update and a nonzero attention gradient |
| `train.py` | Complete pure JAX Adam loop fitting a fixed nonzero teacher |
| `structure_model.py` | Full edge/IPA/sequence/structure-refinement model and toy translation-denoising updates |
| `_common.py` | Shared backend options, proper rotations, positions and synthetic data |

```bash
python examples/minimal.py --backend native
python examples/train.py --backend native --steps 30
python examples/structure_model.py --backend native --steps 5

# Require real fused kernels on the corresponding supported GPU stack.
python examples/train.py --backend flash_attn3 --strict --steps 30
python examples/train.py --backend jax_aiter --strict --steps 30
python examples/structure_model.py --backend flash_attn3 --strict --steps 5
python examples/structure_model.py --backend jax_aiter --strict --steps 5
```

Each executable accepts `--backend`, `--strict`, `--length`, `--batch-size`
and `--seed`. Run `--help` for training-specific options. Without `--strict`,
unsupported GPU requests use native execution and print that decision.
All synthetic batches have no padding; real padded batches need explicit masks.

Synthetic optimization verifies that code can learn; it does not reproduce a
biomolecular training recipe. See [Training](../docs/TRAINING.md) for dimensions,
loss design and practical tuning, and [Backends](../docs/BACKENDS.md) for exact
kernel and accelerator limits.
