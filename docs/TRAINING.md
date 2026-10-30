# Training and choosing model sizes

## Run a complete training loop

[examples/train.py](../examples/train.py) includes initialization, a nonzero
fixed target, loss, gradients, gradient clipping, Adam updates and logging.
The optimizer is written in JAX so no extra framework is required.

```bash
python examples/train.py --backend native --steps 30
python examples/train.py --backend flash_attn3 --strict --steps 30
python examples/train.py --backend jax_aiter --strict --steps 30
```

The latter two commands require the matching hardware and optional library.
Omit `--strict` to run through native fallback on an ordinary CPU. The program
prints which backend actually ran. A lower final loss demonstrates that the
training path works; it does not measure protein or RNA modeling quality.

The example freezes a separately initialized teacher and fits its outputs on
one synthetic batch. The teacher has a nonzero output projection. The student
uses upstream's zero output projection, so its initial loss is nonzero and the
first update trains the output projection. This avoids a misleading example
that simply minimizes the squared output of an already-zero layer.

```bash
# Increase the problem size or change optimizer settings explicitly.
python examples/train.py --backend native --length 64 --batch-size 2 \
  --steps 100 --learning-rate 0.0003 --log-every 10
```

Use [examples/structure_model.py](../examples/structure_model.py) to execute the
complete structure-refinement model. The same backend names select its IPA
kernels; its other building blocks run as ordinary JAX computations.

## From the example to a real task

1. Replace the synthetic batch with residue features, valid residue coordinate
   systems and target structures from your dataset. Split train/validation/test data to avoid
   closely related structures leaking across splits.
2. Choose the prediction target. For supervised structure refinement, predict clean
   orientations and positions from perturbed ones. For diffusion or flow matching,
   supply the task's noise/time conditioning and predict its required target.
3. Add the appropriate loss. A Cartesian translation MSE needs predictions and
   targets in the same global coordinate system. Losses such as FAPE compare
   geometry expressed in each residue's coordinate system; a full structure task
   also needs suitable rotation, torsion and geometry terms. The model does not supply these objectives.
   [AlphaFold2's FAPE description](https://www.nature.com/articles/s41586-021-03819-2).
4. Apply masks to both the model and loss. Normalize by valid residues or valid
   pairs, with a safe denominator for empty samples. Never let padding contribute
   to the reported loss.
5. Save parameters, optimizer state, random-key state, configuration and coordinate
   conventions. Monitor held-out structure metrics as well as training loss.

`params` is a pytree, so the example's optimizer can be replaced with your
preferred JAX optimizer. A useful training step has the same shape as the example:

```python
@jax.jit
def train_step(params, optimizer_state, batch):
    loss, gradients = jax.value_and_grad(loss_function)(params, batch)
    params, optimizer_state = optimizer_update(params, gradients, optimizer_state)
    return params, optimizer_state, loss
```

Here `loss_function` and `optimizer_update` are your task-specific functions;
this short schematic complements the complete runnable implementation above.

## What sizes did the original models use?

`c_hidden` is the scalar width **per attention head**. Increasing both it and the
number of heads multiplies projection size. The paper's IPA dimensions and this
repository's default configuration are different reference points:

| Parameter | AlphaFold2 original IPA | Pinned FlashIPA repository defaults | Runnable small example |
| --- | ---: | ---: | ---: |
| Single channels `c_s` | 384 | 256 | 64 |
| Pair channels `c_z` | 128 | 128 | 32 |
| Scalar channels/head `c_hidden` | 16 | 128 | 32 |
| Heads `no_heads` | 12 | 8 | 4 |
| Query/key points/head | 4 | 8 | 4 |
| Value points/head | 8 | 12 | 8 |
| Pair representation | Dense | Factorized | Factorized |
| Factor rank | Not applicable | 2 | 2 |
| Structure blocks | 8 | 6 | One IPA layer |

Sources: [AlphaFold2 configuration](https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/config.py),
[AlphaFold2 paper and supplement](https://www.nature.com/articles/s41586-021-03819-2),
[pinned FlashIPA defaults](https://github.com/flagshippioneering/flash_ipa/blob/fdeca12cedda766c5dc8ac484cc9eb4b78738500/src/flash_ipa/ipa.py).
The FlashIPA column is the released code's default/example, not a claim that
every experiment in the paper used that configuration. For example, its FoldFlow
experiment used hidden width 128 and five blocks, compared with width 256 and
four blocks in the original model. [FlashIPA §3.3](https://arxiv.org/html/2505.11580v1#S3.SS3).

```python
# Original scalar/point dimension preset. This remains this port's layer:
# its pair down-projection and projection conventions follow FlashIPA upstream.
config = IPAConfig.alphafold2()
# This preset uses dense pairs and therefore native execution.
```

Matching these sizes does not reproduce AlphaFold2: its data pipeline,
Evoformer, losses and training system are not included here. The default JAX
structure model follows FlashIPA's small example architecture.

## Practical tuning tips

These are starting heuristics for a new task, not a published training recipe.

| Symptom or goal | What to try first |
| --- | --- |
| Establish a working baseline | Use the small example, float32 native execution, and overfit one real batch. Check masks, shapes and coordinate units. |
| Loss explodes or becomes NaN | Inspect input finiteness and proper rotations, lower the learning rate, and inspect gradient norms. The example clips the global norm to 1. |
| Choose an initial learning rate | Try Adam near `1e-3` for the tiny example; for a new larger model compare `1e-4`, `3e-4`, and `1e-3` on held-out loss. Add warmup if early updates are unstable. |
| Training is stable but underfits | Increase channels or blocks gradually; measure validation quality and memory after each change. |
| Memory runs out | Reduce batch size or sequence length, use native tiling, and inspect dense pair tensors and sequence attention. Gradient accumulation changes effective batch size without growing one forward pass. |
| Need a longer sequence | Bucket lengths to reuse JIT compilations. Profile full-model forward/backward; a flash IPA core alone does not remove every quadratic tensor. |
| Increase factor rank | Treat rank as capacity and an approximation choice. Compare validation quality; higher rank increases packed kernel widths. |
| Switch to FP16/BF16 kernels | Compare outputs, gradients and learning curves against native float32 before scaling training. Keep master parameters and optimizer moments in float32. |

For factorized IPA, expanded widths are
`c_hidden + 5*no_qk_points + rank` for Q/K and
`c_hidden + 3*no_v_points + rank*(c_z//4)` for V.
The released defaults give **170 and 228**: they fit this AITER adapter's width
limit but exceed this FA3 adapter's conservative 128 limit. The small example
has widths **54 and 72**, fitting both. Changing the number of heads affects
memory and parameters even though these per-head width formulas do not include it.

Do not omit a required padding mask just to reach a fused path. Use native or
build genuinely unpadded fixed-length batches. See [Backends](BACKENDS.md) for
placement, dtype and masking limitations.
