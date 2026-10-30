# Papers and primary implementation references

## Research

- Jumper et al. (2021), [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2).
  IPA is explained in the Methods; Supplementary Methods 1.8.2 and Algorithm 22
  give the detailed construction. This is the original AlphaFold2 IPA reference.
- Dao et al. (2022), [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
  Describes tiling, memory movement and exact online-softmax attention.
- Dao (2023), [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691).
  Improves work distribution and execution efficiency.
- Shah et al. (2024), [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608).
  Describes hardware-specific Hopper optimizations. Its reported results are
  results of that paper, not performance measurements of this port.
- Liu et al. (2025), [Flash Invariant Point Attention](https://arxiv.org/abs/2505.11580).
  Introduces the geometric/factorized reduction and evaluates biomolecular models.

## Source and accelerator documentation

- [Pinned upstream FlashIPA source and coverage](UPSTREAM.md).
- [AlphaFold2 model configuration](https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/config.py)
  for original IPA dimension names and values.
- [Kyutai FA3 JAX binding](https://github.com/kyutai-labs/flash-attn3-jax).
- [ROCm JAX-AITER](https://github.com/ROCm/jax-aiter).
- [JAX installation and supported platforms](https://docs.jax.dev/en/latest/installation.html).
- [JAX JIT compilation](https://docs.jax.dev/en/latest/jit-compilation.html).
- [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html).
- [OpenXLA overview](https://openxla.org/xla).

Source versions for the adapters and the FlashIPA port are recorded in
[UPSTREAM.md](UPSTREAM.md). Hardware support changes over time; follow the
installation guide for the exact versions installed on your machine.
