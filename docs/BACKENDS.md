# Backends, XLA and accelerators

A **backend** is the implementation that evaluates the same selected IPA
representation. Pair mode chooses the representation; backend chooses execution.
Dense and factorized models are not interchangeable merely by changing backend.

## Portable native JAX

```python
backend = BackendConfig(name="native", native_mode="flash",
                        chunk_size=64, key_chunk_size=64)
```

Three native modes are available:

| Mode | Intermediate attention computation | Typical reason to use it |
| --- | --- | --- |
| `dense` | Complete score/probability matrix | Small accuracy references and debugging |
| `chunked` | Query blocks against all keys | Simpler memory/performance comparison |
| `flash` (default) | Query and key tiles with stable online softmax | Portable tiled execution without a full attention matrix |

Native supports dense pair bias/values, factors, token masks, pair masks and
causal attention. It computes geometric scores directly from point differences
and accumulates at least in float32. Native `flash` uses JAX scans and
rematerialization with generic automatic differentiation. It has no handcrafted
backward kernel: autodiff may retain scan carries. Source-level tiling does not
guarantee FlashAttention's fused GPU SRAM traffic or a particular training-memory
bound. Measure actual device memory on the intended workload.

## NVIDIA FlashAttention-3

```python
backend = BackendConfig(name="flash_attn3", kernel_dtype="bfloat16",
                        fallback="error")
```

The adapter calls `flash_attn3_jax.flash_mha` with expanded geometric Q/K/V,
`softmax_scale=1.0`, and the causal flag. It conservatively requires one
Hopper device (H100/H200/GH200), FP16 or BF16 opt-in, and both expanded widths
at most 128. Features are padded to width 128. This reflects the inspected
binding's published-wheel contract; other builds may expose different limits.

Install a compatible JAX CUDA build, then follow the
[FA3 binding's installation instructions](https://github.com/kyutai-labs/flash-attn3-jax).
This package does not compile or bundle that library. Its upstream custom VJP
supplies first-order reverse-mode differentiation; forward-mode and higher
fused derivatives are not promised here.

## AMD JAX-AITER

```python
backend = BackendConfig(name="jax_aiter", kernel_dtype="bfloat16",
                        fallback="error")
```

This adapter calls `jax_aiter.mha.flash_attn_func`, also at scale 1.0, with
zero dropout and the causal flag. It targets the inspected gfx950/MI355X stack,
FP16/BF16, and widths at most 256; padded widths are 64, 128 or 256.
The inspected alpha2 instructions specify Python 3.12, JAX/jaxlib 0.11.0,
ROCm plugin/PJRT 0.11.0.post1 and TheRock 7.14. These are source inspection
references, not requirements imposed on native JAX.

Follow [JAX-AITER's installation instructions](https://github.com/ROCm/jax-aiter),
including its separately supplied MHA binaries. An ordinary AMD JAX installation
alone does not supply this optional kernel. It provides its own reverse-mode
rule for Q/K/V.

## Fallback is visible

Both GPU adapters use native execution when their contracts cannot be met:

- Missing optional library, unsupported hardware, precision or packed width.
- Any explicit token mask or pair mask, even an all-true mask.
- Arbitrary dense pair bias or query-dependent dense pair values.
- Float64 inputs, or requesting float32 kernel packing.

`fallback="native"` warns and continues. `fallback="error"` raises
`BackendUnavailableError`; use this when benchmarking or checking installation.
`backend_report(...)` reports eligibility for projected inputs outside JIT.
It checks every supplied tensor's dtype. For complete mixed-dtype reporting,
also pass its optional `k=`, `k_points=` and `head_weights=` arguments.
The examples print requested/selected names before executing. Runtime binary
compilation or launch errors remain visible; they are not caught and relabeled
as successful native execution. See [Kernel integration](KERNEL_INTEGRATION.md)
for what additional mask/dense-pair kernel support would require.

No GPU package is imported for native execution. Install optional packages
before compiling a function that requests them. Fused model sharding, `pmap`
and arbitrary additional `vmap` axes are not validated; begin with native.

## What does XLA do?

**XLA** means Accelerated Linear Algebra. `jax.jit` traces a numerical function
and sends a compiler representation to XLA, which optimizes operations and
lowers them for the chosen device. That can fuse compatible work and reduce
Python dispatch overhead. Shape changes can trigger new compilations.
[JAX JIT guide](https://docs.jax.dev/en/latest/jit-compilation.html),
[OpenXLA overview](https://openxla.org/xla).

An XLA compilation does not automatically turn arbitrary geometric attention
into a supported FA3 or AITER call. Those are explicit external kernel paths.
The native tiled algorithm remains JAX computation that the compiler lowers.

## Is it useful on TPUs?

Yes: use `name="native", native_mode="flash"` with a working TPU JAX
installation. Its JAX operations and gradients can be compiled for the TPU;
the NVIDIA and AMD binary kernels cannot run there. Tiling can help avoid a
full IPA attention matrix, but tile sizes and the larger model's bottlenecks
must be measured on the TPU. No TPU performance result is claimed here.
Dedicated TPU/Pallas kernels and validated multi-device sharding would be
additional work. [JAX TPU installation](https://docs.jax.dev/en/latest/installation.html#tpu).

## Other accelerators and placement

Use native wherever your JAX/PJRT installation supports the required operations.
A working JAX backend is the prerequisite, not a guarantee for every device.
The official installation matrix includes CPU, NVIDIA, AMD and TPU options,
with experimental Apple/Intel GPU support documented separately. This port
has not validated those experimental devices.
[JAX supported platforms](https://docs.jax.dev/en/latest/installation.html#supported-platforms).

On a machine with multiple devices, choose placement while creating inputs and
compiling, for example `with jax.default_device(jax.devices()[0]): ...`.
Dispatch during tracing uses that default device. Do not reuse a compiled GPU
function with incompatible placement. Keep backend/configuration and `causal`
static when calling JIT; masks are array inputs and may change without changing
shape.

## Measure the complete workload

Compilation, execution and backward are different costs. Warm up compiled code,
synchronize outputs with `block_until_ready()`, and report actual dispatch.
[JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html).

```bash
python benchmarks/benchmark.py --backend native --native-mode flash --lengths 128 512
python benchmarks/benchmark.py --backend flash_attn3 --dtype bfloat16 --strict --lengths 128 512
python benchmarks/benchmark.py --backend jax_aiter --dtype bfloat16 --strict --lengths 128 512
```

The complete `Model` also has dense sequence attention. Its local-neighbor
edge features may compute a dense distance matrix. Factorized IPA therefore
does not make the complete model linear-memory. Dense pair input already costs
quadratic memory regardless of how attention itself is tiled.
