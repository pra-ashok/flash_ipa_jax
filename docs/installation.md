# Installation

Install the JAX runtime for your machine, then install `flash-ipa-jax`. The three attention adapters share one model API; their optional libraries depend on the available accelerator.

| Infrastructure | Adapter name | Required runtime and optional kernel |
| --- | --- | --- |
| CPU, for tests and small examples | `native` | CPU JAX |
| TPU, through XLA | `native` | TPU JAX and `libtpu` |
| NVIDIA GPU, through XLA | `native` | CUDA JAX |
| NVIDIA Hopper GPU, fused attention | `flash_attn3` | CUDA JAX and `flash-attn3-jax` |
| AMD GPU, through XLA | `native` | Compatible ROCm JAX plugin/PJRT |
| AMD MI355X, fused attention | `jax_aiter` | Compatible ROCm JAX and JAX-AITER with MHA binaries |

**XLA is the compiler used by JAX, not a fourth adapter or an extra package to install.** `native` implements tiled online-softmax IPA using JAX operations;
`jax.jit` compiles those operations for the selected device. The GPU adapters call external fused kernels through JAX. See [Backends](BACKENDS.md) for their supported inputs and [OpenXLA](https://openxla.org/xla) for the compiler.

The commands below were checked against upstream installation documentation on 30 October 2026. Hardware-specific recipes describe upstream requirements; this repository's executed checks are recorded in [Validation](../VALIDATION.md).

## Start with a separate environment

Run these commands from the repository root on Linux or macOS. Python 3.12 is a convenient common choice for the optional GPU bindings. The package itself requires Python 3.11 or newer; wheel availability also depends on the chosen JAX and kernel releases.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use a separate environment for each accelerator stack. If a cluster or managed image already supplies a working JAX runtime, keep its matched versions and install this project into that environment instead.

## Optional package extras

The runtime can also be selected while installing this project:

| Editable install from the repository root | Runtime selected |
| --- | --- |
| `python -m pip install -e '.[cpu,test]'` | CPU JAX and pytest |
| `python -m pip install -e '.[tpu]'` | TPU JAX |
| `python -m pip install -e '.[cuda12]'` | CUDA 12 JAX |
| `python -m pip install -e '.[cuda13]'` | CUDA 13 JAX |
| `python -m pip install -e '.[rocm7]'` | JAX with the system ROCm 7 plugin |

Use one accelerator extra per environment. The `rocm7` extra requires JAX 0.11 or newer and Python 3.12 or newer. These extras select a runtime; host requirements still apply. They do not install FA3 or AITER. Follow the pinned recipes below for those optional kernels, particularly when preserving an existing cluster runtime.

## CPU: testing only

Use the CPU path for correctness tests, debugging and tiny optimization examples. Use one of the accelerator setups below for model training. CPU timings do not establish GPU/TPU training performance.

```bash
python -m pip install --upgrade jax
python -m pip install -e '.[test]'
JAX_PLATFORMS=cpu python -m pytest -q -m 'not gpu'
JAX_PLATFORMS=cpu python examples/train.py --backend native --length 16 --steps 5
```

CPU JAX does not require either optional GPU binding. [JAX CPU installation](https://docs.jax.dev/en/latest/installation.html#cpu).

## XLA on TPU

On a provisioned Google Cloud TPU VM:

```bash
python -m pip install --upgrade 'jax[tpu]'
python -m pip install -e .
python -c "import jax; print(jax.devices())"
JAX_PLATFORMS=tpu python examples/train.py --backend native --length 32 --steps 5
```

The TPU extra installs the associated runtime, including `libtpu`.
[JAX TPU installation](https://docs.jax.dev/en/latest/installation.html#google-cloud-tpu).

Select `BackendConfig(name="native", native_mode="flash")` for TPU training. Tile sizes affect performance and need measurement on the intended TPU; this port has no dedicated TPU fused kernel or verified multi-device training recipe. CPU, CUDA and ROCm installations also use XLA when calling `jax.jit`.

## CUDA on NVIDIA GPUs

### Native JAX adapter

Choose one CUDA line that matches the machine. JAX's current Linux requirements include driver 580+ and SM 7.5+ for CUDA 13, or driver 525+ and SM 5.2+ for CUDA 12. Its pip extras install CUDA user-space libraries, while the host still needs a working NVIDIA driver. [JAX NVIDIA installation](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).

```bash
# Choose ONE of these in a fresh environment.
python -m pip install --upgrade 'jax[cuda13]'
# python -m pip install --upgrade 'jax[cuda12]'

python -m pip install -e .
JAX_PLATFORMS=cuda python examples/train.py --backend native --steps 5
```

If your infrastructure supplies CUDA/cuDNN, use JAX's documented
`cuda12-local` or `cuda13-local` extra and its library compatibility table. Check the complete model on `native` before switching to a fused adapter.

### FlashAttention-3 adapter

The published binding targets Linux x86_64, Python 3.11/3.12, CUDA 12.4+ and Hopper SM90, with kernel head width 128. This port pads supported expanded IPA widths to 128. Installing CUDA JAX alone does not install the binding. [FA3 installation](https://github.com/kyutai-labs/flash-attn3-jax#installation).

A conservative starting recipe uses JAX 0.7.1, within the binding's own `jax[cuda12]==0.7.*` test dependency:

```bash
# Use a separate CUDA 12 environment for this reference recipe.
python -m pip install 'jax[cuda12]==0.7.1'
python -m pip install flash-attn3-jax
python -m pip install -e .
python -c "from flash_attn3_jax import flash_mha; print('FA3 import: OK')"
JAX_PLATFORMS=cuda python examples/train.py --backend flash_attn3 --strict --steps 5
```

The JAX version choice follows the binding's
[build and test configuration](https://github.com/kyutai-labs/flash-attn3-jax/blob/main/pyproject.toml), and is not a real-GPU validation result from this project. Check the selected binding release's JAX/FFI compatibility before using newer JAX or CUDA 13. An import alone does not prove that a binary can launch or differentiate; `--strict` executes the requested path and its backward pass.

For a custom build, follow the binding's source-build instructions, including its CUDA toolkit, architecture and head-width options. This adapter currently checks for Hopper even if a custom binding supports additional devices. Other NVIDIA devices can use `native` with a supported CUDA JAX runtime.

## ROCm on AMD GPUs

### Native JAX adapter

Match the JAX, ROCm plugin/PJRT and ROCm runtime versions using  [AMD's installation and compatibility guidance](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html). For a machine with a compatible system ROCm 7 runtime:

```bash
python -m pip install --upgrade 'jax[rocm7-local]'
python -m pip install -e .
JAX_PLATFORMS=rocm python examples/train.py --backend native --steps 5
```

Alternatively, JAX documents architecture-specific TheRock extras with AMD's package index. For example, for `gfx950` on ROCm 7:

```bash
python -m pip install --upgrade 'jax[rocm7-device-gfx950]' \
  --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/
python -m pip install -e .
```

Choose the extra matching your GPU; pip does not detect its architecture. [JAX ROCm installation](https://docs.jax.dev/en/latest/installation.html#amd-gpu-linux).

These native recipes do not establish compatibility with JAX-AITER. Use its separate pinned stack below when requesting `jax_aiter`.

### JAX-AITER adapter

The inspected alpha2 reference stack is:

| Component | Upstream reference |
| --- | --- |
| GPU | MI355X, `gfx950` |
| ROCm | TheRock 7.14 GA |
| Python | 3.12 |
| `jax` / `jaxlib` | 0.11.0 |
| `jax-rocm7-plugin` / `jax-rocm7-pjrt` | 0.11.0.post1 |

Start with the matching upstream TheRock runtime image or equivalent environment, then install the pinned Python runtime packages:

```bash
python -m pip install \
  'jax==0.11.0' 'jaxlib==0.11.0' \
  'jax-rocm7-plugin==0.11.0.post1' 'jax-rocm7-pjrt==0.11.0.post1'
python -c "import jax; print(jax.devices())"
```

Upstream recommends its `+full` wheel, which includes the MHA libraries. After obtaining the matching wheel from the [JAX-AITER releases](https://github.com/ROCm/jax-aiter/releases), install it:

```bash
python -m pip install ./jax_aiter-0.1.0a2+full-cp312-cp312-manylinux_2_39_x86_64.whl
```

A plain wheel instead needs `jax-aiter-fetch-mha`. Import success does not replace an attention test. See the [JAX-AITER install guide](https://github.com/ROCm/jax-aiter#install) for wheel variants and its complete environment recipe.

**Release availability:** on 30 October 2026 the alpha2 release tag linked from the upstream README returned 404. Do not assume the documented wheel is already published on PyPI. If the matching wheel is unavailable, use a compatible source build in the upstream TheRock **development** image, following its [source-build instructions](https://github.com/ROCm/jax-aiter#build-from-source):

```bash
# Run in a separate working directory, keeping the environment above active.
git clone --recursive https://github.com/ROCm/jax-aiter.git
cd jax-aiter
export JA_ROOT_DIR="$PWD"
export GPU_ARCHS=gfx950
export AITER_SYMBOL_VISIBLE=1
make
python jax_aiter/jit/build_jit.py
make ja_mods
JA_WHEEL_VARIANT=full python -m pip install .
```

Record the chosen upstream commit and use its own compatible runtime if its requirements change. Return to the `flash-ipa-jax` repository root, then run:

```bash
python -m pip install -e .
python -c "from jax_aiter.mha import flash_attn_func; print('AITER MHA import: OK')"
JAX_PLATFORMS=rocm python examples/train.py --backend jax_aiter --strict --steps 5
```

## Verify which adapter actually runs

Check runtime versions and devices first:

```bash
python - <<'PY'
import jax
import jaxlib

print("JAX:", jax.__version__, "jaxlib:", jaxlib.__version__)
print("runtime:", jax.default_backend())
print("devices:", jax.devices())
PY
python -m pip check
```

The examples print `requested=...`, `selected=...` and the eligibility reason. A GPU request without `--strict` may warn and select native attention. Use `--strict` when checking installation or benchmarking. The small examples use factorized pair features, supported widths and no padding so they can exercise all three adapters. They are optimization checks, not trained protein models.

For programmatic checks, `backend_report(...)` inspects the projected inputs before JIT. It checks hardware, optional-library presence, precision, masks, pair representation and expanded width. See [Backends](BACKENDS.md) for that contract; unsupported dense pair inputs or masks can require native attention even when the GPU binding is installed correctly.

Keep a working stack reproducible with `python -m pip freeze > requirements-lock.txt`, and record the driver, device and container image alongside it. Avoid upgrading JAX independently of a compiled binding or ROCm plugin. Continue with [Usage](USAGE.md), [Training](TRAINING.md) and [Executable examples](../examples/README.md).
