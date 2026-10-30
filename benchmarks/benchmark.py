"""Synchronized forward and full VJP timings, with explicit dispatch metadata."""

import argparse
from dataclasses import asdict
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flash_ipa_jax import BackendConfig, PairFactors, backend_report, ipa_attention


def make_inputs(n, batch, heads, hidden, points, value_points, rank, channels):
    rng = np.random.default_rng(23)

    def rand(shape):
        return jnp.asarray(rng.normal(size=shape) * 0.3, jnp.float32)

    args = (
        rand((batch, n, heads, hidden)),
        rand((batch, n, heads, hidden)),
        rand((batch, n, heads, hidden)),
        rand((batch, n, heads, points, 3)),
        rand((batch, n, heads, points, 3)),
        rand((batch, n, heads, value_points, 3)),
        rand((heads,)),
    )
    factors = PairFactors(
        rand((batch, n, heads, rank)),
        rand((batch, n, heads, rank)),
        rand((batch, n, rank, channels)),
        rand((batch, n, rank, channels)),
    )
    return args, factors


def measure(fn, state, repeats):
    start = time.perf_counter()
    executable = jax.jit(fn).lower(*state).compile()
    compile_s = time.perf_counter() - start
    for _ in range(2):
        jax.block_until_ready(executable(*state))
    samples = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = jax.block_until_ready(executable(*state))
        samples.append(1000 * (time.perf_counter() - start))
    stats = executable.memory_analysis()
    memory = (
        None
        if stats is None
        else {
            key: int(getattr(stats, key))
            for key in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        }
    )
    return {
        "compile_seconds": compile_s,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "samples_ms": samples,
        "compiler_memory_estimate_bytes": memory,
    }, output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["all", "native", "flash_attn3", "jax_aiter"],
        default="all",
    )
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 512])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--points", type=int, default=8)
    p.add_argument("--value-points", type=int, default=12)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--pair-channels", type=int, default=8)
    p.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    p.add_argument(
        "--native-mode", choices=["dense", "chunked", "flash"], default="flash"
    )
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--key-chunk-size", type=int, default=64)
    p.add_argument(
        "--mask",
        action="store_true",
        help="mask every fifth token; fused adapters must fall back",
    )
    p.add_argument("--causal", action="store_true")
    p.add_argument(
        "--strict", action="store_true", help="fail if any requested backend cannot run"
    )
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--output", type=Path)
    options = p.parse_args()
    if (
        min(
            *options.lengths,
            options.repeats,
            options.batch,
            options.heads,
            options.hidden,
            options.points,
            options.value_points,
            options.rank,
            options.pair_channels,
            options.chunk_size,
            options.key_chunk_size
        )
        <= 0
    ):
        p.error("all sizes and repeats must be positive")
    names = (
        ["native", "flash_attn3", "jax_aiter"]
        if options.backend == "all"
        else [options.backend]
    )
    result = {
        "jax": jax.__version__,
        "python": platform.python_version(),
        "devices": [str(x) for x in jax.devices()],
        "options": vars(options) | {"output": str(options.output)},
        "runs": [],
    }
    for n in options.lengths:
        state = make_inputs(
            n,
            options.batch,
            options.heads,
            options.hidden,
            options.points,
            options.value_points,
            options.rank,
            options.pair_channels,
        )
        a, f = state
        mask = (
            jnp.broadcast_to(jnp.arange(n) % 5 != 0, (options.batch, n))
            if options.mask
            else None
        )
        reference = ipa_attention(*a, factors=f, mask=mask, causal=options.causal)
        jax.block_until_ready(reference)
        for name in names:
            config = BackendConfig(
                name=name,
                kernel_dtype=options.dtype,
                fallback="error" if options.strict else "native",
                native_mode=options.native_mode,
                chunk_size=options.chunk_size,
                key_chunk_size=options.key_chunk_size,
                warn_on_fallback=False,
            )
            report = backend_report(
                a[0], a[2], a[3], a[5], factors=f, mask=mask, backend=config
            )

            def forward(a, f):
                return ipa_attention(
                    *a, factors=f, mask=mask, causal=options.causal, backend=config
                )

            def loss(a, f):
                out = forward(a, f)
                return sum(jnp.mean(x * x) for x in out[:3])

            forward_stats, actual = measure(forward, state, options.repeats)
            error = max(
                float(jnp.max(jnp.abs(x - y)))
                for x, y in zip(actual[:3], reference[:3])
            )
            backward_stats, _ = measure(
                jax.value_and_grad(loss, argnums=(0, 1)), state, options.repeats
            )
            run = {
                "length": n,
                "decision": asdict(report),
                "max_abs_error_vs_native": error,
                "input_dtype": "float32",
                "executed_attention_dtype": (
                    options.dtype if report.selected != "native" else "float32"
                ),
                "forward": forward_stats,
                "forward_and_backward": backward_stats,
            }
            result["runs"].append(run)
    payload = json.dumps(result, indent=2)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
