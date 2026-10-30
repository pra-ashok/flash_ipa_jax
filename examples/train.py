"""Learn a fixed teacher's IPA outputs with pure JAX Adam; no optimizer dependency.

Run: python examples/train.py --backend native --steps 30
GPU: add --backend flash_attn3 --strict, or --backend jax_aiter --strict.
This is an optimization smoke test, not protein structure model training.
"""

import jax
import jax.numpy as jnp

from flash_ipa_jax import InvariantPointAttention

if __package__:
    from ._common import (
        apply_layer,
        argument_parser,
        describe_backend,
        make_backend,
        make_example,
    )
else:
    from _common import (
        apply_layer,
        argument_parser,
        describe_backend,
        make_backend,
        make_example,
    )


def adam_update(params, gradients, first, second, step, learning_rate):
    """Adam with global gradient clipping; keep parameters and moments in float32."""
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(gradients)))
    clip_scale = jnp.minimum(1.0, 1.0 / (norm + 1e-8))
    gradients = jax.tree.map(lambda g: g * clip_scale, gradients)

    first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, first, gradients)
    second = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g**2, second, gradients)
    params = jax.tree.map(
        lambda p, m, v: p
        - learning_rate
        * (m / (1 - 0.9**step))
        / (jnp.sqrt(v / (1 - 0.999**step)) + 1e-8),
        params,
        first,
        second,
    )
    return params, first, second, norm


def run_training(
    *,
    backend="native",
    strict=False,
    seed=0,
    batch_size=1,
    length=32,
    steps=30,
    learning_rate=1e-3,
    log_every=10,
):
    """Return losses and parameters so the executable example can be tested."""
    if steps < 1 or learning_rate <= 0 or log_every < 1:
        raise ValueError("steps, learning_rate and log_every must be positive")
    layer, params, batch = make_example(backend, strict, seed, batch_size, length)
    decision = describe_backend(layer, params, batch)

    # Freeze a separately initialized, nonzero teacher. The target is computed
    # once with native float32 so all requested backends solve the same problem.
    teacher = InvariantPointAttention(layer.config, make_backend("native"))
    teacher_params = teacher.init(jax.random.key(seed + 10), zero_init_output=False)
    target = jax.lax.stop_gradient(
        jax.jit(lambda: apply_layer(teacher, teacher_params, batch))()
    )

    def loss_function(parameters):
        prediction = apply_layer(layer, parameters, batch)
        return jnp.mean((prediction - target) ** 2)

    # JIT compiles the loss, backward pass and parameter update together. Keep
    # the Python training loop outside jit so logging remains straightforward.
    @jax.jit
    def train_step(parameters, first, second, step):
        loss, gradients = jax.value_and_grad(loss_function)(parameters)
        parameters, first, second, norm = adam_update(
            parameters, gradients, first, second, step, learning_rate
        )
        return parameters, first, second, loss, norm

    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    initial_loss = float(jax.jit(loss_function)(params))
    print(f"backend: requested={decision.requested}, selected={decision.selected}")
    print(f"reason: {decision.reason}")
    print("devices:", jax.devices())
    print(f"initial loss: {initial_loss:.6f}")

    for step in range(1, steps + 1):
        params, first, second, loss, norm = train_step(
            params, first, second, jnp.asarray(step, dtype=jnp.int32)
        )
        if step == 1 or step % log_every == 0 or step == steps:
            # Converting to float synchronizes the device for meaningful logs.
            print(
                f"step {step:4d}: loss={float(loss):.6f}, gradient_norm={float(norm):.4f}"
            )

    final_loss = float(jax.jit(loss_function)(params))
    if not bool(jnp.isfinite(final_loss)):
        raise FloatingPointError("Training produced a non-finite loss")
    print(f"final loss: {final_loss:.6f}")
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "params": params,
        "selected_backend": decision.selected,
    }


def main():
    parser = argument_parser(__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    run_training(
        backend=args.backend,
        strict=args.strict,
        seed=args.seed,
        batch_size=args.batch_size,
        length=args.length,
        steps=args.steps,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
