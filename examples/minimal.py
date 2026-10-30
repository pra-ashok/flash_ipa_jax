"""One IPA forward/backward pass: python examples/minimal.py --backend native."""

import jax
import jax.numpy as jnp

if __package__:
    from ._common import apply_layer, argument_parser, describe_backend, make_example
else:
    from _common import apply_layer, argument_parser, describe_backend, make_example


def main():
    args = argument_parser(__doc__).parse_args()
    layer, params, batch = make_example(
        args.backend, args.strict, args.seed, args.batch_size, args.length
    )

    # Upstream initializes the output projection to zero for residual stability.
    # Use a nonzero output projection here so every attention gradient is visible.
    params = layer.init(jax.random.key(args.seed + 1), zero_init_output=False)
    decision = describe_backend(layer, params, batch)
    print(f"backend: requested={decision.requested}, selected={decision.selected}")
    print(f"reason: {decision.reason}")
    print("devices:", jax.devices())

    @jax.jit
    def loss_and_output(parameters):
        update = apply_layer(layer, parameters, batch)
        return jnp.mean(update**2), update

    # A diagnostic squared loss demonstrates autodiff. train.py uses a fixed
    # nonzero target instead, so training cannot succeed by predicting zeros.
    (loss, update), gradients = jax.value_and_grad(loss_and_output, has_aux=True)(
        params
    )
    print("output shape:", update.shape, "loss:", float(loss))
    print(
        "head-weight gradient norm:", float(jnp.linalg.norm(gradients["head_weights"]))
    )


if __name__ == "__main__":
    main()
