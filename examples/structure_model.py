"""Run and optimize the complete structure model on one synthetic denoising batch.

python examples/structure_model.py --backend native --steps 5
Choose flash_attn3 or jax_aiter and add --strict to require a real GPU kernel.
This small translation-only loss is a demonstration, not a biomolecular objective.
"""

import jax
import jax.numpy as jnp

from flash_ipa_jax import EdgeEmbedderConfig, IPAConfig, Model, ModelConfig

if __package__:
    from ._common import argument_parser, describe_backend, make_backend
    from .train import adam_update
else:
    from _common import argument_parser, describe_backend, make_backend
    from train import adam_update


def main():
    parser = argument_parser(__doc__)
    parser.set_defaults(length=8)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if min(args.length, args.batch_size, args.steps) < 1:
        raise ValueError("length, batch size and steps must be positive")

    # A tiny version of the upstream trunk. Pair mode and the selected kernel
    # are independent choices; these per-head dimensions fit both GPU adapters.
    ipa_config = IPAConfig(
        c_s=32,
        c_z=16,
        c_hidden=16,
        no_heads=2,
        no_qk_points=2,
        no_v_points=4,
        z_factor_rank=2,
    )
    edge_config = EdgeEmbedderConfig(c_s=32, c_p=16, feat_dim=8, num_bins=8, k=3)
    config = ModelConfig(
        node_embed_size=32,
        edge_embed_size=16,
        ipa=ipa_config,
        edge_features=edge_config,
        num_blocks=2,
        seq_tfmr_num_heads=4,
        seq_tfmr_num_layers=1,
    )
    model = Model(config, make_backend(args.backend, args.strict))
    parameter_key, feature_key, target_key, noise_key = jax.random.split(
        jax.random.key(args.seed), 4
    )
    params = model.init(parameter_key)

    # All translations entering/leaving Model are in Angstroms. We perturb a
    # fixed target and fit that batch. Proper identity rotations avoid invalid
    # random matrices. A real task also needs rotation/geometry objectives.
    shape = (args.batch_size, args.length)
    target = jax.random.normal(target_key, shape + (3,), dtype=jnp.float32)
    inputs = {
        "node_embeddings": jax.random.normal(
            feature_key, shape + (32,), dtype=jnp.float32
        ),
        "rotations": jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), shape + (3, 3)),
        "translations": target
        + 0.2 * jax.random.normal(noise_key, shape + (3,), dtype=jnp.float32),
    }
    # This batch has no padding. For real padding pass node_mask; the optional
    # fused adapters then select native, preserving correct mask semantics.
    _, first_factor, second_factor, _ = model.edge_embedder.apply(
        params["edge_embedder"],
        inputs["node_embeddings"],
        inputs["translations"],
        jnp.zeros_like(target),
        jnp.ones(shape, dtype=bool),
    )
    decision = describe_backend(
        model.ipa,
        params["trunk"][0]["ipa"],
        {
            "s": inputs["node_embeddings"],
            "z_factor_1": first_factor,
            "z_factor_2": second_factor,
        },
    )
    print(f"IPA backend: requested={decision.requested}, selected={decision.selected}")
    print(f"reason: {decision.reason}")

    def loss_function(parameters):
        prediction = model.apply(parameters, inputs)
        # Inputs and targets share a global coordinate system in this toy task.
        # Do not apply this Cartesian loss to independently aligned structures.
        return jnp.mean((prediction["pred_trans"] - target) ** 2)

    @jax.jit
    def train_step(parameters, first, second, step):
        loss, gradients = jax.value_and_grad(loss_function)(parameters)
        parameters, first, second, norm = adam_update(
            parameters, gradients, first, second, step, learning_rate=1e-3
        )
        return parameters, first, second, loss, norm

    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    for step in range(1, args.steps + 1):
        params, first, second, loss, norm = train_step(
            params, first, second, jnp.asarray(step, dtype=jnp.int32)
        )
        print(f"step {step}: loss={float(loss):.6f}, gradient_norm={float(norm):.4f}")

    prediction = jax.jit(model.apply)(params, inputs)
    print("predicted translations:", prediction["pred_trans"].shape)
    print("predicted rotations:", prediction["pred_rotmats"].shape)
    print("final loss:", float(jnp.mean((prediction["pred_trans"] - target) ** 2)))


if __name__ == "__main__":
    main()
