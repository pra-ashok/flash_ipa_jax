"""Optional checkpoint interoperability; PyTorch is never imported here."""

import jax.numpy as jnp
from .projections import parameter_shapes


def from_torch_state_dict(state, config, *, prefix=""):
    """Convert an IPA state dict; torch is optional and weights transpose [out,in].

    Only the IPA submodule is converted. Missing/incorrect shapes are errors;
    extra keys from a larger model are ignored. Arrays or CPU/GPU torch tensors
    are accepted, including bfloat16 (converted to float32 for interoperability).
    """

    def array(key):
        x = state[prefix + key]
        if hasattr(x, "detach"):
            x = x.detach().cpu().float().numpy()
        return jnp.asarray(x)

    result = {}
    for name, (inputs, outputs) in parameter_shapes(config).items():
        weight, bias = array(name + ".weight"), array(name + ".bias")
        if weight.shape != (outputs, inputs) or bias.shape != (outputs,):
            raise ValueError(f"incorrect checkpoint shape for {name}")
        result[name] = {"weight": weight.T, "bias": bias}
    result["head_weights"] = array("head_weights")
    if result["head_weights"].shape != (config.no_heads,):
        raise ValueError("incorrect checkpoint shape for head_weights")
    return result
