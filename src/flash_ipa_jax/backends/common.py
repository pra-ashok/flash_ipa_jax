"""Shared device inspection and packed-input preparation for GPU adapters."""

import inspect

import jax
import jax.numpy as jnp


def device_description(q):
    """Describe one device; traced inputs use the configured default device."""
    # Traced calls use the configured default device. See README for jit placement.
    if isinstance(q, jax.core.Tracer):
        default = jax.config.jax_default_device
        devices = [default] if default is not None else jax.local_devices()
    else:
        devices = list(q.devices())
    if len(devices) != 1:
        return "unsupported multi-device placement"
    device = devices[0]
    version = getattr(device.client, "platform_version", "")
    return f"{device.platform} {device.device_kind} {version}".lower()


def require_parameters(kernel, required, name):
    """Reject an optional package whose public function contract has changed."""
    if not required.issubset(inspect.signature(kernel).parameters):
        raise ImportError(
            f"{name} public API is incompatible with the inspected adapter"
        )
    return kernel


def pad_qkv(q, k, v, *, padded_dim, dtype):
    """Pad packed Q/K/V to one supported width and the opted-in dtype.

    Zero padding preserves the packed logits and retained output channels.
    The kernel must still receive softmax_scale=1.0 after padding.
    """

    def pad(array):
        padding = ((0, 0), (0, 0), (0, 0), (0, padded_dim - array.shape[-1]))
        return jnp.pad(array, padding).astype(dtype)

    return tuple(map(pad, (q, k, v)))
