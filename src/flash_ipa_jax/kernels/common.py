"""The positive per-head distance scale from the IPA equations."""

import math
import jax


def point_weights(raw_weights, n_points):
    return jax.nn.softplus(raw_weights) / math.sqrt(3 * n_points * 9.0 / 2)
