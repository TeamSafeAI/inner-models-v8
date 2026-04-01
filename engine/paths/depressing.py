"""
Depressing path — gets weaker with repeated use.

Each spike multiplies gain by depress_factor (< 1.0). Gain recovers
back to 1.0 when idle. No permanent learning — just short-term fatigue.

Good for novelty detection, adaptation, habituation.
"""
import math

DEFAULTS = {
    'depress_factor': 0.5,
    'tau_recovery': 500.0,
}

INITIAL_STATE = {
    'current_gain': 1.0,
}


def on_source_fired(syn):
    """Source fired. Reduce gain, deliver weight * gain."""
    syn['current_gain'] *= syn['depress_factor']
    return syn['weight'] * syn['current_gain']


def on_target_fired(syn):
    """Target fired. Depressing paths don't learn."""
    pass


def per_tick(syn):
    """Gain recovers back toward 1.0."""
    tau = syn['tau_recovery']
    if tau > 0:
        syn['current_gain'] = 1.0 + (syn['current_gain'] - 1.0) * math.exp(-1.0 / tau)
