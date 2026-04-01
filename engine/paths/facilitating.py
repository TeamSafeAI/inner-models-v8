"""
Facilitating path — gets stronger with repeated use.

Each spike increases the gain multiplier. Gain decays back to 1.0
when idle. No permanent learning — just short-term potentiation.

Good for attention, working memory, sustained activity detection.
"""
import math

DEFAULTS = {
    'facil_increment': 0.2,
    'tau_recovery': 200.0,
}

INITIAL_STATE = {
    'current_gain': 1.0,
}


def on_source_fired(syn):
    """Source fired. Increase gain, deliver weight * gain."""
    syn['current_gain'] += syn['facil_increment']
    return syn['weight'] * syn['current_gain']


def on_target_fired(syn):
    """Target fired. Facilitating paths don't learn."""
    pass


def per_tick(syn):
    """Gain decays back toward 1.0."""
    tau = syn['tau_recovery']
    if tau > 0:
        syn['current_gain'] = 1.0 + (syn['current_gain'] - 1.0) * math.exp(-1.0 / tau)
