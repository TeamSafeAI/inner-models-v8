"""
Gated path — STDP that only learns when a modulator is active.

Same as plastic, but weight updates only happen when the modulator
neuron group has enough recent activity. Three-factor learning:
pre-synaptic, post-synaptic, and modulatory signal.

Good for reward-driven learning, conditional associations.
"""
import math

DEFAULTS = {
    'learning_rate': 0.01,
    'w_min': 0.0,
    'w_max': 10.0,
    'tau_plus': 20.0,
    'gate_threshold': 0.3,
    'modulator_group': [],
}

INITIAL_STATE = {
    'eligibility': 0.0,
}


def on_source_fired(syn):
    """Source fired. Deliver weight and mark eligibility."""
    syn['eligibility'] += 1.0
    return syn['weight']


def on_target_fired(syn, modulator_activity=0.0):
    """Target fired. Apply STDP only if modulator gate is open.

    modulator_activity: fraction of modulator group that fired recently
                        (provided by the runner from its tracking)
    """
    elig = syn['eligibility']
    if elig <= 0:
        return

    if modulator_activity < syn['gate_threshold']:
        return

    w = syn['weight']
    lr = syn['learning_rate']
    w_min = syn['w_min']
    w_max = syn['w_max']
    range_w = w_max - w_min

    if range_w < 1e-6:
        return

    if w_min < 0 and w_max <= 0:
        dw = -lr * elig * (w - w_min) / range_w
    else:
        dw = lr * elig * (w_max - w) / range_w

    syn['weight'] = max(w_min, min(w_max, w + dw))


def per_tick(syn):
    """Eligibility decays exponentially."""
    tau = syn['tau_plus']
    if tau > 0:
        syn['eligibility'] *= math.exp(-1.0 / tau)
