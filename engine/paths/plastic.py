"""
Plastic path — STDP learning.

Strengthens when pre fires before post (Hebbian).
Soft-bounded: slows down as weight approaches limit.
Sign-aware: inhibitory paths strengthen inhibition, not flip sign.

The learning rule lives HERE, not in the engine.
"""
import math

DEFAULTS = {
    'learning_rate': 0.01,
    'w_min': 0.0,
    'w_max': 10.0,
    'tau_plus': 20.0,
    'tau_minus': 20.0,
}

INITIAL_STATE = {
    'eligibility': 0.0,
}


def on_source_fired(syn):
    """Source neuron fired. Deliver weight and mark eligibility for learning."""
    syn['eligibility'] += 1.0
    return syn['weight']


def on_target_fired(syn):
    """Target neuron fired. Apply STDP if eligible.

    Soft-bounded STDP:
    - Excitatory (w_min >= 0): dw = +lr * elig * (w_max - w) / range
    - Inhibitory (w_min < 0, w_max <= 0): dw = -lr * elig * (w - w_min) / range
      (inverted Hebbian — pre before post STRENGTHENS inhibition)
    """
    elig = syn['eligibility']
    if elig <= 0:
        return

    w = syn['weight']
    lr = syn['learning_rate']
    w_min = syn['w_min']
    w_max = syn['w_max']
    range_w = w_max - w_min

    if range_w < 1e-6:
        return

    if w_min < 0 and w_max <= 0:
        # Inhibitory: strengthen inhibition (more negative)
        dw = -lr * elig * (w - w_min) / range_w
    else:
        # Excitatory: standard Hebbian
        dw = lr * elig * (w_max - w) / range_w

    syn['weight'] = max(w_min, min(w_max, w + dw))


def per_tick(syn):
    """Eligibility decays exponentially."""
    tau = syn['tau_plus']
    if tau > 0:
        syn['eligibility'] *= math.exp(-1.0 / tau)
