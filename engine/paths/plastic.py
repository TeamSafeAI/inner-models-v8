"""
Plastic path — STDP learning with LTP and LTD.

Pre-before-post (causal): potentiation (LTP).
Post-before-pre (anti-causal): depression (LTD).
Soft-bounded: slows down as weight approaches limit.
Sign-aware: inhibitory paths strengthen inhibition, not flip sign.

Two eligibility traces:
  - elig_pre: marks source spikes (decays with tau_plus)
  - elig_post: marks target spikes (decays with tau_minus)

LTP: target fires while elig_pre > 0 -> strengthen
LTD: source fires while elig_post > 0 -> weaken

Ref: Bi & Poo 2001 asymmetric STDP window.
The learning rule lives HERE, not in the engine.
"""
import math

DEFAULTS = {
    'learning_rate': 0.01,
    'w_min': 0.0,
    'w_max': 10.0,
    'tau_plus': 20.0,
    'tau_minus': 20.0,
    'ltd_ratio': 0.5,   # LTD strength relative to LTP (biology: ~0.5-1.0)
}

INITIAL_STATE = {
    'eligibility': 0.0,      # pre-spike trace (elig_pre)
    'elig_post': 0.0,        # post-spike trace for LTD
}


def on_source_fired(syn):
    """Source neuron fired. Deliver weight, mark pre-trace, check LTD."""
    syn['eligibility'] += 1.0

    # LTD: source fired after target (post-before-pre)
    elig_post = syn.get('elig_post', 0.0)
    if elig_post > 0:
        w = syn['weight']
        lr = syn['learning_rate']
        ltd_ratio = syn.get('ltd_ratio', 0.5)
        w_min = syn['w_min']
        w_max = syn['w_max']
        range_w = w_max - w_min
        if range_w > 1e-6:
            if w_min < 0 and w_max <= 0:
                # Inhibitory: weaken inhibition (toward zero)
                dw = lr * ltd_ratio * elig_post * (w_max - w) / range_w
            else:
                # Excitatory: weaken (toward w_min)
                dw = -lr * ltd_ratio * elig_post * (w - w_min) / range_w
            syn['weight'] = max(w_min, min(w_max, w + dw))

    return syn['weight']


def on_target_fired(syn):
    """Target neuron fired. Apply LTP if pre fired recently, mark post-trace.

    Soft-bounded STDP (LTP):
    - Excitatory (w_min >= 0): dw = +lr * elig * (w_max - w) / range
    - Inhibitory (w_min < 0, w_max <= 0): dw = -lr * elig * (w - w_min) / range
      (inverted Hebbian -- pre before post STRENGTHENS inhibition)
    """
    # Mark post-spike for future LTD
    syn['elig_post'] = syn.get('elig_post', 0.0) + 1.0

    # LTP: target fired while pre-trace active
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
    """Both eligibility traces decay exponentially."""
    tau_plus = syn['tau_plus']
    if tau_plus > 0:
        syn['eligibility'] *= math.exp(-1.0 / tau_plus)
    tau_minus = syn.get('tau_minus', 20.0)
    if tau_minus > 0:
        syn['elig_post'] = syn.get('elig_post', 0.0) * math.exp(-1.0 / tau_minus)
