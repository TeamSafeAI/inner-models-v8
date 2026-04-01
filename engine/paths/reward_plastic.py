"""
Reward-plastic path — 3-factor STDP (pre, post, reward).

Biological model (dopamine-gated striatal STDP):
  1. Source fires -> fast trace builds (tau_trace ~20ms, like spike timing)
  2. Target fires -> trace converts to eligibility (slow decay, tau_eligible ~500ms)
  3. Reward signal arrives -> eligibility converts to weight change

Key difference from plastic.py:
  - NO weight change on target firing alone
  - Weight change ONLY when deliver_reward() is called
  - Positive reward -> potentiation (strengthen what was eligible)
  - Negative reward -> depression (weaken what was eligible)
  - Zero reward -> eligibility decays without effect

This is the biological 3-factor rule: pre × post × modulator.
Without reward, the brain learns nothing. With reward, it learns
what worked. With punishment, it learns what to avoid.

The learning rule lives HERE, not in the engine.
"""
import math

DEFAULTS = {
    'learning_rate': 0.01,
    'w_min': 0.0,
    'w_max': 10.0,
    'tau_trace': 20.0,       # Fast: source spike timing (ms)
    'tau_eligible': 500.0,   # Slow: eligibility window for reward (ms)
}

INITIAL_STATE = {
    'trace': 0.0,           # Fast trace from source firing
    'eligibility': 0.0,     # Slow eligibility from pre+post coincidence
}


def on_source_fired(syn):
    """Source neuron fired. Build fast trace and deliver current weight."""
    syn['trace'] += 1.0
    return syn['weight']


def on_target_fired(syn):
    """Target neuron fired. Convert trace to eligibility.

    NO weight change here — that only happens on reward.
    """
    trace = syn['trace']
    if trace > 0:
        syn['eligibility'] += trace
        # Trace consumed (timing window used up)
        syn['trace'] *= 0.5  # partial consumption, not full reset


def on_reward(syn, reward):
    """Reward signal arrived. Convert eligibility to weight change.

    Soft-bounded, sign-aware (same math as plastic.py but gated by reward).
    reward > 0: potentiation (strengthen eligible connections)
    reward < 0: depression (weaken eligible connections)
    reward = 0: no change (eligibility just decays away)
    """
    elig = syn['eligibility']
    if abs(elig) < 1e-6 or abs(reward) < 1e-6:
        return

    w = syn['weight']
    lr = syn['learning_rate']
    w_min = syn['w_min']
    w_max = syn['w_max']
    range_w = w_max - w_min

    if range_w < 1e-6:
        return

    if w_min < 0 and w_max <= 0:
        # Inhibitory: positive reward strengthens inhibition (more negative)
        dw = -lr * elig * reward * (w - w_min) / range_w
    else:
        # Excitatory: positive reward strengthens, negative weakens
        if reward > 0:
            dw = lr * elig * reward * (w_max - w) / range_w
        else:
            # Negative reward: weaken (push toward w_min)
            dw = lr * elig * reward * (w - w_min) / range_w

    syn['weight'] = max(w_min, min(w_max, w + dw))

    # Eligibility partially consumed by reward
    syn['eligibility'] *= 0.5


def per_tick(syn):
    """Trace and eligibility decay at different rates."""
    tau_trace = syn['tau_trace']
    tau_elig = syn['tau_eligible']

    if tau_trace > 0:
        syn['trace'] *= math.exp(-1.0 / tau_trace)
    if tau_elig > 0:
        syn['eligibility'] *= math.exp(-1.0 / tau_elig)
