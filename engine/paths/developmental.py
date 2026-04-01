"""
Developmental path -- STDP + activity-dependent pruning.

The blank slate synapse. Starts alive, learns via STDP,
dies if pre-post correlation stays too low during critical period.

Biology: Human brains overproduce synapses by 2x (peaking ~age 2),
then PRUNE through adolescence. The topology emerges from experience.
Animals get pre-wired circuits. Humans get a dense tangle that gets sculpted.

Mechanism: Fisher Information (FI) measures synapse importance using
only local pre/post correlations. FI = coincidence_rate * (1 - coincidence_rate).
Peaks at 0.5 (balanced correlation). Near 0 = never co-active = useless.

Phases:
  1. Critical period (first critical_period ticks):
     - STDP active (same as plastic.py)
     - FI tracking: running count of source fires + pre-post coincidences
     - Every eval_interval ticks: compute FI, prune if below threshold
  2. Post-critical (after critical_period ticks):
     - Surviving synapses behave as regular plastic
     - No more pruning -- topology is set

The learning rule lives HERE, not in the engine.
"""
import math

DEFAULTS = {
    'learning_rate': 0.01,
    'w_min': 0.0,
    'w_max': 10.0,
    'tau_plus': 20.0,
    'critical_period': 10000,     # ticks before pruning stops
    'pruning_threshold': 0.02,    # min coincidence rate to survive
    'eval_interval': 2000,        # evaluate FI every N ticks
    'min_source_fires': 20,       # need this many source fires before pruning
}

INITIAL_STATE = {
    'eligibility': 0.0,
    'source_fires': 0,
    'coincidences': 0,
    'alive': True,
}


def on_source_fired(syn):
    """Source neuron fired. Deliver weight, mark eligibility, count fire."""
    if not syn.get('alive', True):
        return 0.0
    syn['source_fires'] = syn.get('source_fires', 0) + 1
    syn['eligibility'] += 1.0
    return syn['weight']


def on_target_fired(syn):
    """Target neuron fired. Apply STDP if eligible. Count coincidence."""
    if not syn.get('alive', True):
        return

    elig = syn['eligibility']
    if elig > 0:
        # Count as coincidence (pre fired recently, now post fires)
        syn['coincidences'] = syn.get('coincidences', 0) + 1

        # Standard STDP (same as plastic.py)
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


def evaluate_fi(syn):
    """Compute Fisher Information proxy. Returns (fi, should_prune).

    Called periodically during critical period.
    """
    source_fires = syn.get('source_fires', 0)
    min_fires = syn.get('min_source_fires', 20)

    if source_fires < min_fires:
        return 0.0, False  # Not enough data, don't prune yet

    coincidences = syn.get('coincidences', 0)
    rate = coincidences / source_fires
    fi = rate * (1.0 - rate)  # Fisher information proxy

    threshold = syn.get('pruning_threshold', 0.02)
    # Prune if coincidence rate itself is too low (< threshold)
    # FI peaks at rate=0.5, but we care about rate being non-zero
    should_prune = rate < threshold

    return fi, should_prune


def per_tick(syn):
    """Eligibility decays exponentially."""
    if not syn.get('alive', True):
        return
    tau = syn['tau_plus']
    if tau > 0:
        syn['eligibility'] *= math.exp(-1.0 / tau)
