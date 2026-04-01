"""
Gap junction — bidirectional electrical coupling.

Not spike-based. Every tick, current flows between the two neurons
based on their voltage difference. Instantaneous, no delay.

I = conductance * (V_other - V_self) for each side.

Good for synchronization, oscillators, electrical synapses.
"""

DEFAULTS = {
    'conductance': 0.1,
}

INITIAL_STATE = {}


def continuous(syn, v_source, v_target):
    """Compute current for both sides. Returns (I_to_target, I_to_source).

    Current flows from higher voltage to lower voltage.
    """
    g = syn['conductance']
    dv = v_source - v_target
    return g * dv, g * (-dv)


def on_source_fired(syn):
    """Gap junctions don't use spike delivery."""
    return 0.0


def on_target_fired(syn):
    """Gap junctions don't learn."""
    pass


def per_tick(syn):
    """Nothing to decay."""
    pass
