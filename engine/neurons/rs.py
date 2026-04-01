"""
RS — Regular Spiking neuron.

Workhorse excitatory neuron. Fires single spikes that adapt (slow down)
over sustained input. High recovery kick (d=8) spaces out spikes.
~70% of cortex. Pyramidal neurons.
"""

# Izhikevich parameters
a = 0.02   # recovery time constant (slow)
b = 0.2    # recovery sensitivity
c = -65.0  # reset voltage after spike
d = 8.0    # recovery kick after spike (large = more adaptation)

SPIKE_THRESHOLD = 30.0


def update(v, u, I, a_=None, b_=None):
    """Update voltage and recovery given input current.

    Two half-steps of the Izhikevich equation for numerical stability.
    """
    _a = a_ if a_ is not None else a
    _b = b_ if b_ is not None else b
    v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
    if v > 35.0:
        v = 35.0
    elif v < -100.0:
        v = -100.0
    v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
    u += _a * (_b * v - u)
    return v, u


def on_fire(v, u, c_=None, d_=None):
    """Reset after spike."""
    _c = c_ if c_ is not None else c
    _d = d_ if d_ is not None else d
    return _c, u + _d
