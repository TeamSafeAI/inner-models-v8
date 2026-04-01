"""
LTS — Low-Threshold Spiking neuron.

Delayed onset, rebound bursts. Higher recovery sensitivity (b=0.25) makes
it more responsive to voltage changes. Key behavior: fires AFTER inhibition
releases (rebound). Good for delayed inhibition, gating, oscillation.
Somatostatin interneurons, thalamic reticular nucleus.
"""

a = 0.02   # recovery time constant (slow)
b = 0.25   # recovery sensitivity (HIGHER — more responsive to voltage)
c = -65.0  # reset voltage
d = 2.0    # recovery kick (small)

SPIKE_THRESHOLD = 30.0


def update(v, u, I, a_=None, b_=None):
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
    _c = c_ if c_ is not None else c
    _d = d_ if d_ is not None else d
    return _c, u + _d
