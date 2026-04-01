"""
FS — Fast Spiking neuron.

Sharp inhibition. Fires rapidly with minimal adaptation. Recovery is 5x
faster than other types (a=0.1). Low recovery kick (d=2) means it can
sustain high firing rates. ~15% of cortex. Parvalbumin interneurons.
"""

a = 0.1    # recovery time constant (FAST — 5x other types)
b = 0.2    # recovery sensitivity
c = -65.0  # reset voltage after spike
d = 2.0    # recovery kick after spike (small = can fire again quickly)

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
