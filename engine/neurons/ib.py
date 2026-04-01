"""
IB — Intrinsically Bursting neuron.

Fires an initial burst of 2-4 rapid spikes, then switches to regular
spiking. Higher reset voltage (c=-55) keeps it closer to threshold after
first spike, enabling the burst. Commits once triggered — good for
sustained output, motor commands. Layer 5 pyramidal neurons.
"""

a = 0.02   # recovery time constant (slow)
b = 0.2    # recovery sensitivity
c = -55.0  # reset voltage (HIGH — stays near threshold = burst)
d = 4.0    # recovery kick (moderate)

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
