"""
Neuron type registry.

Each neuron type is a module with:
  - a, b, c, d: Izhikevich parameters
  - SPIKE_THRESHOLD: voltage that triggers a spike
  - update(v, u, I) -> (v, u): voltage/recovery update
  - on_fire(v, u) -> (v, u): reset after spike
"""
from engine.neurons import rs, fs, ib, ch, lts

TYPES = {
    'RS':  rs,
    'FS':  fs,
    'IB':  ib,
    'CH':  ch,
    'LTS': lts,
}


def get(type_name):
    """Get neuron module by type name."""
    if type_name not in TYPES:
        raise ValueError(f"Unknown neuron type '{type_name}'. Must be one of: {list(TYPES.keys())}")
    return TYPES[type_name]
