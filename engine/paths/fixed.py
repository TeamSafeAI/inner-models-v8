"""
Fixed path — static weight, never changes.

Signal goes in, comes out at the same weight every time.
No learning, no adaptation, no state. Just a wire.
"""

DEFAULTS = {}
INITIAL_STATE = {}


def on_source_fired(syn):
    """Source neuron fired. Return effective weight to deliver."""
    return syn['weight']


def on_target_fired(syn):
    """Target neuron fired. Fixed paths don't learn."""
    pass


def per_tick(syn):
    """Nothing to decay."""
    pass
