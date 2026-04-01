"""
Path type registry.

Each path type is a module with:
  - DEFAULTS: default parameters
  - INITIAL_STATE: initial runtime state
  - on_source_fired(syn) -> effective_weight
  - on_target_fired(syn): learning rule (if any)
  - per_tick(syn): decay/recovery
  - For gap_junction: continuous(syn, v_src, v_tgt) -> (I_tgt, I_src)
"""
from engine.paths import fixed, plastic, facilitating, depressing, gated, gap_junction, reward_plastic, developmental

TYPES = {
    'fixed':            fixed,
    'plastic':          plastic,
    'facilitating':     facilitating,
    'depressing':       depressing,
    'gated':            gated,
    'gap_junction':     gap_junction,
    'reward_plastic':   reward_plastic,
    'developmental':    developmental,
}


def get(type_name):
    """Get path module by type name."""
    if type_name not in TYPES:
        raise ValueError(f"Unknown path type '{type_name}'. Must be one of: {list(TYPES.keys())}")
    return TYPES[type_name]
