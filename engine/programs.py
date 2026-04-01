"""
programs.py — Stimulus delivery.

Load a JSON program, resolve neuron targets, deliver current per tick.
This is how you give the brain input.
"""
import os, json


def load(path):
    """Load a program from JSON file."""
    with open(path) as f:
        prog = json.load(f)
    print(f"  Program: {prog.get('name', os.path.basename(path))}")
    if prog.get('description'):
        print(f"  {prog['description']}")
    print(f"  {len(prog['phases'])} phases, loop={prog.get('loop', False)}")
    return prog


def resolve_targets(program, brain):
    """Map program target specs to neuron indices."""
    targets = program.get('targets', [])
    if not targets:
        return []

    id_to_idx = brain['id_to_idx']

    if isinstance(targets[0], int):
        # Direct neuron DB IDs
        return [id_to_idx[t] for t in targets if t in id_to_idx]
    else:
        # Names or group specs — would need name resolution
        # For now, return empty and let caller handle
        return []


def get_current(program, tick, n_neurons, targets, rng=None):
    """Get external current array for this tick.

    Returns list of length n_neurons with current to inject.
    """
    I = [0.0] * n_neurons

    if not targets:
        return I

    phases = program['phases']
    loop = program.get('loop', False)

    # Find active phase
    if 'tick_start' in phases[0]:
        # Absolute tick ranges
        for phase in phases:
            ts = phase.get('tick_start', 0)
            te = phase.get('tick_end', ts + 1000)
            if ts <= tick < te:
                current = phase.get('current', 0.0)
                noise = phase.get('noise', 0.0)
                for t in targets:
                    I[t] = current
                    if noise > 0 and rng is not None:
                        I[t] += rng.randn() * noise
                break
    else:
        # Duration-based phases
        total_dur = sum(p.get('duration', 1000) for p in phases)
        if loop:
            tick = tick % total_dur
        elapsed = 0
        for phase in phases:
            dur = phase.get('duration', 1000)
            if elapsed <= tick < elapsed + dur:
                current = phase.get('current', 0.0)
                noise = phase.get('noise', 0.0)
                for t in targets:
                    I[t] = current
                    if noise > 0 and rng is not None:
                        I[t] += rng.randn() * noise
                break
            elapsed += dur

    return I
