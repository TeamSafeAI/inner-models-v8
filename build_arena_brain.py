"""
build_arena_brain.py -- Build a ~7700N brain for arena chemotaxis.

Uses the arena_v1 recipe (proven 5-layer architecture + sensory_bank + reward_plastic).
After building, populates body_map and sensor_map tables so the arena
simulation can map neurons to body segments and sensory modalities.

Usage:
  py build_arena_brain.py
  py build_arena_brain.py --seed 42
"""
import os, sys, json, sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from brain_generator import generate


def populate_body_map(db_path, recipe):
    """Populate body_map table: motor neurons -> body segments.

    Maps forward_motor and backward_motor output neurons to the 24-segment
    worm body. Each motor instance's dorsal/ventral outputs get assigned
    to segments proportionally.

    forward_motor: 8N per instance, dorsal=[0,2,4,6], ventral=[1,3,5,7]
    backward_motor: 10N per instance, dorsal=[0,2,4,6,8], ventral=[1,3,5,7,9]
    """
    conn = sqlite3.connect(db_path)

    # Get all neuron IDs in order
    rows = conn.execute("SELECT id FROM neurons ORDER BY id").fetchall()
    all_ids = [r[0] for r in rows]

    # Figure out component ordering and neuron ID ranges
    SIZES = {
        'sensory_bank': 3, 'mechanosensory': 8, 'binary_decision': 12,
        'forward_motor': 8, 'backward_motor': 10,
        'working_memory_cell': 7, 'emotional_state': 6,
        'state_machine_3': 15, 'winner_take_all': 6,
        'cross_inhibition': 12, 'head_steering': 10,
        'sequence_detector': 6, 'novelty_detector': 3,
        'lateral_inhibition': 14, 'gain_controller': 3,
        'bistable_motif': 5, 'rebound_timer': 3,
        'activity_brake': 6, 'cpg_oscillator': 8,
        'burst_rebound': 10, 'modulatory': 8,
    }

    components = recipe['components']
    n_body_segments = 24

    # Walk through components to find motor neuron ID ranges
    next_id = 1
    fwd_motor_ranges = []  # list of (base_id, count)
    bwd_motor_ranges = []

    for comp in components:
        comp_type = comp['type']
        count = comp['count']
        size = SIZES.get(comp_type, 0)
        if size == 0:
            print(f"  WARNING: Unknown component type '{comp_type}', skipping")
            continue

        if comp_type == 'forward_motor':
            for i in range(count):
                fwd_motor_ranges.append(next_id + i * size)
        elif comp_type == 'backward_motor':
            for i in range(count):
                bwd_motor_ranges.append(next_id + i * size)

        next_id += count * size

    # Map forward_motor instances to segments
    entries = []
    n_fwd = len(fwd_motor_ranges)
    for inst_idx, base_id in enumerate(fwd_motor_ranges):
        # Distribute instances across segments
        # Each instance has 4 dorsal + 4 ventral outputs
        segs_per_inst = max(1, n_body_segments // max(1, n_fwd))
        start_seg = (inst_idx * n_body_segments) // max(1, n_fwd)

        dorsal_locals = [0, 2, 4, 6]
        ventral_locals = [1, 3, 5, 7]

        for j, local in enumerate(dorsal_locals):
            seg = min(start_seg + j, n_body_segments - 1)
            entries.append((base_id + local, seg, 'dorsal', 'excitatory'))
        for j, local in enumerate(ventral_locals):
            seg = min(start_seg + j, n_body_segments - 1)
            entries.append((base_id + local, seg, 'ventral', 'excitatory'))

    # Map backward_motor instances to segments
    n_bwd = len(bwd_motor_ranges)
    for inst_idx, base_id in enumerate(bwd_motor_ranges):
        start_seg = (inst_idx * n_body_segments) // max(1, n_bwd)

        dorsal_locals = [0, 2, 4, 6, 8]
        ventral_locals = [1, 3, 5, 7, 9]

        for j, local in enumerate(dorsal_locals):
            seg = min(start_seg + j, n_body_segments - 1)
            entries.append((base_id + local, seg, 'dorsal', 'excitatory'))
        for j, local in enumerate(ventral_locals):
            seg = min(start_seg + j, n_body_segments - 1)
            entries.append((base_id + local, seg, 'ventral', 'excitatory'))

    # Write to DB
    for nid, seg, side, effect in entries:
        conn.execute("INSERT INTO body_map (neuron_id, segment, side, effect) VALUES (?,?,?,?)",
                     (nid, seg, side, effect))

    conn.commit()
    print(f"  body_map: {len(entries)} entries ({n_fwd} fwd + {n_bwd} bwd motor instances)")

    return conn


def populate_sensor_map(conn, recipe):
    """Populate sensor_map table: sensory neurons -> modalities.

    sensory_bank: channel_in [0,1] = chemical sensors
    mechanosensory: anterior_in [0,1,2] = mechanical head, posterior_in [3,4,5] = mechanical tail
    """
    SIZES = {
        'sensory_bank': 3, 'mechanosensory': 8, 'binary_decision': 12,
        'forward_motor': 8, 'backward_motor': 10,
        'working_memory_cell': 7, 'emotional_state': 6,
        'state_machine_3': 15, 'winner_take_all': 6,
        'cross_inhibition': 12, 'head_steering': 10,
        'sequence_detector': 6, 'novelty_detector': 3,
        'lateral_inhibition': 14, 'gain_controller': 3,
        'bistable_motif': 5, 'rebound_timer': 3,
        'activity_brake': 6, 'cpg_oscillator': 8,
        'burst_rebound': 10, 'modulatory': 8,
    }

    components = recipe['components']
    entries = []
    next_id = 1

    for comp in components:
        comp_type = comp['type']
        count = comp['count']
        size = SIZES.get(comp_type, 0)
        if size == 0:
            next_id += count * size
            continue

        if comp_type == 'sensory_bank':
            for inst in range(count):
                base_id = next_id + inst * size
                # channel_in [0,1] = chemical concentration sensors
                for local in [0, 1]:
                    # Distribute left/right across instances
                    side = 'left' if inst % 2 == 0 else 'right'
                    entries.append((base_id + local, 'chemical', 'head', 'tonic', side))

        elif comp_type == 'mechanosensory':
            for inst in range(count):
                base_id = next_id + inst * size
                # anterior_in [0,1,2] = head touch/wall
                for local in [0, 1, 2]:
                    side = 'left' if inst % 2 == 0 else 'right'
                    entries.append((base_id + local, 'mechanical', 'head', 'tonic', side))
                # posterior_in [3,4,5] = tail touch
                for local in [3, 4, 5]:
                    side = 'left' if inst % 2 == 0 else 'right'
                    entries.append((base_id + local, 'mechanical', 'tail', 'tonic', side))

        next_id += count * size

    for nid, modality, location, rtype, side in entries:
        conn.execute(
            "INSERT INTO sensor_map (neuron_id, modality, location, response_type, side) "
            "VALUES (?,?,?,?,?)",
            (nid, modality, location, rtype, side))

    conn.commit()
    print(f"  sensor_map: {len(entries)} entries")
    conn.close()


def build(seed=42):
    recipe_path = os.path.join(BASE, 'recipes', 'arena_v1.json')
    with open(recipe_path) as f:
        recipe = json.load(f)

    recipe['seed'] = seed
    recipe['output'] = f'brains/zoo/arena_v1_s{seed}.db'

    db_path = generate(recipe)

    # Populate body/sensor maps
    conn = populate_body_map(db_path, recipe)
    populate_sensor_map(conn, recipe)

    print(f"\n  Arena brain: {db_path}")
    return db_path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    build(args.seed)
