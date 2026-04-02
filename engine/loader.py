"""
loader.py — Load a brain from its .db file.

Reads neurons and synapses from the database, looks up their type
modules, and returns a brain dict ready for the runner.
"""
import os, sys, json, sqlite3

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from engine import neurons as neuron_registry
from engine import paths as path_registry


def load(db_path):
    """Load a brain database into memory.

    Returns a dict with:
        neurons: list of neuron dicts (id, type_module, v, u, ...)
        synapses: list of synapse dicts (id, source, target, type_module, weight, params, state, ...)
        gap_junctions: list of gap junction dicts (separated because they're continuous, not spike-based)
        id_to_idx: mapping from DB neuron ID to array index
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Brain database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Load neurons
    neuron_list = []
    for row in conn.execute("SELECT * FROM neurons ORDER BY id"):
        type_name = row['neuron_type']
        type_module = neuron_registry.get(type_name)

        neuron_list.append({
            'id': row['id'],
            'type': type_name,
            'module': type_module,
            'a': row['a'],
            'b': row['b'],
            'c': row['c'],
            'd': row['d'],
            'v': row['v'],
            'u': row['u'],
            'last_spike': row['last_spike'],
            'pos_x': row['pos_x'],
            'pos_y': row['pos_y'],
            'pos_z': row['pos_z'],
        })

    # Map DB IDs to array indices
    id_to_idx = {n['id']: i for i, n in enumerate(neuron_list)}

    # Load synapses
    chemical_synapses = []
    gap_junctions = []

    for row in conn.execute("SELECT * FROM synapses ORDER BY id"):
        type_name = row['synapse_type']
        type_module = path_registry.get(type_name)

        params = json.loads(row['params']) if row['params'] else {}
        state = json.loads(row['state']) if row['state'] else {}

        # Build synapse dict: merge defaults, params, and state
        defaults = dict(type_module.DEFAULTS)
        defaults.update(params)
        initial_state = dict(type_module.INITIAL_STATE)
        initial_state.update(state)

        syn = {
            'id': row['id'],
            'source': id_to_idx[row['source']],
            'target': id_to_idx[row['target']],
            'source_db_id': row['source'],
            'target_db_id': row['target'],
            'type': type_name,
            'module': type_module,
            'weight': row['weight'],
            'delay': row['delay'],
        }
        # Add all params and state as flat keys
        syn.update(defaults)
        syn.update(initial_state)

        if type_name == 'gap_junction':
            gap_junctions.append(syn)
        else:
            chemical_synapses.append(syn)

    # Load body_map (motor neuron -> body segment mapping)
    body_map = {}
    try:
        for row in conn.execute("SELECT * FROM body_map"):
            nid = row['neuron_id']
            if nid in id_to_idx:
                body_map[id_to_idx[nid]] = {
                    'segment': row['segment'],
                    'side': row['side'],
                    'effect': row['effect'],
                }
    except sqlite3.OperationalError:
        pass  # table doesn't exist

    # Load sensor_map (sensory neuron -> modality mapping)
    sensor_map = {}
    try:
        for row in conn.execute("SELECT * FROM sensor_map"):
            nid = row['neuron_id']
            if nid in id_to_idx:
                sensor_map[id_to_idx[nid]] = {
                    'modality': row['modality'],
                    'location': row['location'],
                    'response_type': row['response_type'],
                    'side': row['side'],
                }
    except sqlite3.OperationalError:
        pass  # table doesn't exist

    conn.close()

    # Build source/target lookup indices
    syn_by_source = {}
    syn_by_target = {}
    for i, syn in enumerate(chemical_synapses):
        syn_by_source.setdefault(syn['source'], []).append(i)
        syn_by_target.setdefault(syn['target'], []).append(i)

    return {
        'db_path': db_path,
        'neurons': neuron_list,
        'synapses': chemical_synapses,
        'gap_junctions': gap_junctions,
        'id_to_idx': id_to_idx,
        'syn_by_source': syn_by_source,
        'syn_by_target': syn_by_target,
        'body_map': body_map,
        'sensor_map': sensor_map,
    }


def save(brain):
    """Save current brain state back to the database."""
    conn = sqlite3.connect(brain['db_path'])

    for n in brain['neurons']:
        conn.execute(
            "UPDATE neurons SET v=?, u=?, last_spike=?, a=?, b=?, c=?, d=? WHERE id=?",
            (n['v'], n['u'], n['last_spike'], n['a'], n['b'], n['c'], n['d'], n['id'])
        )

    for syn in brain['synapses'] + brain['gap_junctions']:
        # Rebuild state dict from synapse
        module = syn['module']
        state = {}
        for key in module.INITIAL_STATE:
            if key in syn:
                state[key] = syn[key]

        conn.execute(
            "UPDATE synapses SET weight=?, state=? WHERE id=?",
            (syn['weight'], json.dumps(state), syn['id'])
        )

    conn.commit()
    conn.close()
