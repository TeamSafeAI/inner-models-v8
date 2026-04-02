"""
schema.py — V8 brain database schema with typed synapses.

V7 had flat synapses (source, target, weight, delay). V8 adds:
  - synapse_type: fixed, plastic, facilitating, depressing, gated
  - params: JSON blob for type-specific intrinsic properties
  - state: JSON blob for runtime state that persists across saves

V8 also stores neuron_type as a readable string (RS, FS, IB, CH, LTS)
instead of V7's integer type codes.

Usage:
  from schema import create_brain_db, add_neuron, add_synapse, load_brain, save_state
"""

import os
import json
import sqlite3


# ── Izhikevich neuron type parameters (a, b, c, d) ──
# From Izhikevich 2003

NEURON_TYPES = {
    'RS':  (0.02, 0.2,  -65, 8),   # Regular Spiking — standard excitatory pyramidal
    'FS':  (0.1,  0.2,  -65, 2),   # Fast Spiking — standard inhibitory interneuron
    'IB':  (0.02, 0.2,  -55, 4),   # Intrinsically Bursting — motor output, sustainers
    'CH':  (0.02, 0.2,  -50, 2),   # Chattering — oscillators, modulators
    'LTS': (0.02, 0.25, -65, 2),   # Low-Threshold Spiking — suppressor inhibitory
}

VALID_NEURON_TYPES = set(NEURON_TYPES.keys())


# ── Synapse type default intrinsic parameters ──

SYNAPSE_DEFAULTS = {
    'fixed': {},
    'plastic': {
        'learning_rate': 0.01,
        'w_min': 0.0,
        'w_max': 10.0,
        'tau_plus': 20.0,
        'tau_minus': 20.0,
        'ltd_ratio': 0.5,
    },
    'facilitating': {
        'tau_facil': 50.0,
        'tau_recovery': 200.0,
        'facil_increment': 0.2,
    },
    'depressing': {
        'tau_depress': 50.0,
        'tau_recovery': 500.0,
        'depress_factor': 0.5,
    },
    'gated': {
        'learning_rate': 0.01,
        'w_min': 0.0,
        'w_max': 10.0,
        'gate_threshold': 0.3,
        'modulator_group': '',
    },
    'gap_junction': {
        'conductance': 0.1,  # I = g * (V_pre - V_post), bidirectional
    },
    'reward_plastic': {
        'learning_rate': 0.01,
        'w_min': 0.0,
        'w_max': 10.0,
        'tau_trace': 20.0,       # Fast: source spike timing (ms)
        'tau_eligible': 500.0,   # Slow: eligibility window for reward (ms)
    },
    'developmental': {
        'learning_rate': 0.01,
        'w_min': 0.0,
        'w_max': 10.0,
        'tau_plus': 20.0,
        'critical_period': 10000,     # ticks before pruning stops
        'pruning_threshold': 0.02,    # min coincidence rate to survive
        'eval_interval': 2000,        # evaluate FI every N ticks
        'min_source_fires': 20,       # need enough data before pruning
    },
}

# Default runtime state per synapse type
SYNAPSE_INITIAL_STATE = {
    'fixed': {},
    'plastic': {'eligibility': 0.0, 'elig_post': 0.0},
    'facilitating': {'current_gain': 1.0},
    'depressing': {'current_gain': 1.0},
    'gated': {'eligibility': 0.0},
    'gap_junction': {},
    'reward_plastic': {'trace': 0.0, 'eligibility': 0.0},
    'developmental': {'eligibility': 0.0, 'source_fires': 0, 'coincidences': 0, 'alive': True},
}

VALID_SYNAPSE_TYPES = set(SYNAPSE_DEFAULTS.keys())


# ── Schema SQL ──

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS neurons (
    id INTEGER PRIMARY KEY,
    neuron_type TEXT NOT NULL,
    a REAL NOT NULL,
    b REAL NOT NULL,
    c REAL NOT NULL,
    d REAL NOT NULL,
    v REAL NOT NULL DEFAULT -65.0,
    u REAL NOT NULL,
    last_spike INTEGER NOT NULL DEFAULT -1000,
    pos_x REAL NOT NULL DEFAULT 0.0,
    pos_y REAL NOT NULL DEFAULT 0.0,
    pos_z REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS synapses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source INTEGER NOT NULL,
    target INTEGER NOT NULL,
    weight REAL NOT NULL,
    delay INTEGER NOT NULL,
    synapse_type TEXT NOT NULL DEFAULT 'fixed',
    params TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (source) REFERENCES neurons(id),
    FOREIGN KEY (target) REFERENCES neurons(id)
);

CREATE INDEX IF NOT EXISTS idx_syn_source ON synapses(source);
CREATE INDEX IF NOT EXISTS idx_syn_target ON synapses(target);
CREATE INDEX IF NOT EXISTS idx_syn_type ON synapses(synapse_type);

CREATE TABLE IF NOT EXISTS body_map (
    neuron_id INTEGER NOT NULL,
    segment INTEGER NOT NULL,
    side TEXT NOT NULL,
    effect TEXT NOT NULL,
    FOREIGN KEY (neuron_id) REFERENCES neurons(id)
);
CREATE INDEX IF NOT EXISTS idx_body_map_seg ON body_map(segment);
CREATE INDEX IF NOT EXISTS idx_body_map_neuron ON body_map(neuron_id);

CREATE TABLE IF NOT EXISTS sensor_map (
    neuron_id INTEGER NOT NULL,
    modality TEXT NOT NULL,
    location TEXT NOT NULL,
    response_type TEXT NOT NULL DEFAULT 'tonic',
    side TEXT NOT NULL DEFAULT 'bilateral',
    FOREIGN KEY (neuron_id) REFERENCES neurons(id)
);
CREATE INDEX IF NOT EXISTS idx_sensor_map_modality ON sensor_map(modality);
CREATE INDEX IF NOT EXISTS idx_sensor_map_neuron ON sensor_map(neuron_id);
"""


# ── Core functions ──

def create_brain_db(path):
    """Create an empty brain database with V8 schema. Returns the connection."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def add_neuron(conn, neuron_type, pos_x=0.0, pos_y=0.0, pos_z=0.0, neuron_id=None):
    """Insert a neuron with Izhikevich params looked up from its type.

    Args:
        conn: sqlite3 connection
        neuron_type: one of RS, FS, IB, CH, LTS
        pos_x, pos_y, pos_z: spatial position
        neuron_id: explicit ID (optional, auto-assigned if None)

    Returns:
        The neuron's row ID.
    """
    if neuron_type not in VALID_NEURON_TYPES:
        raise ValueError(f"Unknown neuron_type '{neuron_type}'. Must be one of: {sorted(VALID_NEURON_TYPES)}")

    a, b, c, d = NEURON_TYPES[neuron_type]
    v = -65.0
    u = b * v  # standard Izhikevich initial condition

    if neuron_id is not None:
        conn.execute(
            "INSERT INTO neurons (id, neuron_type, a, b, c, d, v, u, pos_x, pos_y, pos_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (neuron_id, neuron_type, a, b, c, d, v, u, pos_x, pos_y, pos_z)
        )
        return neuron_id
    else:
        cur = conn.execute(
            "INSERT INTO neurons (neuron_type, a, b, c, d, v, u, pos_x, pos_y, pos_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (neuron_type, a, b, c, d, v, u, pos_x, pos_y, pos_z)
        )
        return cur.lastrowid


def add_synapse(conn, source, target, weight, delay, synapse_type='fixed', params_override=None):
    """Insert a synapse with type-specific params and initial state.

    Args:
        conn: sqlite3 connection
        source: source neuron ID
        target: target neuron ID
        weight: synaptic weight
        delay: axonal delay in ticks
        synapse_type: one of fixed, plastic, facilitating, depressing, gated
        params_override: dict to merge over the type's default params (optional)

    Returns:
        The synapse's row ID.
    """
    if synapse_type not in VALID_SYNAPSE_TYPES:
        raise ValueError(f"Unknown synapse_type '{synapse_type}'. Must be one of: {sorted(VALID_SYNAPSE_TYPES)}")

    # Build params: start from defaults, overlay any overrides
    params = dict(SYNAPSE_DEFAULTS[synapse_type])
    if params_override:
        params.update(params_override)

    # Initial runtime state
    state = dict(SYNAPSE_INITIAL_STATE[synapse_type])

    cur = conn.execute(
        "INSERT INTO synapses (source, target, weight, delay, synapse_type, params, state) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source, target, weight, delay, synapse_type, json.dumps(params), json.dumps(state))
    )
    return cur.lastrowid


def load_brain(path):
    """Load a brain database, returning parsed neurons and synapses.

    Returns:
        (neurons, synapses) where:
          neurons = list of dicts with keys: id, neuron_type, a, b, c, d, v, u, last_spike, pos_x, pos_y, pos_z
          synapses = list of dicts with keys: id, source, target, weight, delay, synapse_type, params (dict), state (dict)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brain database not found: {path}")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Load neurons
    neurons = []
    for row in conn.execute("SELECT * FROM neurons ORDER BY id"):
        neurons.append({
            'id': row['id'],
            'neuron_type': row['neuron_type'],
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

    # Load synapses with parsed JSON
    synapses = []
    for row in conn.execute("SELECT * FROM synapses ORDER BY id"):
        synapses.append({
            'id': row['id'],
            'source': row['source'],
            'target': row['target'],
            'weight': row['weight'],
            'delay': row['delay'],
            'synapse_type': row['synapse_type'],
            'params': json.loads(row['params']),
            'state': json.loads(row['state']),
        })

    conn.close()
    return neurons, synapses


def save_state(conn, neuron_states, synapse_states=None):
    """Save runtime state back to the database.

    Args:
        conn: sqlite3 connection
        neuron_states: list of (id, v, u, last_spike) tuples
        synapse_states: list of (id, state_dict) tuples (optional)
            Only needed for synapses with runtime state (facilitating, depressing, plastic, gated).
    """
    if neuron_states:
        conn.executemany(
            "UPDATE neurons SET v=?, u=?, last_spike=? WHERE id=?",
            [(v, u, ls, nid) for nid, v, u, ls in neuron_states]
        )

    if synapse_states:
        for sid, state_dict in synapse_states:
            conn.execute(
                "UPDATE synapses SET state=? WHERE id=?",
                (json.dumps(state_dict), sid)
            )

    conn.commit()


# ── Body map functions ──

def add_body_map_entry(conn, neuron_id, segment, side, effect):
    """Map a motor neuron to a body segment.

    Args:
        conn: sqlite3 connection
        neuron_id: neuron row ID
        segment: body segment index (0-23)
        side: 'dorsal' or 'ventral'
        effect: 'excitatory' or 'inhibitory'
    """
    conn.execute(
        "INSERT INTO body_map (neuron_id, segment, side, effect) VALUES (?, ?, ?, ?)",
        (neuron_id, segment, side, effect)
    )


def load_body_map(conn):
    """Load body map from database.

    Returns:
        dict: {neuron_id: {'segment': int, 'side': str, 'effect': str}}
    """
    conn.row_factory = sqlite3.Row
    body_map = {}
    for row in conn.execute("SELECT * FROM body_map"):
        body_map[row['neuron_id']] = {
            'segment': row['segment'],
            'side': row['side'],
            'effect': row['effect'],
        }
    return body_map


# ── Sensor map functions ──

def add_sensor_entry(conn, neuron_id, modality, location, response_type='tonic',
                     side='bilateral'):
    """Map a sensory neuron to a modality and body location.

    Args:
        conn: sqlite3 connection
        neuron_id: neuron row ID
        modality: 'chemical', 'mechanical', 'thermal'
        location: 'head', 'body', 'tail'
        response_type: 'ON' (responds to increases), 'OFF' (responds to
            decreases), or 'tonic' (responds to absolute level)
        side: 'left', 'right', or 'bilateral'
    """
    conn.execute(
        "INSERT INTO sensor_map (neuron_id, modality, location, response_type, side) "
        "VALUES (?, ?, ?, ?, ?)",
        (neuron_id, modality, location, response_type, side)
    )


def load_sensor_map(conn):
    """Load sensor map from database.

    Returns:
        dict: {neuron_id: {'modality': str, 'location': str}}
    """
    conn.row_factory = sqlite3.Row
    sensor_map = {}
    try:
        for row in conn.execute("SELECT * FROM sensor_map"):
            keys = row.keys()
            sensor_map[row['neuron_id']] = {
                'modality': row['modality'],
                'location': row['location'],
                'response_type': row['response_type'] if 'response_type' in keys else 'tonic',
                'side': row['side'] if 'side' in keys else 'bilateral',
            }
    except sqlite3.OperationalError:
        pass  # table doesn't exist in older DBs
    return sensor_map


# ── Test / demo ──

if __name__ == '__main__':
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brains', '_schema_test.db')

    # Clean up any previous test
    if os.path.exists(test_path):
        os.remove(test_path)

    print(f"Creating test brain at: {test_path}")
    conn = create_brain_db(test_path)

    # Add one neuron of each type
    neuron_ids = {}
    for i, ntype in enumerate(sorted(NEURON_TYPES.keys())):
        nid = add_neuron(conn, ntype, pos_x=float(i), pos_y=0.0, pos_z=0.0)
        neuron_ids[ntype] = nid
        a, b, c, d = NEURON_TYPES[ntype]
        print(f"  Neuron {nid}: {ntype} (a={a}, b={b}, c={c}, d={d})")

    print()

    # Add one synapse of each type between sequential neurons
    nids = list(neuron_ids.values())
    synapse_ids = {}
    for i, stype in enumerate(sorted(SYNAPSE_DEFAULTS.keys())):
        src = nids[i % len(nids)]
        tgt = nids[(i + 1) % len(nids)]
        sid = add_synapse(conn, src, tgt, weight=1.5, delay=3, synapse_type=stype)
        synapse_ids[stype] = sid
        print(f"  Synapse {sid}: {stype} ({src}->{tgt}, w=1.5, d=3)")
        defaults = SYNAPSE_DEFAULTS[stype]
        if defaults:
            print(f"    params: {defaults}")
        init_state = SYNAPSE_INITIAL_STATE[stype]
        if init_state:
            print(f"    state:  {init_state}")

    conn.commit()
    print()

    # Reload and verify
    neurons, synapses = load_brain(test_path)
    print(f"Loaded: {len(neurons)} neurons, {len(synapses)} synapses")

    # Verify round-trip of params and state
    for s in synapses:
        expected_params = SYNAPSE_DEFAULTS[s['synapse_type']]
        expected_state = SYNAPSE_INITIAL_STATE[s['synapse_type']]
        assert s['params'] == expected_params, f"Params mismatch for synapse {s['id']}"
        assert s['state'] == expected_state, f"State mismatch for synapse {s['id']}"

    print("Round-trip verification: PASSED")

    # Test save_state
    test_neuron_states = [(n['id'], -60.0, -12.0, 500) for n in neurons]
    test_synapse_states = []
    for s in synapses:
        if s['synapse_type'] == 'facilitating':
            test_synapse_states.append((s['id'], {'current_gain': 1.8}))
        elif s['synapse_type'] == 'depressing':
            test_synapse_states.append((s['id'], {'current_gain': 0.3}))
        elif s['synapse_type'] in ('plastic', 'gated'):
            test_synapse_states.append((s['id'], {'eligibility': 0.75}))

    conn2 = sqlite3.connect(test_path)
    save_state(conn2, test_neuron_states, test_synapse_states)
    conn2.close()

    # Reload and verify state was saved
    neurons2, synapses2 = load_brain(test_path)
    for n in neurons2:
        assert n['v'] == -60.0, f"Neuron {n['id']} v not saved"
        assert n['u'] == -12.0, f"Neuron {n['id']} u not saved"
        assert n['last_spike'] == 500, f"Neuron {n['id']} last_spike not saved"

    for s in synapses2:
        if s['synapse_type'] == 'facilitating':
            assert s['state']['current_gain'] == 1.8
        elif s['synapse_type'] == 'depressing':
            assert s['state']['current_gain'] == 0.3
        elif s['synapse_type'] in ('plastic', 'gated'):
            assert s['state']['eligibility'] == 0.75

    print("State save/load verification: PASSED")

    # Clean up
    conn.close()
    os.remove(test_path)
    print(f"\nTest database cleaned up. Schema is ready.")
