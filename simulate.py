"""
simulate.py — V8 Izhikevich simulation with per-synapse behavior dispatch.

The brain is self-contained: all behavior is in the data. This engine reads
neuron_type and synapse_type from the DB and dispatches accordingly.
No external plasticity modules — those behaviors are intrinsic to synapse types.

Usage:
  py simulate.py brain.db
  py simulate.py brain.db --ticks 50000 --learn --save
  py simulate.py brain.db --program programs/pulse_train.json --learn --save
  py simulate.py brain.db --quiet
"""
import os, sys, json, math, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from schema import load_brain as _load_brain_raw, save_state as _save_state_raw
from schema import load_body_map as _load_body_map_raw
from schema import load_sensor_map as _load_sensor_map_raw
from schema import NEURON_TYPES, SYNAPSE_DEFAULTS
import sqlite3

# Synapse type string constants
SYN_FIXED = 'fixed'
SYN_PLASTIC = 'plastic'
SYN_FACILITATING = 'facilitating'
SYN_DEPRESSING = 'depressing'
SYN_GATED = 'gated'
SYN_GAP_JUNCTION = 'gap_junction'

DEFAULTS = {
    'plastic':      {'tau_plus': 20.0, 'learning_rate': 0.01, 'w_min': 0.0, 'w_max': 10.0},
    'facilitating': {'facil_increment': 0.3, 'tau_recovery': 200.0},
    'depressing':   {'depress_factor': 0.5, 'tau_recovery': 200.0},
    'gated':        {'tau_plus': 20.0, 'learning_rate': 0.01, 'w_min': 0.0, 'w_max': 10.0,
                     'gate_threshold': 0.05, 'modulator_group': []},
    'gap_junction': {'conductance': 0.1},
}

quiet_global = False

def _prop(props, key, type_name, fallback=None):
    if props and key in props: return props[key]
    d = DEFAULTS.get(type_name, {})
    return d.get(key, fallback)


def load_brain(path):
    """Load brain from schema and convert to numpy arrays for simulation."""
    neurons, synapses = _load_brain_raw(path)
    n = len(neurons)

    brain = {
        'n': n,
        'db_path': path,
        'v': np.array([nr['v'] for nr in neurons]),
        'u': np.array([nr['u'] for nr in neurons]),
        'a': np.array([nr['a'] for nr in neurons]),
        'b': np.array([nr['b'] for nr in neurons]),
        'c': np.array([nr['c'] for nr in neurons]),
        'd': np.array([nr['d'] for nr in neurons]),
        'last_spike': np.array([nr['last_spike'] for nr in neurons]),
        'neuron_ids': [nr['id'] for nr in neurons],
        'neuron_types': [nr['neuron_type'] for nr in neurons],
    }

    ns = len(synapses)
    brain['syn_src'] = np.array([s['source'] for s in synapses], dtype=int) if ns else np.array([], dtype=int)
    brain['syn_tgt'] = np.array([s['target'] for s in synapses], dtype=int) if ns else np.array([], dtype=int)
    brain['syn_w'] = np.array([s['weight'] for s in synapses]) if ns else np.array([])
    brain['syn_delay'] = np.array([s['delay'] for s in synapses], dtype=int) if ns else np.array([], dtype=int)

    # Synapse types as string array for dispatch
    brain['syn_type'] = np.array([s['synapse_type'] for s in synapses]) if ns else np.array([], dtype='U20')

    # Merge params and state for each synapse (state overrides params for runtime values)
    syn_props = []
    for s in synapses:
        merged = dict(s.get('params', {}))
        merged.update(s.get('state', {}))
        syn_props.append(merged)
    brain['syn_props'] = syn_props

    # Synapse DB IDs for saving back
    brain['syn_ids'] = [s['id'] for s in synapses]

    # Map neuron IDs to array indices (needed if DB IDs aren't 0-based contiguous)
    id_to_idx = {neurons[i]['id']: i for i in range(n)}
    if ns:
        brain['syn_src'] = np.array([id_to_idx[s['source']] for s in synapses], dtype=int)
        brain['syn_tgt'] = np.array([id_to_idx[s['target']] for s in synapses], dtype=int)
    brain['id_to_idx'] = id_to_idx

    # Separate gap junctions — they use continuous voltage coupling, not spike-based delivery
    gap_mask = brain['syn_type'] == SYN_GAP_JUNCTION if ns else np.array([], dtype=bool)
    n_gap = int(np.sum(gap_mask)) if ns else 0
    if n_gap > 0:
        brain['gap_src'] = brain['syn_src'][gap_mask]
        brain['gap_tgt'] = brain['syn_tgt'][gap_mask]
        brain['gap_g'] = np.array([
            (syn_props[i].get('conductance', 0.1) if syn_props[i] else 0.1)
            for i in range(ns) if gap_mask[i]
        ])
        # Remove gap junctions from chemical synapse arrays
        chem_mask = ~gap_mask
        brain['syn_src'] = brain['syn_src'][chem_mask]
        brain['syn_tgt'] = brain['syn_tgt'][chem_mask]
        brain['syn_w'] = brain['syn_w'][chem_mask]
        brain['syn_delay'] = brain['syn_delay'][chem_mask]
        brain['syn_type'] = brain['syn_type'][chem_mask]
        brain['syn_props'] = [p for p, m in zip(syn_props, gap_mask) if not m]
        brain['syn_ids'] = [sid for sid, m in zip(brain['syn_ids'], gap_mask) if not m]
    else:
        brain['gap_src'] = np.array([], dtype=int)
        brain['gap_tgt'] = np.array([], dtype=int)
        brain['gap_g'] = np.array([])
    brain['n_gap'] = n_gap

    # Load body map (motor neuron → body segment wiring)
    conn = sqlite3.connect(path)
    raw_body_map = _load_body_map_raw(conn)
    # Re-key by array index instead of DB neuron ID
    brain['body_map'] = {}
    for nid, entry in raw_body_map.items():
        if nid in id_to_idx:
            brain['body_map'][id_to_idx[nid]] = entry

    # Load sensor map (sensory neuron → modality + location)
    raw_sensor_map = _load_sensor_map_raw(conn)
    conn.close()
    brain['sensor_map'] = {}
    for nid, entry in raw_sensor_map.items():
        if nid in id_to_idx:
            brain['sensor_map'][id_to_idx[nid]] = entry

    return brain


def save_brain_state(brain, db_path):
    """Save neuron + synapse state back to DB."""
    conn = sqlite3.connect(db_path)
    n_ids = brain['neuron_ids']
    neuron_states = [(n_ids[i], float(brain['v'][i]), float(brain['u'][i]),
                      int(brain['last_spike'][i])) for i in range(brain['n'])]

    synapse_states = []
    for i, sid in enumerate(brain['syn_ids']):
        sp = brain['syn_props'][i]
        state = {}
        st = brain['syn_type'][i]
        if st in (SYN_PLASTIC, SYN_GATED):
            state['eligibility'] = sp.get('eligibility', 0.0)
        if st in (SYN_FACILITATING, SYN_DEPRESSING):
            state['current_gain'] = sp.get('current_gain', 1.0)
        synapse_states.append((sid, state))

    # Also update weights for plastic/gated synapses
    for i, sid in enumerate(brain['syn_ids']):
        conn.execute("UPDATE synapses SET weight=? WHERE id=?",
                     (float(brain['syn_w'][i]), sid))

    _save_state_raw(conn, neuron_states, synapse_states)
    conn.close()


def simulate(brain, ticks, learn, program, quiet, rng):
    n = brain['n']
    v, u = brain['v'].copy(), brain['u'].copy()
    a, b, c, d = brain['a'], brain['b'], brain['c'], brain['d']
    last_spike = brain['last_spike'].copy()

    syn_src, syn_tgt = brain['syn_src'], brain['syn_tgt']
    syn_w, syn_delay = brain['syn_w'].copy(), brain['syn_delay']
    syn_type, syn_props = brain['syn_type'], brain['syn_props']
    n_syn = len(syn_src)

    # Per-neuron outgoing synapse index
    syn_by_src = {}
    for i in range(n_syn):
        syn_by_src.setdefault(syn_src[i], []).append(i)

    # Delay buffer
    max_delay = int(np.max(syn_delay)) if n_syn > 0 else 1
    buf_size = max_delay + 1
    spike_buf = np.zeros((buf_size, n))

    # Per-synapse state
    eligibility = np.zeros(n_syn)
    gain = np.ones(n_syn)

    # Type masks
    m_plastic = syn_type == SYN_PLASTIC
    m_facil = syn_type == SYN_FACILITATING
    m_depress = syn_type == SYN_DEPRESSING
    m_gated = syn_type == SYN_GATED
    m_has_gain = m_facil | m_depress
    m_has_elig = m_plastic | m_gated

    # Per-synapse parameters
    tau_plus = np.full(n_syn, 20.0)
    lr = np.full(n_syn, 0.01)
    w_min, w_max = np.zeros(n_syn), np.full(n_syn, 10.0)
    facil_inc = np.full(n_syn, 0.3)
    depress_fac = np.full(n_syn, 0.5)
    tau_recovery = np.full(n_syn, 200.0)
    gate_threshold = np.full(n_syn, 0.05)
    gate_groups = [None] * n_syn

    for i in range(n_syn):
        st, props = syn_type[i], syn_props[i] or {}
        # Restore saved state
        if 'eligibility' in props: eligibility[i] = props['eligibility']
        if 'current_gain' in props: gain[i] = props['current_gain']
        # Load type-specific params
        if st == SYN_PLASTIC or st == SYN_GATED:
            tn = 'plastic' if st == SYN_PLASTIC else 'gated'
            tau_plus[i] = _prop(props, 'tau_plus', tn)
            # Accept both 'learning_rate' (schema) and 'lr' (legacy) keys
            lr_val = props.get('learning_rate', props.get('lr', None)) if props else None
            lr[i] = lr_val if lr_val is not None else DEFAULTS[tn]['learning_rate']
            w_min[i] = _prop(props, 'w_min', tn)
            w_max[i] = _prop(props, 'w_max', tn)
            if st == SYN_GATED:
                gate_threshold[i] = _prop(props, 'gate_threshold', 'gated')
                mg = _prop(props, 'modulator_group', 'gated', [])
                gate_groups[i] = np.array(mg, dtype=int) if mg else np.array([], dtype=int)
        elif st == SYN_FACILITATING:
            facil_inc[i] = _prop(props, 'facil_increment', 'facilitating')
            tau_recovery[i] = _prop(props, 'tau_recovery', 'facilitating')
        elif st == SYN_DEPRESSING:
            depress_fac[i] = _prop(props, 'depress_factor', 'depressing')
            tau_recovery[i] = _prop(props, 'tau_recovery', 'depressing')

    # Precompute decay factors
    elig_decay = np.where(m_has_elig, np.exp(-1.0 / tau_plus), 1.0)
    gain_decay = np.where(m_has_gain, np.exp(-1.0 / tau_recovery), 1.0)

    # Program targets
    prog_targets = _resolve_targets(program, brain) if program else None

    # Modulator spike tracking (sliding window for gated synapses)
    mod_win = 50
    mod_counts = np.zeros(n)
    mod_ring = np.zeros((mod_win, n), dtype=bool)

    # Build reverse index: target neuron -> list of synapse indices (for post-spike STDP)
    syn_by_tgt = {}
    for i in range(n_syn):
        if m_has_elig[i]:
            syn_by_tgt.setdefault(syn_tgt[i], []).append(i)

    # Gap junction arrays
    gap_src, gap_tgt, gap_g = brain['gap_src'], brain['gap_tgt'], brain['gap_g']
    n_gap = brain['n_gap']

    total_spikes, interval_spikes = 0, 0

    for t in range(ticks):
        buf_idx = t % buf_size
        I = spike_buf[buf_idx].copy()
        spike_buf[buf_idx] = 0.0

        if program and prog_targets is not None:
            I += _get_current(program, t, rng, n, prog_targets)

        # Gap junctions: I = g * (V_pre - V_post), bidirectional
        if n_gap > 0:
            dv = v[gap_src] - v[gap_tgt]
            np.add.at(I, gap_tgt, gap_g * dv)       # current into target
            np.add.at(I, gap_src, gap_g * (-dv))     # current into source (reverse)

        # Background drive: 2.0 uA tonic + 1.5 sigma noise
        I += rng.randn(n) * 1.5 + 2.0

        # Izhikevich half-step (clamp between halves to prevent overflow)
        v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
        np.clip(v, -100.0, 35.0, out=v)
        v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
        bad = ~np.isfinite(v)
        if np.any(bad):
            v[bad] = c[bad]
        u += a * (b * v - u)

        fired_mask = v >= 30.0
        fired_idx = np.where(fired_mask)[0]
        n_fired = len(fired_idx)
        total_spikes += n_fired
        interval_spikes += n_fired

        # Update modulator tracking
        ri = t % mod_win
        mod_counts -= mod_ring[ri].astype(float)
        mod_ring[ri] = fired_mask
        mod_counts += fired_mask.astype(float)

        if n_fired > 0:
            v[fired_idx] = c[fired_idx]
            u[fired_idx] += d[fired_idx]
            last_spike[fired_idx] = t

            # Deliver spikes per synapse type
            for fi in fired_idx:
                if fi not in syn_by_src:
                    continue
                for si in syn_by_src[fi]:
                    st, w = syn_type[si], syn_w[si]
                    if st == SYN_FIXED:
                        eff_w = w
                    elif st == SYN_PLASTIC:
                        eff_w = w
                        if learn: eligibility[si] += 1.0
                    elif st == SYN_FACILITATING:
                        gain[si] += facil_inc[si]
                        eff_w = w * gain[si]
                    elif st == SYN_DEPRESSING:
                        gain[si] *= depress_fac[si]
                        eff_w = w * gain[si]
                    elif st == SYN_GATED:
                        eff_w = w
                        if learn: eligibility[si] += 1.0
                    else:
                        eff_w = w
                    spike_buf[(t + syn_delay[si]) % buf_size, syn_tgt[si]] += eff_w

            # Post-spike STDP: potentiate synapses targeting neurons that just fired
            if learn:
                for fi in fired_idx:
                    if fi not in syn_by_tgt:
                        continue
                    for si in syn_by_tgt[fi]:
                        if eligibility[si] <= 0:
                            continue
                        st = syn_type[si]
                        if st == SYN_GATED:
                            mg = gate_groups[si]
                            if mg is None or len(mg) == 0:
                                continue
                            if np.sum(mod_counts[mg]) / (len(mg) * mod_win) < gate_threshold[si]:
                                continue
                        # Soft-bounded STDP (sign-aware for inhibitory synapses)
                        range_w = w_max[si] - w_min[si]
                        if range_w < 1e-6:
                            continue
                        if w_min[si] < 0 and w_max[si] <= 0:
                            # Inhibitory synapse: iSTDP (inverted Hebbian)
                            # Pre-before-post -> strengthen inhibition (more negative)
                            dw = -lr[si] * eligibility[si] * (syn_w[si] - w_min[si]) / range_w
                        else:
                            # Excitatory synapse: standard STDP
                            dw = lr[si] * eligibility[si] * (w_max[si] - syn_w[si]) / range_w
                        syn_w[si] = np.clip(syn_w[si] + dw, w_min[si], w_max[si])

        # Per-tick decay
        if learn and np.any(m_has_elig):
            eligibility[m_has_elig] *= elig_decay[m_has_elig]
        if np.any(m_has_gain):
            gain[m_has_gain] = 1.0 + (gain[m_has_gain] - 1.0) * gain_decay[m_has_gain]
        v[v > 30.0] = 30.0

        if not quiet and (t + 1) % 1000 == 0:
            rate = interval_spikes / (n * 1000)
            print(f"  tick {t+1:7d}  |  {interval_spikes:6d} spikes  |  rate {rate:.4f}")
            interval_spikes = 0

    avg_rate = total_spikes / (n * ticks) if ticks > 0 else 0
    print(f"\nDone: {ticks} ticks, {total_spikes} total spikes, avg rate {avg_rate:.4f}")

    # Pack state back for saving
    brain['v'], brain['u'], brain['last_spike'], brain['syn_w'] = v, u, last_spike, syn_w
    for i in range(n_syn):
        if syn_props[i] is None: syn_props[i] = {}
        st = syn_type[i]
        if st in (SYN_PLASTIC, SYN_GATED):
            syn_props[i]['eligibility'] = float(eligibility[i])
            syn_props[i]['weight'] = float(syn_w[i])
        if st in (SYN_FACILITATING, SYN_DEPRESSING):
            syn_props[i]['current_gain'] = float(gain[i])
    return brain


# ── Tick-by-tick interface ──

class BrainState:
    """Step-by-step brain interface. Same physics as simulate(), callable per tick.

    Usage:
        brain = load_brain('path.db')
        bs = BrainState(brain, learn=False, seed=42)
        for t in range(10000):
            fired = bs.step(external_I=my_current_array)
    """

    def __init__(self, brain, learn=False, seed=42):
        self.learn = learn
        self.rng = np.random.RandomState(seed)
        self.tick = 0
        self.n = brain['n']
        self.body_map = brain.get('body_map', {})
        self.sensor_map = brain.get('sensor_map', {})

        # Neuron state
        self.v = brain['v'].copy()
        self.u = brain['u'].copy()
        self.a, self.b, self.c, self.d = brain['a'], brain['b'], brain['c'], brain['d']
        self.last_spike = brain['last_spike'].copy()

        # Chemical synapse arrays
        syn_src, syn_tgt = brain['syn_src'], brain['syn_tgt']
        self.syn_src, self.syn_tgt = syn_src, syn_tgt
        self.syn_w = brain['syn_w'].copy()
        self.syn_delay = brain['syn_delay']
        self.syn_type = brain['syn_type']
        self.syn_props = brain['syn_props']
        self.n_syn = len(syn_src)

        # Per-neuron outgoing synapse index
        self.syn_by_src = {}
        for i in range(self.n_syn):
            self.syn_by_src.setdefault(syn_src[i], []).append(i)

        # Delay buffer
        max_delay = int(np.max(self.syn_delay)) if self.n_syn > 0 else 1
        self.buf_size = max_delay + 1
        self.spike_buf = np.zeros((self.buf_size, self.n))

        # Per-synapse state
        n_syn = self.n_syn
        self.eligibility = np.zeros(n_syn)
        self.gain = np.ones(n_syn)

        # Type masks
        self.m_plastic = self.syn_type == SYN_PLASTIC
        self.m_facil = self.syn_type == SYN_FACILITATING
        self.m_depress = self.syn_type == SYN_DEPRESSING
        self.m_gated = self.syn_type == SYN_GATED
        self.m_has_gain = self.m_facil | self.m_depress
        self.m_has_elig = self.m_plastic | self.m_gated

        # Per-synapse parameters (same extraction as simulate())
        self.tau_plus = np.full(n_syn, 20.0)
        self.lr = np.full(n_syn, 0.01)
        self.w_min = np.zeros(n_syn)
        self.w_max = np.full(n_syn, 10.0)
        self.facil_inc = np.full(n_syn, 0.3)
        self.depress_fac = np.full(n_syn, 0.5)
        self.tau_recovery = np.full(n_syn, 200.0)
        self.gate_threshold = np.full(n_syn, 0.05)
        self.gate_groups = [None] * n_syn

        for i in range(n_syn):
            st, props = self.syn_type[i], self.syn_props[i] or {}
            if 'eligibility' in props: self.eligibility[i] = props['eligibility']
            if 'current_gain' in props: self.gain[i] = props['current_gain']
            if st == SYN_PLASTIC or st == SYN_GATED:
                tn = 'plastic' if st == SYN_PLASTIC else 'gated'
                self.tau_plus[i] = _prop(props, 'tau_plus', tn)
                lr_val = props.get('learning_rate', props.get('lr', None)) if props else None
                self.lr[i] = lr_val if lr_val is not None else DEFAULTS[tn]['learning_rate']
                self.w_min[i] = _prop(props, 'w_min', tn)
                self.w_max[i] = _prop(props, 'w_max', tn)
                if st == SYN_GATED:
                    self.gate_threshold[i] = _prop(props, 'gate_threshold', 'gated')
                    mg = _prop(props, 'modulator_group', 'gated', [])
                    self.gate_groups[i] = np.array(mg, dtype=int) if mg else np.array([], dtype=int)
            elif st == SYN_FACILITATING:
                self.facil_inc[i] = _prop(props, 'facil_increment', 'facilitating')
                self.tau_recovery[i] = _prop(props, 'tau_recovery', 'facilitating')
            elif st == SYN_DEPRESSING:
                self.depress_fac[i] = _prop(props, 'depress_factor', 'depressing')
                self.tau_recovery[i] = _prop(props, 'tau_recovery', 'depressing')

        self.elig_decay = np.where(self.m_has_elig, np.exp(-1.0 / self.tau_plus), 1.0)
        self.gain_decay = np.where(self.m_has_gain, np.exp(-1.0 / self.tau_recovery), 1.0)

        # Modulator tracking
        self.mod_win = 50
        self.mod_counts = np.zeros(self.n)
        self.mod_ring = np.zeros((self.mod_win, self.n), dtype=bool)

        # Reverse index for post-spike STDP
        self.syn_by_tgt = {}
        for i in range(n_syn):
            if self.m_has_elig[i]:
                self.syn_by_tgt.setdefault(syn_tgt[i], []).append(i)

        # Gap junctions
        self.gap_src = brain['gap_src']
        self.gap_tgt = brain['gap_tgt']
        self.gap_g = brain['gap_g']
        self.n_gap = brain['n_gap']

    def step(self, external_I=None):
        """Advance one tick. Returns indices of neurons that fired.

        Args:
            external_I: optional numpy array (shape n) of injected current.
                        Added to background drive. Use for proprioception, stimulation.
        Returns:
            fired_idx: numpy array of neuron indices that spiked this tick.
        """
        t = self.tick
        buf_idx = t % self.buf_size
        I = self.spike_buf[buf_idx].copy()
        self.spike_buf[buf_idx] = 0.0

        # Gap junctions
        if self.n_gap > 0:
            dv = self.v[self.gap_src] - self.v[self.gap_tgt]
            np.add.at(I, self.gap_tgt, self.gap_g * dv)
            np.add.at(I, self.gap_src, self.gap_g * (-dv))

        # Background drive
        I += self.rng.randn(self.n) * 1.5 + 2.0

        # External injection
        if external_I is not None:
            I += external_I

        # Izhikevich half-step (clamp between halves to prevent overflow)
        self.v += 0.5 * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I)
        np.clip(self.v, -100.0, 35.0, out=self.v)
        self.v += 0.5 * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I)
        bad = ~np.isfinite(self.v)
        if np.any(bad):
            self.v[bad] = self.c[bad]
        self.u += self.a * (self.b * self.v - self.u)

        fired_mask = self.v >= 30.0
        fired_idx = np.where(fired_mask)[0]

        # Modulator tracking
        ri = t % self.mod_win
        self.mod_counts -= self.mod_ring[ri].astype(float)
        self.mod_ring[ri] = fired_mask
        self.mod_counts += fired_mask.astype(float)

        if len(fired_idx) > 0:
            self.v[fired_idx] = self.c[fired_idx]
            self.u[fired_idx] += self.d[fired_idx]
            self.last_spike[fired_idx] = t

            # Deliver spikes
            for fi in fired_idx:
                if fi not in self.syn_by_src:
                    continue
                for si in self.syn_by_src[fi]:
                    st, w = self.syn_type[si], self.syn_w[si]
                    if st == SYN_FIXED:
                        eff_w = w
                    elif st == SYN_PLASTIC:
                        eff_w = w
                        if self.learn: self.eligibility[si] += 1.0
                    elif st == SYN_FACILITATING:
                        self.gain[si] += self.facil_inc[si]
                        eff_w = w * self.gain[si]
                    elif st == SYN_DEPRESSING:
                        self.gain[si] *= self.depress_fac[si]
                        eff_w = w * self.gain[si]
                    elif st == SYN_GATED:
                        eff_w = w
                        if self.learn: self.eligibility[si] += 1.0
                    else:
                        eff_w = w
                    self.spike_buf[(t + self.syn_delay[si]) % self.buf_size, self.syn_tgt[si]] += eff_w

            # Post-spike STDP
            if self.learn:
                for fi in fired_idx:
                    if fi not in self.syn_by_tgt:
                        continue
                    for si in self.syn_by_tgt[fi]:
                        if self.eligibility[si] <= 0:
                            continue
                        st = self.syn_type[si]
                        if st == SYN_GATED:
                            mg = self.gate_groups[si]
                            if mg is None or len(mg) == 0:
                                continue
                            if np.sum(self.mod_counts[mg]) / (len(mg) * self.mod_win) < self.gate_threshold[si]:
                                continue
                        # Soft-bounded STDP (sign-aware for inhibitory synapses)
                        range_w = self.w_max[si] - self.w_min[si]
                        if range_w < 1e-6:
                            continue
                        if self.w_min[si] < 0 and self.w_max[si] <= 0:
                            # Inhibitory synapse: iSTDP (inverted Hebbian)
                            dw = -self.lr[si] * self.eligibility[si] * (self.syn_w[si] - self.w_min[si]) / range_w
                        else:
                            # Excitatory synapse: standard STDP
                            dw = self.lr[si] * self.eligibility[si] * (self.w_max[si] - self.syn_w[si]) / range_w
                        self.syn_w[si] = np.clip(self.syn_w[si] + dw, self.w_min[si], self.w_max[si])

        # Per-tick decay
        if self.learn and np.any(self.m_has_elig):
            self.eligibility[self.m_has_elig] *= self.elig_decay[self.m_has_elig]
        if np.any(self.m_has_gain):
            self.gain[self.m_has_gain] = 1.0 + (self.gain[self.m_has_gain] - 1.0) * self.gain_decay[self.m_has_gain]
        self.v[self.v > 30.0] = 30.0

        self.tick += 1
        return fired_idx


# ── Program handling ──

def load_program(path):
    with open(path) as f:
        prog = json.load(f)
    if not quiet_global:
        print(f"  Program: {prog.get('name', os.path.basename(path))}")
        if prog.get('description'): print(f"  {prog['description']}")
        print(f"  {len(prog['phases'])} phases, loop={prog.get('loop', False)}")
    return prog

def _resolve_targets(program, brain):
    targets = program.get('targets', [])
    if targets and isinstance(targets[0], int):
        return np.array(targets, dtype=int)
    cluster_map = brain.get('cluster_map', {})
    ids = []
    for label in targets:
        ids.extend(cluster_map.get(label, []))
    if ids:
        return np.array(sorted(set(ids)), dtype=int)
    n10 = max(1, brain['n'] // 10)
    if not quiet_global:
        print(f"  No neurons for targets {targets} — using first {n10}")
    return np.arange(n10)

def _get_current(program, tick, rng, n, target_ids):
    phases = program['phases']
    loop = program.get('loop', False)
    # tick_start/tick_end format
    if 'tick_start' in phases[0]:
        ext = np.zeros(n)
        for p in phases:
            if p['tick_start'] <= tick < p['tick_end']:
                ids = np.array(p['targets'], dtype=int) if 'targets' in p and isinstance(p['targets'][0], int) else target_ids
                ext[ids] = p.get('current', 0.0) + rng.randn(len(ids)) * p.get('noise', 0.0)
                break
        return ext
    # duration format
    total = sum(p['duration'] for p in phases if p['duration'] > 0)
    if total == 0:
        phase = phases[0]
    else:
        t_in = tick % total if loop else tick
        elapsed, phase = 0, phases[-1]
        for p in phases:
            if p['duration'] <= 0: continue
            if t_in < elapsed + p['duration']:
                phase = p; break
            elapsed += p['duration']
    ext = np.zeros(n)
    cur, noise = phase.get('current', 0.0), phase.get('noise', 0.0)
    if cur != 0.0 or noise != 0.0:
        ext[target_ids] = cur + rng.randn(len(target_ids)) * noise
    return ext


# ── CLI ──

def main():
    global quiet_global
    parser = argparse.ArgumentParser(description='V8 simulation engine — per-synapse dispatch')
    parser.add_argument('brain', help='Path to brain .db file')
    parser.add_argument('--ticks', type=int, default=10000, help='Ticks to run (default 10000)')
    parser.add_argument('--save', action='store_true', help='Save state back to DB after sim')
    parser.add_argument('--learn', action='store_true', help='Enable plasticity on plastic/gated synapses')
    parser.add_argument('--program', help='Path to program JSON')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default 42)')
    args = parser.parse_args()
    quiet_global = args.quiet

    db_path = args.brain
    if not os.path.isabs(db_path):
        db_path = os.path.join(BASE, db_path)
    if not os.path.exists(db_path):
        print(f"Not found: {db_path}"); sys.exit(1)

    if not args.quiet: print(f"Loading brain: {db_path}")
    brain = load_brain(db_path)
    n, n_syn = brain['n'], len(brain['syn_src'])

    if not args.quiet:
        type_counts = {}
        for st in [SYN_FIXED, SYN_PLASTIC, SYN_FACILITATING, SYN_DEPRESSING, SYN_GATED]:
            cnt = int(np.sum(brain['syn_type'] == st))
            if cnt > 0:
                type_counts[st] = cnt
        n_gap = brain['n_gap']
        if n_gap > 0:
            type_counts['gap_junction'] = n_gap
        print(f"  {n} neurons, {n_syn + n_gap:,} synapses ({n_gap} gap junctions)")
        print(f"  Synapse types: {', '.join(f'{v} {k}' for k,v in type_counts.items()) or 'none'}")
        exc_types = {'RS', 'IB', 'CH'}
        n_exc = sum(1 for t in brain['neuron_types'] if t in exc_types)
        print(f"  Neurons: {n_exc}E / {n - n_exc}I")

    program = None
    if args.program:
        prog_path = args.program if os.path.isabs(args.program) else os.path.join(BASE, args.program)
        if not os.path.exists(prog_path):
            print(f"Program not found: {prog_path}"); sys.exit(1)
        program = load_program(prog_path)

    if not args.quiet:
        print(f"  Learning: {'ON' if args.learn else 'OFF'}, Ticks: {args.ticks}")
        print(f"Running...")

    rng = np.random.RandomState(args.seed)
    brain = simulate(brain, args.ticks, args.learn, program, args.quiet, rng)

    if args.save:
        save_brain_state(brain, db_path)
        print(f"Brain state saved to {db_path}")


if __name__ == '__main__':
    main()
