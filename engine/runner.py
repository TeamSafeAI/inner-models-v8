"""
runner.py — THE engine. NumPy-optimized.

Same behavior as original. Same math. Just faster.
Parts still own their rules — the engine applies them in bulk.

Optimizations:
1. Vectorized Izhikevich neuron update (all N neurons at once)
2. Vectorized gap junction coupling
3. NumPy spike buffer (2D array)
4. Skip per_tick for fixed/gap_junction (~80% of synapses)
5. Vectorized per_tick decay (one multiply per synapse group)
6. Pre-built fixed synapse arrays for delivery (no function calls)
7. Pre-computed exp() decay factors (once at init, not per tick)

Formulas match module files exactly:
  Neuron update: engine/neurons/*.py (all identical Izhikevich)
  Plastic/gated decay: engine/paths/plastic.py, gated.py
  Facilitating/depressing recovery: engine/paths/facilitating.py, depressing.py

NOTE: Neuron dicts are NOT updated during tick() for performance.
  Use brain.v[i], brain.u[i] for current state.
  Call brain.sync_state() before save() or dict inspection.
"""
import math
import numpy as np
from engine.recorder import Recorder


class Brain:
    """A running brain. Load it, tick it, record it."""

    def __init__(self, brain_data, learn=True):
        self.data = brain_data
        self.neurons = brain_data['neurons']
        self.synapses = brain_data['synapses']
        self.gap_junctions = brain_data['gap_junctions']
        self.learn = learn
        self.n = len(self.neurons)
        self.tick_count = 0

        # --- NumPy neuron state ---
        self.v = np.array([n['v'] for n in self.neurons], dtype=np.float64)
        self.u = np.array([n['u'] for n in self.neurons], dtype=np.float64)
        self.a = np.array([n['a'] for n in self.neurons], dtype=np.float64)
        self.b = np.array([n['b'] for n in self.neurons], dtype=np.float64)
        self.c = np.array([n['c'] for n in self.neurons], dtype=np.float64)
        self.d = np.array([n['d'] for n in self.neurons], dtype=np.float64)
        self.last_spike = np.array([n['last_spike'] for n in self.neurons], dtype=np.int64)

        # --- Gap junction arrays ---
        ng = len(self.gap_junctions)
        if ng > 0:
            self.gj_src = np.array([g['source'] for g in self.gap_junctions], dtype=np.intp)
            self.gj_tgt = np.array([g['target'] for g in self.gap_junctions], dtype=np.intp)
            self.gj_g = np.array([g['conductance'] for g in self.gap_junctions], dtype=np.float64)
        else:
            self.gj_src = self.gj_tgt = np.empty(0, dtype=np.intp)
            self.gj_g = np.empty(0, dtype=np.float64)
        self.has_gj = ng > 0

        # --- Spike delay buffer ---
        max_delay = max((s['delay'] for s in self.synapses), default=1)
        self.buf_size = max_delay + 1
        self.spike_buf = np.zeros((self.buf_size, self.n), dtype=np.float64)

        # --- Pre-categorize synapses ---
        self._build_synapse_structures()

        # --- Modulator tracking (only for gated synapses) ---
        self.mod_window = 50
        if self.has_gated:
            self.mod_ring = np.zeros((self.mod_window, self.n), dtype=np.int8)
            self.mod_counts = np.zeros(self.n, dtype=np.int32)

        # Recorder
        self.recorder = Recorder(brain_data)

    def _build_synapse_structures(self):
        """Categorize synapses by type, build arrays for vectorized ops."""
        synapses = self.synapses

        # Fixed: per-source arrays (target, weight, delay)
        fixed_lists = {}

        # Dynamic: per-source lookups as (synapse_index, array_position)
        plastic_by_src = {}
        gated_by_src = {}
        reward_by_src = {}
        facil_by_src = {}
        dep_by_src = {}
        dev_by_src = {}

        # Learning: per-target lookups (only plastic/gated/reward_plastic/developmental)
        plastic_by_tgt = {}
        gated_by_tgt = {}
        reward_by_tgt = {}
        dev_by_tgt = {}

        # Collectors for array building
        plastic_idx, plastic_tau = [], []
        gated_idx, gated_tau = [], []
        reward_idx, reward_tau_trace, reward_tau_elig = [], [], []
        facil_idx, facil_tau, facil_inc_vals = [], [], []
        dep_idx, dep_tau, dep_fac_vals = [], [], []
        dev_idx, dev_tau = [], []

        for i, syn in enumerate(synapses):
            src, tgt, stype = syn['source'], syn['target'], syn['type']

            if stype == 'fixed':
                if src not in fixed_lists:
                    fixed_lists[src] = ([], [], [])
                fl = fixed_lists[src]
                fl[0].append(tgt)
                fl[1].append(syn['weight'])
                fl[2].append(syn['delay'])

            elif stype == 'plastic':
                pos = len(plastic_idx)
                plastic_idx.append(i)
                plastic_tau.append(syn.get('tau_plus', 20.0))
                plastic_by_src.setdefault(src, []).append((i, pos))
                plastic_by_tgt.setdefault(tgt, []).append((i, pos))

            elif stype == 'gated':
                pos = len(gated_idx)
                gated_idx.append(i)
                gated_tau.append(syn.get('tau_plus', 20.0))
                gated_by_src.setdefault(src, []).append((i, pos))
                gated_by_tgt.setdefault(tgt, []).append((i, pos))

            elif stype == 'reward_plastic':
                pos = len(reward_idx)
                reward_idx.append(i)
                reward_tau_trace.append(syn.get('tau_trace', 20.0))
                reward_tau_elig.append(syn.get('tau_eligible', 500.0))
                reward_by_src.setdefault(src, []).append((i, pos))
                reward_by_tgt.setdefault(tgt, []).append((i, pos))

            elif stype == 'facilitating':
                pos = len(facil_idx)
                facil_idx.append(i)
                facil_tau.append(syn.get('tau_recovery', 200.0))
                facil_inc_vals.append(syn.get('facil_increment', 0.2))
                facil_by_src.setdefault(src, []).append((i, pos))

            elif stype == 'depressing':
                pos = len(dep_idx)
                dep_idx.append(i)
                dep_tau.append(syn.get('tau_recovery', 500.0))
                dep_fac_vals.append(syn.get('depress_factor', 0.5))
                dep_by_src.setdefault(src, []).append((i, pos))

            elif stype == 'developmental':
                pos = len(dev_idx)
                dev_idx.append(i)
                dev_tau.append(syn.get('tau_plus', 20.0))
                dev_by_src.setdefault(src, []).append((i, pos))
                dev_by_tgt.setdefault(tgt, []).append((i, pos))

        # Fixed: convert to NumPy arrays per source
        self.fixed_out = {}
        for src, (tgts, wts, dlys) in fixed_lists.items():
            self.fixed_out[src] = (
                np.array(tgts, dtype=np.intp),
                np.array(wts, dtype=np.float64),
                np.array(dlys, dtype=np.intp),
            )

        # Store lookups
        self.plastic_by_source = plastic_by_src
        self.plastic_by_target = plastic_by_tgt
        self.gated_by_source = gated_by_src
        self.gated_by_target = gated_by_tgt
        self.reward_by_source = reward_by_src
        self.reward_by_target = reward_by_tgt
        self.facil_by_source = facil_by_src
        self.dep_by_source = dep_by_src
        self.dev_by_source = dev_by_src
        self.dev_by_target = dev_by_tgt

        # Plastic: eligibility + pre-computed decay factor
        self.plastic_idx = plastic_idx
        self.plastic_elig = np.zeros(len(plastic_idx), dtype=np.float64)
        self.plastic_decay = np.array(
            [math.exp(-1.0 / t) if t > 0 else 0.0 for t in plastic_tau],
            dtype=np.float64)

        # Gated: same structure
        self.gated_idx = gated_idx
        self.gated_elig = np.zeros(len(gated_idx), dtype=np.float64)
        self.gated_decay = np.array(
            [math.exp(-1.0 / t) if t > 0 else 0.0 for t in gated_tau],
            dtype=np.float64)
        self.has_gated = len(gated_idx) > 0

        # Reward-plastic: trace (fast) + eligibility (slow) + two decay factors
        self.reward_idx = reward_idx
        self.reward_trace = np.zeros(len(reward_idx), dtype=np.float64)
        self.reward_elig = np.zeros(len(reward_idx), dtype=np.float64)
        self.reward_trace_decay = np.array(
            [math.exp(-1.0 / t) if t > 0 else 0.0 for t in reward_tau_trace],
            dtype=np.float64)
        self.reward_elig_decay = np.array(
            [math.exp(-1.0 / t) if t > 0 else 0.0 for t in reward_tau_elig],
            dtype=np.float64)
        self.has_reward = len(reward_idx) > 0

        # Synaptic homeostasis: group reward synapses by target neuron
        # After reward delivery, normalize inputs per target to prevent runaway weights
        # (Ref: Friedrich et al 2014, "rewarded STDP alone insufficient without homeostasis")
        if self.has_reward:
            self.reward_target_groups = {}  # target_id -> list of reward array indices
            for ri, si in enumerate(reward_idx):
                tid = synapses[si]['target']
                if tid not in self.reward_target_groups:
                    self.reward_target_groups[tid] = []
                self.reward_target_groups[tid].append(ri)
            # Cache initial weight total per target for normalization target
            self.reward_target_w0 = {}
            for tid, indices in self.reward_target_groups.items():
                total = sum(synapses[reward_idx[ri]]['weight'] for ri in indices)
                self.reward_target_w0[tid] = max(total, 1e-6)

        # Facilitating: gain + decay + increment
        self.facil_idx = facil_idx
        self.facil_gain = np.ones(len(facil_idx), dtype=np.float64)
        self.facil_decay = np.array(
            [math.exp(-1.0 / t) if t > 0 else 0.0 for t in facil_tau],
            dtype=np.float64)
        self.facil_inc = np.array(facil_inc_vals, dtype=np.float64)

        # Depressing: gain + decay + factor
        self.dep_idx = dep_idx
        self.dep_gain = np.ones(len(dep_idx), dtype=np.float64)
        self.dep_decay = np.array(
            [math.exp(-1.0 / t) if t > 0 else 0.0 for t in dep_tau],
            dtype=np.float64)
        self.dep_fac = np.array(dep_fac_vals, dtype=np.float64)

        # Developmental: eligibility + decay + alive mask + FI counters
        self.dev_idx = dev_idx
        self.dev_decay = np.array(
            [math.exp(-1.0 / t) if t > 0 else 0.0 for t in dev_tau],
            dtype=np.float64)
        self.has_dev = len(dev_idx) > 0
        if self.has_dev:
            # Restore state from saved synapses (supports multi-session)
            self.dev_elig = np.array(
                [synapses[i].get('eligibility', 0.0) for i in dev_idx], dtype=np.float64)
            self.dev_alive = np.array(
                [synapses[i].get('alive', True) for i in dev_idx], dtype=bool)
            self.dev_src_fires = np.array(
                [synapses[i].get('source_fires', 0) for i in dev_idx], dtype=np.int64)
            self.dev_coincidences = np.array(
                [synapses[i].get('coincidences', 0) for i in dev_idx], dtype=np.int64)
            # Cache critical period params from first dev synapse
            s0 = synapses[dev_idx[0]]
            self.dev_critical_period = s0.get('critical_period', 10000)
            self.dev_eval_interval = s0.get('eval_interval', 2000)
            self.dev_prune_thresh = s0.get('pruning_threshold', 0.02)
            self.dev_min_fires = s0.get('min_source_fires', 20)
            # If synapses already have significant experience, skip critical period
            max_fires = int(np.max(self.dev_src_fires)) if len(self.dev_src_fires) > 0 else 0
            if max_fires > self.dev_min_fires * 10:
                self.dev_critical_done = True
                n_dead = int(np.sum(~self.dev_alive))
                if n_dead > 0:
                    print(f"  [dev] Loaded: {n_dead} already pruned, critical period complete")
            else:
                self.dev_critical_done = False
        else:
            self.dev_elig = np.zeros(0, dtype=np.float64)
            self.dev_alive = np.ones(0, dtype=bool)
            self.dev_src_fires = np.zeros(0, dtype=np.int64)
            self.dev_coincidences = np.zeros(0, dtype=np.int64)

        # Stats
        n_fixed = sum(len(v[0]) for v in self.fixed_out.values())
        n_dyn = len(plastic_idx) + len(gated_idx) + len(reward_idx) + len(facil_idx) + len(dep_idx) + len(dev_idx)
        total = n_fixed + n_dyn
        pct = (n_fixed / total * 100) if total > 0 else 0
        dev_str = f", {len(dev_idx)} developmental" if dev_idx else ""
        print(f"  Synapses: {n_fixed} fixed ({pct:.0f}%) + {n_dyn} dynamic = {total}{dev_str}")

    def tick(self, external_I=None):
        """One tick of the brain. Returns list of fired neuron indices."""
        t = self.tick_count
        v, u = self.v, self.u

        # 1. Collect spikes from delay buffer
        buf_idx = t % self.buf_size
        I = self.spike_buf[buf_idx].copy()
        self.spike_buf[buf_idx] = 0.0

        # 2. External current
        if external_I is not None:
            I += external_I

        # 3. Gap junctions (vectorized)
        if self.has_gj:
            dv = v[self.gj_src] - v[self.gj_tgt]
            gi = self.gj_g * dv
            np.add.at(I, self.gj_tgt, gi)
            np.add.at(I, self.gj_src, -gi)

        # 4. Izhikevich update (vectorized, matches engine/neurons/*.py)
        v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
        np.clip(v, -100.0, 35.0, out=v)
        v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
        u += self.a * (self.b * v - u)

        # 5. Spike detection + reset
        fired = np.flatnonzero(v >= 30.0)
        if len(fired) > 0:
            v[fired] = self.c[fired]
            u[fired] += self.d[fired]
            self.last_spike[fired] = t

        # 6. Modulator tracking
        if self.has_gated:
            ri = t % self.mod_window
            self.mod_counts -= self.mod_ring[ri]
            self.mod_ring[ri] = 0
            if len(fired) > 0:
                self.mod_ring[ri][fired] = 1
                self.mod_counts[fired] += 1

        # 7. Spike delivery
        fired_list = fired.tolist()
        synapses = self.synapses
        spike_buf = self.spike_buf
        buf_size = self.buf_size

        for fi in fired_list:
            # 7a. Fixed: array-based, no function calls
            fixed = self.fixed_out.get(fi)
            if fixed is not None:
                tgts, wts, dlys = fixed
                np.add.at(spike_buf, ((t + dlys) % buf_size, tgts), wts)

            # 7b. Plastic: bump eligibility, deliver weight
            for si, pi in self.plastic_by_source.get(fi, ()):
                self.plastic_elig[pi] += 1.0
                syn = synapses[si]
                spike_buf[(t + syn['delay']) % buf_size, syn['target']] += syn['weight']

            # 7c. Gated: bump eligibility, deliver weight
            for si, gi in self.gated_by_source.get(fi, ()):
                self.gated_elig[gi] += 1.0
                syn = synapses[si]
                spike_buf[(t + syn['delay']) % buf_size, syn['target']] += syn['weight']

            # 7c2. Reward-plastic: bump trace, deliver weight
            for si, ri in self.reward_by_source.get(fi, ()):
                self.reward_trace[ri] += 1.0
                syn = synapses[si]
                spike_buf[(t + syn['delay']) % buf_size, syn['target']] += syn['weight']

            # 7d. Facilitating: bump gain, deliver weight * gain
            for si, gi in self.facil_by_source.get(fi, ()):
                self.facil_gain[gi] += self.facil_inc[gi]
                syn = synapses[si]
                spike_buf[(t + syn['delay']) % buf_size, syn['target']] += \
                    syn['weight'] * self.facil_gain[gi]

            # 7e. Depressing: multiply gain, deliver weight * gain
            for si, gi in self.dep_by_source.get(fi, ()):
                self.dep_gain[gi] *= self.dep_fac[gi]
                syn = synapses[si]
                spike_buf[(t + syn['delay']) % buf_size, syn['target']] += \
                    syn['weight'] * self.dep_gain[gi]

            # 7f. Developmental: bump eligibility, count source fire, deliver if alive
            for si, di in self.dev_by_source.get(fi, ()):
                if not self.dev_alive[di]:
                    continue
                self.dev_elig[di] += 1.0
                self.dev_src_fires[di] += 1
                syn = synapses[si]
                spike_buf[(t + syn['delay']) % buf_size, syn['target']] += syn['weight']

        # 8. Learning (STDP on plastic/gated, only for fired targets)
        if self.learn and len(fired) > 0:
            for fi in fired_list:
                for si, pi in self.plastic_by_target.get(fi, ()):
                    self._apply_stdp(synapses[si], self.plastic_elig[pi])

                if self.has_gated:
                    for si, gi in self.gated_by_target.get(fi, ()):
                        syn = synapses[si]
                        mg = syn.get('modulator_group', [])
                        if mg:
                            activity = float(self.mod_counts[mg].sum()) / \
                                (len(mg) * self.mod_window)
                        else:
                            activity = 0.0
                        if activity >= syn.get('gate_threshold', 0.3):
                            self._apply_stdp(syn, self.gated_elig[gi])

                # Reward-plastic: target fired -> convert trace to eligibility (NO weight change)
                if self.has_reward:
                    for si, ri in self.reward_by_target.get(fi, ()):
                        trace = self.reward_trace[ri]
                        if trace > 0:
                            self.reward_elig[ri] += trace
                            self.reward_trace[ri] *= 0.5  # partial consumption

                # Developmental: STDP + coincidence counting
                if self.has_dev:
                    for si, di in self.dev_by_target.get(fi, ()):
                        if not self.dev_alive[di]:
                            continue
                        elig = self.dev_elig[di]
                        if elig > 0:
                            self.dev_coincidences[di] += 1
                            self._apply_stdp(synapses[si], elig)

        # 9. Per-tick decay (vectorized — THE big optimization)
        if len(self.plastic_elig) > 0:
            self.plastic_elig *= self.plastic_decay
        if len(self.gated_elig) > 0:
            self.gated_elig *= self.gated_decay
        if self.has_reward:
            self.reward_trace *= self.reward_trace_decay
            self.reward_elig *= self.reward_elig_decay
        if len(self.facil_gain) > 0:
            self.facil_gain[:] = 1.0 + (self.facil_gain - 1.0) * self.facil_decay
        if len(self.dep_gain) > 0:
            self.dep_gain[:] = 1.0 + (self.dep_gain - 1.0) * self.dep_decay
        if self.has_dev:
            self.dev_elig *= self.dev_decay

        # 10. Developmental pruning (periodic, only during critical period)
        if self.has_dev and not self.dev_critical_done and t < self.dev_critical_period and t > 0 and t % self.dev_eval_interval == 0:
            self._prune_developmental()
        if self.has_dev and not self.dev_critical_done and t >= self.dev_critical_period:
            self.dev_critical_done = True

        # 11. Record
        self.recorder.record_tick(fired_list)
        self.tick_count += 1
        return fired_list

    @staticmethod
    def _apply_stdp(syn, elig):
        """Soft-bounded STDP. Matches plastic.py / gated.py on_target_fired."""
        if elig <= 0:
            return
        w = syn['weight']
        lr = syn['learning_rate']
        w_min = syn['w_min']
        w_max = syn['w_max']
        rng = w_max - w_min
        if rng < 1e-6:
            return
        if w_min < 0 and w_max <= 0:
            dw = -lr * elig * (w - w_min) / rng
        else:
            dw = lr * elig * (w_max - w) / rng
        syn['weight'] = max(w_min, min(w_max, w + dw))

    def _prune_developmental(self):
        """Evaluate FI and prune low-correlation developmental synapses."""
        pruned = 0
        synapses = self.synapses
        for di in range(len(self.dev_idx)):
            if not self.dev_alive[di]:
                continue
            src_fires = self.dev_src_fires[di]
            if src_fires < self.dev_min_fires:
                continue  # not enough data yet
            coincidences = self.dev_coincidences[di]
            rate = coincidences / src_fires
            if rate < self.dev_prune_thresh:
                # Kill this synapse
                self.dev_alive[di] = False
                si = self.dev_idx[di]
                synapses[si]['weight'] = 0.0
                pruned += 1
        if pruned > 0:
            alive = int(self.dev_alive.sum())
            total = len(self.dev_idx)
            print(f"  [dev] Pruned {pruned} synapses. {alive}/{total} alive ({alive/total*100:.0f}%)")

    def deliver_reward(self, magnitude):
        """Deliver reward signal to all reward_plastic synapses.

        magnitude > 0: positive reward (potentiate eligible connections)
        magnitude < 0: negative reward (depress eligible connections)
        magnitude = 0: no effect

        Call this from external code (arena, test scripts) when the agent
        receives a reward or punishment signal.
        """
        if not self.has_reward or abs(magnitude) < 1e-6:
            return

        synapses = self.synapses
        for ri, si in enumerate(self.reward_idx):
            elig = self.reward_elig[ri]
            if abs(elig) < 1e-6:
                continue

            syn = synapses[si]
            w = syn['weight']
            lr = syn['learning_rate']
            w_min = syn['w_min']
            w_max = syn['w_max']
            range_w = w_max - w_min
            if range_w < 1e-6:
                continue

            if w_min < 0 and w_max <= 0:
                # Inhibitory
                dw = -lr * elig * magnitude * (w - w_min) / range_w
            else:
                # Excitatory
                if magnitude > 0:
                    dw = lr * elig * magnitude * (w_max - w) / range_w
                else:
                    dw = lr * elig * magnitude * (w - w_min) / range_w

            syn['weight'] = max(w_min, min(w_max, w + dw))
            # Partially consume eligibility
            self.reward_elig[ri] *= 0.5

        # Synaptic homeostasis: soft normalization per target neuron
        # Pulls total input toward initial value, prevents runaway but allows learning
        # Rate 0.1 = 10% correction per reward delivery (~400 deliveries/session)
        homeo_rate = 0.1
        for tid, indices in self.reward_target_groups.items():
            if len(indices) < 2:
                continue
            current_total = 0.0
            for ri in indices:
                current_total += synapses[self.reward_idx[ri]]['weight']
            if current_total < 1e-6:
                continue
            target_total = self.reward_target_w0[tid]
            # Soft pull: scale = 1 + rate * (target/current - 1)
            ratio = target_total / current_total
            scale = 1.0 + homeo_rate * (ratio - 1.0)
            if abs(scale - 1.0) < 1e-8:
                continue
            for ri in indices:
                si = self.reward_idx[ri]
                w = synapses[si]['weight']
                synapses[si]['weight'] = max(synapses[si]['w_min'],
                                             min(synapses[si]['w_max'], w * scale))

    def sync_state(self):
        """Write NumPy state back to dicts (call before save)."""
        for i, n in enumerate(self.neurons):
            n['v'] = float(self.v[i])
            n['u'] = float(self.u[i])
            n['last_spike'] = int(self.last_spike[i])
        for pi, si in enumerate(self.plastic_idx):
            self.synapses[si]['eligibility'] = float(self.plastic_elig[pi])
        for gi, si in enumerate(self.gated_idx):
            self.synapses[si]['eligibility'] = float(self.gated_elig[gi])
        for ri, si in enumerate(self.reward_idx):
            self.synapses[si]['trace'] = float(self.reward_trace[ri])
            self.synapses[si]['eligibility'] = float(self.reward_elig[ri])
        for di, si in enumerate(self.dev_idx):
            self.synapses[si]['eligibility'] = float(self.dev_elig[di])
            self.synapses[si]['source_fires'] = int(self.dev_src_fires[di])
            self.synapses[si]['coincidences'] = int(self.dev_coincidences[di])
            self.synapses[si]['alive'] = bool(self.dev_alive[di])
        for gi, si in enumerate(self.facil_idx):
            self.synapses[si]['current_gain'] = float(self.facil_gain[gi])
        for gi, si in enumerate(self.dep_idx):
            self.synapses[si]['current_gain'] = float(self.dep_gain[gi])

    def save(self):
        """Save brain state to database."""
        from engine.loader import save
        self.sync_state()
        save(self.data)

    def run(self, ticks, external_I=None, quiet=False):
        """Run for N ticks. Returns recorder."""
        if external_I is not None:
            external_I = np.asarray(external_I, dtype=np.float64)

        for t in range(ticks):
            self.tick(external_I=external_I)

            if not quiet and (t + 1) % 1000 == 0:
                total = len(self.recorder.spikes)
                rate = total / (self.n * (t + 1))
                print(f"  tick {t+1:7d}  |  rate {rate:.4f}")

        return self.recorder
