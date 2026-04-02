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

    def __init__(self, brain_data, learn=True, reward_homeostasis=False):
        self.data = brain_data
        self.neurons = brain_data['neurons']
        self.synapses = brain_data['synapses']
        self.gap_junctions = brain_data['gap_junctions']
        self.learn = learn
        self.reward_homeostasis = reward_homeostasis
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
        plastic_idx, plastic_tau, plastic_tau_minus, plastic_ltd_ratio = [], [], [], []
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
                plastic_tau_minus.append(syn.get('tau_minus', 20.0))
                plastic_ltd_ratio.append(syn.get('ltd_ratio', 0.5))
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

        # Plastic: eligibility (pre-trace) + post-trace for LTD + decay factors
        self.plastic_idx = plastic_idx
        self.plastic_elig = np.zeros(len(plastic_idx), dtype=np.float64)
        self.plastic_elig_post = np.zeros(len(plastic_idx), dtype=np.float64)
        self.plastic_decay = np.array(
            [math.exp(-1.0 / t) if t > 0 else 0.0 for t in plastic_tau],
            dtype=np.float64)
        self.plastic_decay_post = np.array(
            [math.exp(-1.0 / t) if t > 0 else 0.0 for t in plastic_tau_minus],
            dtype=np.float64)
        self.plastic_ltd_ratio = np.array(plastic_ltd_ratio, dtype=np.float64)

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

            # 7b. Plastic: bump pre-elig, apply LTD if post fired recently, deliver weight
            for si, pi in self.plastic_by_source.get(fi, ()):
                self.plastic_elig[pi] += 1.0
                # LTD: source fired after target (post-before-pre = anti-causal)
                ep = self.plastic_elig_post[pi]
                if ep > 0.01:
                    syn = synapses[si]
                    w = syn['weight']
                    lr = syn['learning_rate']
                    w_min, w_max = syn['w_min'], syn['w_max']
                    rng = w_max - w_min
                    if rng > 1e-6:
                        ltd_r = self.plastic_ltd_ratio[pi]
                        if w_min < 0 and w_max <= 0:
                            dw = lr * ltd_r * ep * (w_max - w) / rng
                        else:
                            dw = -lr * ltd_r * ep * (w - w_min) / rng
                        syn['weight'] = max(w_min, min(w_max, w + dw))
                    spike_buf[(t + syn['delay']) % buf_size, syn['target']] += syn['weight']
                else:
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
                    self.plastic_elig_post[pi] += 1.0  # mark post-spike for LTD
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
            self.plastic_elig_post *= self.plastic_decay_post
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
        # OFF by default -- use sleep() compression instead
        # When enabled: 10% correction per reward delivery (~400 deliveries/session)
        if not self.reward_homeostasis:
            return
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
            # Save drifted params (may have changed from original type)
            n['a'] = float(self.a[i])
            n['b'] = float(self.b[i])
            n['c'] = float(self.c[i])
            n['d'] = float(self.d[i])
        for pi, si in enumerate(self.plastic_idx):
            self.synapses[si]['eligibility'] = float(self.plastic_elig[pi])
            self.synapses[si]['elig_post'] = float(self.plastic_elig_post[pi])
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

    def sprout(self, max_new=50, window=2000, min_cofire=3, weight=0.5,
               max_distance=8.0, seed=None):
        """Synaptogenesis: grow new plastic synapses between co-active neurons.

        Biology: axonal sprouting creates new connections between neurons that
        fire together but lack direct connections. Axons only reach nearby
        neurons (spatial constraint), so co-firing + proximity = new synapse.

        Uses recent spike history + neuron spatial positions (pos_x, pos_y, pos_z)
        to find co-active, spatially close pairs without existing connections.

        Args:
            max_new: max new synapses to create per call
            window: how many recent ticks of spike data to analyze
            min_cofire: minimum co-firing count to consider a pair
            weight: initial weight for new synapses (small but non-zero)
            max_distance: maximum spatial distance for sprouting (axon reach)
            seed: RNG seed for tie-breaking when more candidates than max_new

        Returns:
            dict with sprouting stats
        """
        from engine.paths import plastic as plastic_module

        spikes = self.recorder.spikes
        if len(spikes) < min_cofire * 2:
            return {'sprouted': 0, 'candidates': 0}

        # Get recent spikes within window
        cutoff = self.tick_count - window
        recent = [(t, n) for t, n in spikes if t >= cutoff]
        if not recent:
            return {'sprouted': 0, 'candidates': 0}

        # Build spatial position arrays for distance checks
        pos_x = np.array([n.get('pos_x', 0.0) for n in self.neurons], dtype=np.float64)
        pos_y = np.array([n.get('pos_y', 0.0) for n in self.neurons], dtype=np.float64)
        pos_z = np.array([n.get('pos_z', 0.0) for n in self.neurons], dtype=np.float64)

        # Build per-tick fired sets (bin into 5-tick windows for co-firing)
        bin_size = 5
        bins = {}
        for t, n in recent:
            b = t // bin_size
            if b not in bins:
                bins[b] = set()
            bins[b].add(n)

        # Count co-firings for neuron pairs (sample bins to keep tractable)
        rng = np.random.RandomState(seed or 0)
        bin_keys = list(bins.keys())
        if len(bin_keys) > 200:
            bin_keys = rng.choice(bin_keys, 200, replace=False).tolist()

        cofire = {}
        for b in bin_keys:
            neurons = list(bins[b])
            if len(neurons) < 2 or len(neurons) > 50:
                continue  # skip empty or huge bins (noise)
            for i in range(len(neurons)):
                for j in range(i + 1, len(neurons)):
                    pair = (min(neurons[i], neurons[j]), max(neurons[i], neurons[j]))
                    cofire[pair] = cofire.get(pair, 0) + 1

        # Filter by minimum co-firing AND spatial proximity
        max_d2 = max_distance * max_distance
        candidates = []
        for pair, count in cofire.items():
            if count < min_cofire:
                continue
            n1, n2 = pair
            dx = pos_x[n1] - pos_x[n2]
            dy = pos_y[n1] - pos_y[n2]
            dz = pos_z[n1] - pos_z[n2]
            d2 = dx*dx + dy*dy + dz*dz
            if d2 <= max_d2:
                candidates.append((count, d2, pair))

        # Sort by co-firing count (primary), then proximity (secondary)
        candidates.sort(key=lambda x: (-x[0], x[1]))

        if not candidates:
            return {'sprouted': 0, 'candidates': 0}

        # Build existing connection set
        existing = set()
        for syn in self.synapses:
            existing.add((syn['source'], syn['target']))
            existing.add((syn['target'], syn['source']))
        for gj in self.gap_junctions:
            existing.add((gj['source'], gj['target']))
            existing.add((gj['target'], gj['source']))

        # Create new synapses for unconnected co-active nearby pairs
        new_syns = []
        neuron_db_ids = [n['id'] for n in self.neurons]
        max_syn_id = max((s['id'] for s in self.synapses), default=0)

        for count, d2, (n1, n2) in candidates:
            if len(new_syns) >= max_new:
                break
            if (n1, n2) not in existing:
                max_syn_id += 1
                syn = {
                    'id': max_syn_id,
                    'source': n1,
                    'target': n2,
                    'source_db_id': neuron_db_ids[n1],
                    'target_db_id': neuron_db_ids[n2],
                    'type': 'plastic',
                    'module': plastic_module,
                    'weight': weight,
                    'delay': 1,
                }
                syn.update(dict(plastic_module.DEFAULTS))
                syn.update(dict(plastic_module.INITIAL_STATE))
                new_syns.append(syn)
                existing.add((n1, n2))

        if not new_syns:
            return {'sprouted': 0, 'candidates': len(candidates)}

        # Add to synapse list and rebuild structures
        self.synapses.extend(new_syns)
        self.data['synapses'] = self.synapses
        self._build_synapse_structures()

        return {
            'sprouted': len(new_syns),
            'candidates': len(candidates),
            'top_cofire': candidates[0][0] if candidates else 0,
        }

    def drift(self, drift_rate=0.05, min_ticks=5000, silent_threshold=0.001):
        """Parameter drift: nudge underperforming neurons toward useful configs.

        Biology: neurons can't switch type. But we're synthetic -- we can
        engineer a developmental mechanism that biology never found.

        Identifies "silent" neurons (firing rate far below average) and
        nudges their (a, b, c, d) parameters toward the average of their
        active connected neighbors. Small steps, reversible, continuous.

        NOT type-switching. A gradual drift that lets a misplaced neuron
        become something its local circuit actually needs.

        Args:
            drift_rate: how much to nudge per call (0.05 = 5% toward target)
            min_ticks: minimum ticks elapsed before drift is applied
            silent_threshold: firing rate below this = "silent" (spikes/tick)

        Returns:
            dict with drift stats
        """
        if self.tick_count < min_ticks:
            return {'drifted': 0, 'silent': 0}

        # Count spikes per neuron from recorder
        spike_counts = np.zeros(self.n, dtype=np.int64)
        for _, nid in self.recorder.spikes:
            spike_counts[nid] += 1
        rates = spike_counts / max(self.tick_count, 1)

        # Find silent neurons
        silent_mask = rates < silent_threshold
        n_silent = int(silent_mask.sum())
        if n_silent == 0:
            return {'drifted': 0, 'silent': 0}

        # Build neighbor map: for each neuron, who sends it spikes?
        # (only from active neurons -- we want to drift TOWARD useful params)
        active_mask = rates >= silent_threshold
        neighbor_params = {}  # neuron_idx -> list of (a,b,c,d) from active sources

        for syn in self.synapses:
            src, tgt = syn['source'], syn['target']
            if silent_mask[tgt] and active_mask[src]:
                if tgt not in neighbor_params:
                    neighbor_params[tgt] = []
                neighbor_params[tgt].append((
                    self.a[src], self.b[src], self.c[src], self.d[src]
                ))

        # Drift silent neurons toward their active neighbors' average params
        drifted = 0
        for nid in np.flatnonzero(silent_mask):
            neighbors = neighbor_params.get(int(nid))
            if not neighbors or len(neighbors) < 2:
                continue  # no active neighbors to learn from

            # Average neighbor params
            avg_a = sum(p[0] for p in neighbors) / len(neighbors)
            avg_b = sum(p[1] for p in neighbors) / len(neighbors)
            avg_c = sum(p[2] for p in neighbors) / len(neighbors)
            avg_d = sum(p[3] for p in neighbors) / len(neighbors)

            # Nudge toward average (small step)
            self.a[nid] += drift_rate * (avg_a - self.a[nid])
            self.b[nid] += drift_rate * (avg_b - self.b[nid])
            self.c[nid] += drift_rate * (avg_c - self.c[nid])
            self.d[nid] += drift_rate * (avg_d - self.d[nid])
            drifted += 1

        return {
            'drifted': drifted,
            'silent': n_silent,
            'total': self.n,
            'silent_pct': n_silent / self.n * 100,
        }

    def sleep(self, ticks, compression=0.8, noise_amplitude=2.0, seed=None):
        """Sleep phase: noise replay + synaptic homeostasis.

        Biology: during sleep, spontaneous activity replays recent patterns
        while synaptic downscaling prevents runaway weight growth.

        Phase 1: Run brain with Gaussian noise (replay).
          - Regular STDP on plastic/developmental synapses continues
          - No reward delivery -> reward_plastic weights frozen during replay
          - Strong patterns get reinforced via coincident replay

        Phase 2: Power-law compression on reward_plastic weights.
          - w_new = w_init * (w / w_init)^compression
          - Pulls weights toward initial value while preserving relative order
          - compression=0.8 -> keep 80% of learned deviation (log-space)
          - compression=1.0 -> no change, compression=0.5 -> aggressive

        Ref: Sleep-based homeostatic regularization (arxiv 2601.08447)
        """
        if ticks <= 0 and (not self.has_reward or compression >= 1.0):
            return {}

        rng = np.random.RandomState(seed or 0)

        # Phase 1: Noise replay (consolidates plastic/developmental STDP)
        replay_spikes = 0
        if ticks > 0:
            noise = np.zeros(self.n, dtype=np.float64)
            for t in range(ticks):
                noise[:] = rng.randn(self.n) * noise_amplitude
                fired = self.tick(external_I=noise)
                replay_spikes += len(fired)

        # Phase 2: Power-law compression on reward weights
        n_compressed = 0
        pre_avg = 0.0
        post_avg = 0.0
        if self.has_reward and compression < 1.0:
            w_init = 1.0  # All reward_plastic start at 1.0 (schema default)
            pre_total = 0.0
            post_total = 0.0
            nr = len(self.reward_idx)
            for ri, si in enumerate(self.reward_idx):
                syn = self.synapses[si]
                w = syn['weight']
                pre_total += w
                if w <= 0 or abs(w - w_init) < 1e-6:
                    post_total += w
                    continue
                # Power-law: preserves ordering, compresses toward w_init
                w_new = w_init * (w / w_init) ** compression
                syn['weight'] = max(syn['w_min'], min(syn['w_max'], w_new))
                post_total += syn['weight']
                n_compressed += 1
            if nr > 0:
                pre_avg = pre_total / nr
                post_avg = post_total / nr

        # Phase 3: Power-law compression on plastic (STDP) weights
        n_plastic_compressed = 0
        pre_plastic_avg = 0.0
        post_plastic_avg = 0.0
        np_plastic = len(self.plastic_idx)
        if np_plastic > 0 and compression < 1.0:
            p_init = 2.0  # Plastic synapses start at 2.0 (recipe default)
            pre_p_total = 0.0
            post_p_total = 0.0
            for pi, si in enumerate(self.plastic_idx):
                syn = self.synapses[si]
                w = syn['weight']
                pre_p_total += w
                if w <= 0 or abs(w - p_init) < 1e-6:
                    post_p_total += w
                    continue
                w_new = p_init * (w / p_init) ** compression
                syn['weight'] = max(syn['w_min'], min(syn['w_max'], w_new))
                post_p_total += syn['weight']
                n_plastic_compressed += 1
            if np_plastic > 0:
                pre_plastic_avg = pre_p_total / np_plastic
                post_plastic_avg = post_p_total / np_plastic

        # Phase 4: Synaptogenesis (grow new connections from co-firing)
        sprout_result = self.sprout(max_new=50, window=max(ticks, 2000), seed=seed)

        # Phase 5: Parameter drift (nudge silent neurons toward useful configs)
        drift_result = self.drift()

        result = {
            'replay_ticks': ticks,
            'replay_spikes': replay_spikes,
            'compressed': n_compressed,
            'pre_avg_w': pre_avg,
            'post_avg_w': post_avg,
            'plastic_compressed': n_plastic_compressed,
            'pre_plastic_avg': pre_plastic_avg,
            'post_plastic_avg': post_plastic_avg,
            'sprouted': sprout_result.get('sprouted', 0),
            'sprout_candidates': sprout_result.get('candidates', 0),
            'drifted': drift_result.get('drifted', 0),
            'silent': drift_result.get('silent', 0),
            'silent_pct': drift_result.get('silent_pct', 0),
        }
        return result

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
