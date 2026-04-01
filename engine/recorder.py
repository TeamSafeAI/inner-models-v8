"""
recorder.py — Records what the brain actually does.

No assertions, no pass/fail. Just honest measurement.
Attach to a running brain, it captures spikes, weights, voltages.
"""


class Recorder:
    """Records brain activity during simulation."""

    def __init__(self, brain):
        self.n_neurons = len(brain['neurons'])
        self.n_synapses = len(brain['synapses'])
        self.tick = 0

        # Spike log: list of (tick, neuron_index) tuples
        self.spikes = []

        # Per-interval stats
        self.interval = 1000  # ticks per reporting interval
        self.interval_spikes = 0
        self.intervals = []  # list of (tick, spike_count, rate) tuples

        # Weight snapshots (optional, call snapshot_weights() to capture)
        self.weight_snapshots = []

    def record_tick(self, fired_indices):
        """Called each tick with list of neuron indices that fired."""
        for idx in fired_indices:
            self.spikes.append((self.tick, idx))

        self.interval_spikes += len(fired_indices)

        if (self.tick + 1) % self.interval == 0:
            rate = self.interval_spikes / (self.n_neurons * self.interval)
            self.intervals.append((self.tick + 1, self.interval_spikes, rate))
            self.interval_spikes = 0

        self.tick += 1

    def snapshot_weights(self, synapses):
        """Capture current synapse weights."""
        weights = [(s['id'], s['type'], s['weight']) for s in synapses]
        self.weight_snapshots.append((self.tick, weights))

    def report(self):
        """Print summary of what happened."""
        total = len(self.spikes)
        ticks = self.tick
        n = self.n_neurons

        print(f"\n=== Recording: {ticks} ticks, {n} neurons ===")
        print(f"  Total spikes: {total}")
        if ticks > 0 and n > 0:
            print(f"  Avg firing rate: {total / (n * ticks):.4f}")

        if self.intervals:
            print(f"\n  Per-{self.interval} tick intervals:")
            for tick, count, rate in self.intervals:
                print(f"    t={tick:7d}  spikes={count:6d}  rate={rate:.4f}")

        if self.weight_snapshots:
            print(f"\n  Weight snapshots: {len(self.weight_snapshots)}")
            for tick, weights in self.weight_snapshots:
                inh = [w for _, t, w in weights if w < 0]
                exc = [w for _, t, w in weights if w > 0]
                inh_str = f"inh: mean={sum(inh)/len(inh):.2f}" if inh else "inh: none"
                exc_str = f"exc: mean={sum(exc)/len(exc):.2f}" if exc else "exc: none"
                print(f"    t={tick}: {inh_str}, {exc_str}")

    def firing_rates_by_type(self, brain):
        """Report firing rates grouped by neuron type."""
        ticks = self.tick
        if ticks == 0:
            return

        # Count spikes per neuron
        counts = [0] * self.n_neurons
        for tick, idx in self.spikes:
            counts[idx] += 1

        # Group by type
        by_type = {}
        for i, n in enumerate(brain['neurons']):
            t = n['type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(counts[i])

        print(f"\n  Firing rates by neuron type ({ticks} ticks):")
        for t in sorted(by_type.keys()):
            rates = [c / ticks for c in by_type[t]]
            avg = sum(rates) / len(rates) if rates else 0
            active = sum(1 for r in rates if r > 0)
            print(f"    {t}: {len(rates)} neurons, {active} active, avg rate={avg:.4f}")
