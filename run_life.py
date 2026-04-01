"""
run_life.py -- Give a brain a life, not a test.

Chains multiple arena sessions with different starting positions.
Each session is a new "day" - different place, same brain, accumulating experience.
The brain saves after each session, preserving everything it learned.

Usage:
  py run_life.py --brain arena_dev_v2_life_s42.db --sessions 10
  py run_life.py --brain arena_dev_v2_life_s42.db --sessions 50 --ticks 50000
"""
import os, sys, time, argparse, sqlite3
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from run_arena import run_arena


def run_life(db_name, n_sessions=10, ticks_per=100000, sensory_gain=5.0,
             reward_magnitude=1.0, start_seed=1):
    """Chain multiple arena sessions with varied starting positions."""

    print("=" * 70)
    print(f"  LIFE: {db_name}")
    print(f"  {n_sessions} sessions x {ticks_per:,} ticks = {n_sessions * ticks_per:,} total")
    print(f"  Seeds {start_seed} to {start_seed + n_sessions - 1} (varied starts)")
    print("=" * 70)

    t0 = time.time()
    history = []

    for session in range(n_sessions):
        seed = start_seed + session
        print(f"\n{'='*70}")
        print(f"  SESSION {session + 1}/{n_sessions} (seed={seed})")
        print(f"{'='*70}")

        result = run_arena(db_name, ticks=ticks_per, learn=True, seed=seed,
                          sensory_gain=sensory_gain, reward_magnitude=reward_magnitude,
                          report_interval=10000)

        if result and 'brain' in result:
            result['brain'].save()

            # Collect session summary
            summary = {
                'session': session + 1,
                'seed': seed,
                'd_final': result.get('d_final', 0),
                'd0': result.get('d0', 0),
                'min_dist': result.get('min_distance', 0),
                'net_reward': result.get('net_reward', 0),
                'displacement': result.get('displacement', 0),
                'motor_spikes': result.get('motor_spikes', 0),
                'sensory_spikes': result.get('sensory_spikes', 0),
            }

            # Get reward weight stats
            rw = [s['weight'] for s in result['brain'].synapses
                  if s['type'] == 'reward_plastic']
            if rw:
                summary['avg_w'] = np.mean(rw)
                summary['std_w'] = np.std(rw)
                summary['min_w'] = np.min(rw)
                summary['max_w'] = np.max(rw)

            history.append(summary)
            print(f"\n  Session {session+1} saved. avg_w={summary.get('avg_w',0):.3f} "
                  f"std_w={summary.get('std_w',0):.3f}")

    elapsed = time.time() - t0
    total_ticks = n_sessions * ticks_per

    # Summary
    print(f"\n{'='*70}")
    print(f"  LIFE SUMMARY: {n_sessions} sessions, {total_ticks:,} ticks in {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"  {'Sess':>4s} | {'Seed':>4s} | {'dFood':>6s} | {'Min':>5s} | {'Reward':>7s} | "
          f"{'Disp':>5s} | {'avg_w':>6s} | {'std_w':>6s}")
    print(f"  {'-'*65}")
    for h in history:
        print(f"  {h['session']:4d} | {h['seed']:4d} | {h['d_final']:6.1f} | "
              f"{h['min_dist']:5.1f} | {h['net_reward']:+7.1f} | "
              f"{h['displacement']:5.1f} | {h.get('avg_w',0):6.3f} | "
              f"{h.get('std_w',0):6.3f}")

    # Weight distribution evolution
    if history:
        print(f"\n  Weight trajectory: "
              f"{history[0].get('avg_w',0):.3f} -> {history[-1].get('avg_w',0):.3f} "
              f"(std: {history[0].get('std_w',0):.3f} -> {history[-1].get('std_w',0):.3f})")

    return history


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--brain', default='arena_dev_v2_life_s42.db')
    p.add_argument('--sessions', type=int, default=10)
    p.add_argument('--ticks', type=int, default=100000)
    p.add_argument('--sensory-gain', type=float, default=5.0)
    p.add_argument('--reward-mag', type=float, default=1.0)
    p.add_argument('--start-seed', type=int, default=1)
    args = p.parse_args()

    run_life(args.brain, n_sessions=args.sessions, ticks_per=args.ticks,
             sensory_gain=args.sensory_gain, reward_magnitude=args.reward_mag,
             start_seed=args.start_seed)
