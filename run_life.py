"""
run_life.py -- Give a brain a life, not a test.

Chains multiple arena sessions with different starting positions.
Each session is a new "day" - different place, same brain, accumulating experience.
Between sessions, optional "sleep" compresses weights and replays patterns.
The brain saves after each session, preserving everything it learned.

Usage:
  py run_life.py --brain arena_dev_v2_life_s42.db --sessions 10
  py run_life.py --brain arena_dev_v2_life_s42.db --sessions 50 --ticks 50000
  py run_life.py --brain arena_dev_v2_life_s42.db --sessions 10 --sleep 2000
"""
import os, sys, time, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from run_arena import run_arena


def run_life(db_name, n_sessions=10, ticks_per=100000, sensory_gain=5.0,
             reward_magnitude=1.0, start_seed=1, sleep_ticks=0,
             sleep_compression=0.8, surprise_reward=False,
             surprise_alpha=0.25, reward_homeostasis=False):
    """Chain multiple arena sessions with varied starting positions."""

    sleep_str = f", sleep={sleep_ticks} ticks (comp={sleep_compression})" if sleep_ticks > 0 else ""
    print("=" * 70)
    print(f"  LIFE: {db_name}")
    print(f"  {n_sessions} sessions x {ticks_per:,} ticks = {n_sessions * ticks_per:,} total")
    print(f"  Seeds {start_seed} to {start_seed + n_sessions - 1} (varied starts)")
    if sleep_ticks > 0:
        print(f"  Sleep: {sleep_ticks:,} ticks between sessions, compression={sleep_compression}")
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
                          report_interval=10000, surprise_reward=surprise_reward,
                          surprise_alpha=surprise_alpha,
                          reward_homeostasis=reward_homeostasis)

        if result and 'brain' in result:
            brain = result['brain']

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

            # Get reward weight stats (pre-sleep)
            rw = [s['weight'] for s in brain.synapses
                  if s['type'] == 'reward_plastic']
            if rw:
                summary['avg_w'] = np.mean(rw)
                summary['std_w'] = np.std(rw)
                summary['min_w'] = np.min(rw)
                summary['max_w'] = np.max(rw)

            # Sleep between sessions (not after the last one)
            if sleep_ticks > 0 and session < n_sessions - 1:
                print(f"\n  --- SLEEP ({sleep_ticks:,} ticks, compression={sleep_compression}) ---")
                t_sleep = time.time()
                sleep_result = brain.sleep(
                    ticks=sleep_ticks,
                    compression=sleep_compression,
                    seed=seed + 1000  # different noise per sleep
                )
                sleep_elapsed = time.time() - t_sleep
                sr = sleep_result
                print(f"  Sleep: {sr['replay_spikes']:,} replay spikes in {sleep_elapsed:.1f}s")
                print(f"  Reward compression: avg_w {sr['pre_avg_w']:.3f} -> {sr['post_avg_w']:.3f} "
                      f"({sr['compressed']} synapses)")
                if sr.get('plastic_compressed', 0) > 0:
                    print(f"  Plastic compression: avg_w {sr['pre_plastic_avg']:.3f} -> "
                          f"{sr['post_plastic_avg']:.3f} ({sr['plastic_compressed']} synapses)")
                if sr.get('sprouted', 0) > 0:
                    print(f"  Synaptogenesis: {sr['sprouted']} new synapses "
                          f"(from {sr['sprout_candidates']} candidates)")
                if sr.get('drifted', 0) > 0:
                    print(f"  Parameter drift: {sr['drifted']} neurons nudged "
                          f"({sr['silent']} silent, {sr['silent_pct']:.1f}% of brain)")

                # Post-sleep weight stats
                rw_post = [s['weight'] for s in brain.synapses
                           if s['type'] == 'reward_plastic']
                if rw_post:
                    summary['post_sleep_avg_w'] = np.mean(rw_post)
                    summary['post_sleep_std_w'] = np.std(rw_post)

            brain.save()
            history.append(summary)
            print(f"\n  Session {session+1} saved. avg_w={summary.get('avg_w',0):.3f} "
                  f"std_w={summary.get('std_w',0):.3f}")

    elapsed = time.time() - t0
    total_ticks = n_sessions * ticks_per

    # Summary
    print(f"\n{'='*70}")
    print(f"  LIFE SUMMARY: {n_sessions} sessions, {total_ticks:,} ticks in {elapsed:.0f}s")
    print(f"{'='*70}")
    has_sleep = any('post_sleep_avg_w' in h for h in history)
    if has_sleep:
        print(f"  {'Sess':>4s} | {'Seed':>4s} | {'dFood':>6s} | {'Min':>5s} | {'Reward':>7s} | "
              f"{'Disp':>5s} | {'avg_w':>6s} | {'->slp':>6s} | {'std_w':>6s}")
    else:
        print(f"  {'Sess':>4s} | {'Seed':>4s} | {'dFood':>6s} | {'Min':>5s} | {'Reward':>7s} | "
              f"{'Disp':>5s} | {'avg_w':>6s} | {'std_w':>6s}")
    print(f"  {'-'*70}")
    for h in history:
        line = (f"  {h['session']:4d} | {h['seed']:4d} | {h['d_final']:6.1f} | "
                f"{h['min_dist']:5.1f} | {h['net_reward']:+7.1f} | "
                f"{h['displacement']:5.1f} | {h.get('avg_w',0):6.3f} | ")
        if has_sleep:
            line += f"{h.get('post_sleep_avg_w', h.get('avg_w',0)):6.3f} | "
        line += f"{h.get('std_w',0):6.3f}"
        print(line)

    # Weight distribution evolution
    if history:
        first_w = history[0].get('avg_w', 0)
        last_w = history[-1].get('avg_w', 0)
        first_std = history[0].get('std_w', 0)
        last_std = history[-1].get('std_w', 0)
        print(f"\n  Weight trajectory: {first_w:.3f} -> {last_w:.3f} "
              f"(std: {first_std:.3f} -> {last_std:.3f})")
        if has_sleep:
            last_post = history[-1].get('post_sleep_avg_w', last_w)
            print(f"  Post-sleep final: {last_post:.3f}")

    return history


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--brain', default='arena_dev_v2_life_s42.db')
    p.add_argument('--sessions', type=int, default=10)
    p.add_argument('--ticks', type=int, default=100000)
    p.add_argument('--sensory-gain', type=float, default=5.0)
    p.add_argument('--reward-mag', type=float, default=1.0)
    p.add_argument('--start-seed', type=int, default=1)
    p.add_argument('--sleep', type=int, default=0,
                   help='Ticks of sleep between sessions (0=disabled)')
    p.add_argument('--sleep-compression', type=float, default=0.8,
                   help='Power-law compression factor (0.8=keep 80%% of deviation)')
    p.add_argument('--surprise-reward', action='store_true',
                   help='Use prediction-error surprise gating for reward')
    p.add_argument('--surprise-alpha', type=float, default=0.25,
                   help='EMA alpha for surprise expectation')
    p.add_argument('--reward-homeostasis', action='store_true',
                   help='Enable during-reward homeostasis (default: OFF, use sleep instead)')
    args = p.parse_args()

    run_life(args.brain, n_sessions=args.sessions, ticks_per=args.ticks,
             sensory_gain=args.sensory_gain, reward_magnitude=args.reward_mag,
             start_seed=args.start_seed, sleep_ticks=args.sleep,
             sleep_compression=args.sleep_compression,
             surprise_reward=args.surprise_reward,
             surprise_alpha=args.surprise_alpha,
             reward_homeostasis=args.reward_homeostasis)
