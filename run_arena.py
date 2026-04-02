"""
run_arena.py -- Full brain-body-environment chemotaxis with v8 engine.

Real WormBody (24-segment rigid chain, anisotropic drag, proprioceptive feedback).
Real Arena (Gaussian food gradients, circular boundary).
v8 Brain (NumPy-optimized, reward_plastic for 3-factor learning).

Brain-body-environment loop (1ms per tick):
  1. Arena: concentration at head position -> sensory current
  2. Body: bend angles -> proprioceptive feedback to motor neurons
  3. Brain: tick (includes STDP + reward_plastic)
  4. Motor: neuron spikes -> body muscle activation
  5. Body: physics step (spring+damper joints, anisotropic drag)
  6. Arena: boundary clamp, trajectory record
  7. Reward: deliver_reward based on concentration change

Usage:
  py run_arena.py                           # Build + run
  py run_arena.py --brain arena_v1_s42.db
  py run_arena.py --ticks 120000            # 2 minutes sim time
  py run_arena.py --compare                 # Learn vs no-learn
"""
import numpy as np, time, sys, os, argparse

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from engine.loader import load
from engine.runner import Brain
from worm_body import WormBody, N_JOINTS, N_SEGMENTS, PROPRIO_GAIN, PROPRIO_LOCAL_FRAC, PROPRIO_ANT_FRAC
from arena import Arena


def run_arena(db_name, ticks=60000, learn=True, seed=42,
              sensory_gain=5.0, reward_magnitude=1.0,
              report_interval=5000, reward_interval=200,
              start_distance=20.0, surprise_reward=False,
              surprise_alpha=0.25, dishbrain_feedback=False,
              reward_homeostasis=False):
    """Full brain-body-environment chemotaxis loop.

    Args:
        db_name: brain DB filename in brains/zoo/
        ticks: simulation ticks (1 tick = 1ms)
        learn: enable STDP + reward_plastic
        seed: random seed
        sensory_gain: current per unit concentration to sensory neurons
        reward_magnitude: reward signal strength for deliver_reward()
        report_interval: ticks between progress reports
        reward_interval: ticks between reward delivery
        start_distance: starting distance from food
    """
    # Accept both "arena_v1_s42.db" and "brains/zoo/arena_v1_s42.db"
    db_path = os.path.join(BASE, 'brains', 'zoo', os.path.basename(db_name))
    if not os.path.exists(db_path):
        print(f"  Brain not found: {db_path}")
        print(f"  Run: py build_arena_brain.py")
        return None

    print(f"{'='*70}")
    print(f"  ARENA CHEMOTAXIS: {db_name}")
    print(f"  learn={learn}, ticks={ticks:,d}, sensory_gain={sensory_gain}")
    print(f"{'='*70}")

    # Load brain
    data = load(db_path)
    brain = Brain(data, learn=learn, reward_homeostasis=reward_homeostasis)
    n = brain.n
    body_map = data.get('body_map', {})
    sensor_map = data.get('sensor_map', {})

    # Classify sensors
    chemical_sensors = []
    mechanical_head = []
    mechanical_tail = []
    left_sensors = []
    right_sensors = []

    for idx, entry in sensor_map.items():
        if entry['modality'] == 'chemical':
            chemical_sensors.append(idx)
            if entry.get('side') == 'left':
                left_sensors.append(idx)
            elif entry.get('side') == 'right':
                right_sensors.append(idx)
        elif entry['modality'] == 'mechanical':
            if entry['location'] == 'head':
                mechanical_head.append(idx)
            else:
                mechanical_tail.append(idx)

    # Build proprio targets (motor neurons with excitatory body_map entries)
    proprio_targets = {}
    for idx, entry in body_map.items():
        if entry['effect'] == 'excitatory':
            proprio_targets[idx] = entry['segment']

    # Count synapse types
    n_reward = sum(1 for s in brain.synapses if s['type'] == 'reward_plastic')
    n_plastic = sum(1 for s in brain.synapses if s['type'] == 'plastic')

    print(f"  Brain: {n}N, {len(brain.synapses)} synapses")
    print(f"  Chemical sensors: {len(chemical_sensors)} (L:{len(left_sensors)} R:{len(right_sensors)})")
    print(f"  Mechanical sensors: {len(mechanical_head)} head + {len(mechanical_tail)} tail")
    print(f"  Body map: {len(body_map)} motor neurons, {len(proprio_targets)} proprio targets")
    print(f"  Reward plastic: {n_reward}, regular plastic: {n_plastic}")

    # Arena: food at (15, 0)
    arena = Arena(radius=30.0)
    arena.add_food(15.0, 0.0, peak=1.0, sigma=12.0)
    food_x, food_y = 15.0, 0.0

    # Body: start at specified distance from food
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, 2 * np.pi)
    start_x = food_x + start_distance * np.cos(angle)
    start_y = food_y + start_distance * np.sin(angle)
    start_heading = rng.uniform(0, 2 * np.pi)

    body = WormBody()
    body.pos = np.array([start_x, start_y], dtype=np.float64)
    body.heading = start_heading

    # Clamp to arena if outside
    body.pos = arena.clamp_to_boundary(body.pos)

    d0 = np.sqrt((body.pos[0] - food_x)**2 + (body.pos[1] - food_y)**2)
    c0 = arena.concentration_at(body.pos[0], body.pos[1])

    print(f"  Start: ({body.pos[0]:.1f}, {body.pos[1]:.1f}), heading: {np.degrees(body.heading):.0f} deg")
    print(f"  Distance to food: {d0:.1f}, initial conc: {c0:.4f}")

    # Adaptation state for derivative sensing
    adapted_conc = c0
    tau_adapt = 500.0  # ms

    # Tracking
    trajectory = []
    conc_log = []
    dist_log = []
    total_spikes = 0
    motor_spikes = 0
    sensory_spikes = 0
    reward_total = 0.0
    n_rewards = 0

    # Reward weight snapshots
    reward_w0 = [brain.synapses[si]['weight'] for si in brain.reward_idx] \
        if brain.has_reward else []

    prev_conc = c0
    expected_dc = 0.0  # For surprise-gated reward
    chaos_remaining = 0  # DishBrain-style chaos countdown

    print(f"\n  {'Tick':>8s} | {'Conc':>6s} | {'dFood':>6s} | {'Motor':>6s} | "
          f"{'Sense':>6s} | {'Reward':>7s} | {'pos':>20s}")
    print(f"  {'-'*80}")

    start_time = time.perf_counter()

    for tick in range(ticks):
        # 1. Sensory input: concentration at head
        head = body.pos
        heading = body.heading
        conc = arena.concentration_at(head[0], head[1])

        # Derivative sensing (dC/dt adaptation)
        alpha = 1.0 / tau_adapt
        delta_conc = conc - adapted_conc
        adapted_conc += alpha * (conc - adapted_conc)

        # Build external current
        sensory_I = np.zeros(n)

        # Chemical sensors: absolute + derivative
        for idx in chemical_sensors:
            sensory_I[idx] = conc * sensory_gain + max(0, delta_conc) * sensory_gain * 3.0

        # DishBrain chaos: inject random noise to ALL neurons (not just sensors)
        # Real DishBrain: structured input = good, random input = bad
        # Chaos covers entire brain so motor/decision neurons feel the disorder
        if dishbrain_feedback and chaos_remaining > 0:
            sensory_I += rng.randn(n) * 5.0
            chaos_remaining -= 1

        # Spatial sensing: left/right amphids sample different positions
        head_half_width = 0.04
        normal_x = -np.sin(heading)
        normal_y = np.cos(heading)
        conc_left = arena.concentration_at(
            head[0] + normal_x * head_half_width,
            head[1] + normal_y * head_half_width)
        conc_right = arena.concentration_at(
            head[0] - normal_x * head_half_width,
            head[1] - normal_y * head_half_width)

        for idx in left_sensors:
            sensory_I[idx] += conc_left * sensory_gain * 0.5
        for idx in right_sensors:
            sensory_I[idx] += conc_right * sensory_gain * 0.5

        # Wall detection -> mechanical head sensors
        d_wall = arena.radius - np.linalg.norm(head)
        if d_wall < 2.0:
            wall_strength = (2.0 - d_wall) / 2.0 * 15.0  # up to 15 uA at wall
            for idx in mechanical_head:
                sensory_I[idx] += wall_strength

        # 2. Proprioceptive feedback: body bends -> motor neuron current
        proprio_I = np.zeros(n)
        angles = body.get_bend_angles()

        for idx, seg in proprio_targets.items():
            entry = body_map[idx]

            # Local restorative feedback
            j_local = min(seg, N_JOINTS - 1)
            local_gain = PROPRIO_GAIN * PROPRIO_LOCAL_FRAC
            if entry['side'] == 'dorsal':
                proprio_I[idx] += -angles[j_local] * local_gain
            else:
                proprio_I[idx] += angles[j_local] * local_gain

            # Anterior wave-propagating feedback
            weighted_angle = 0.0
            total_weight = 0.0
            for offset in range(1, 4):
                j = seg - offset
                if j < 0 or j >= N_JOINTS:
                    break
                w = 1.0 / offset
                weighted_angle += angles[j] * w
                total_weight += w
            if total_weight > 0:
                weighted_angle /= total_weight
                ant_gain = PROPRIO_GAIN * PROPRIO_ANT_FRAC
                if entry['side'] == 'dorsal':
                    proprio_I[idx] += weighted_angle * ant_gain
                else:
                    proprio_I[idx] += -weighted_angle * ant_gain

        # 3. Combine currents + tonic baseline
        I_ext = np.full(n, 2.5)  # tonic baseline
        I_ext += sensory_I
        I_ext += proprio_I

        # 4. Brain tick
        fired = brain.tick(external_I=I_ext)
        total_spikes += len(fired)

        # 5. Motor spikes -> body
        for fi in fired:
            if fi in body_map:
                entry = body_map[fi]
                body.apply_motor_spike(entry['segment'], entry['side'], entry['effect'])
                motor_spikes += 1
            if fi in sensor_map:
                sensory_spikes += 1

        # 6. Body physics step
        body.step(dt=0.001)

        # 7. Arena boundary
        body.pos = arena.clamp_to_boundary(body.pos)

        # 8. Reward delivery
        if tick % reward_interval == reward_interval - 1:
            new_conc = arena.concentration_at(body.pos[0], body.pos[1])
            dc = new_conc - prev_conc

            if surprise_reward:
                # Surprise-gated: only reward when outcome differs from expectation
                # Mimics dopamine prediction error (Schultz 1997, TD learning)
                prediction_error = dc - expected_dc
                expected_dc += surprise_alpha * (dc - expected_dc)

                if abs(prediction_error) > 0.0005:
                    # Scale reward by surprise magnitude, clamp to bounds
                    reward = float(np.clip(
                        prediction_error * 5.0,
                        -reward_magnitude, reward_magnitude))
                    brain.deliver_reward(reward)
                    reward_total += reward
                    n_rewards += 1
            else:
                # Original: binary reward based on concentration change
                if abs(dc) > 0.0001:
                    reward = np.sign(dc) * reward_magnitude
                    brain.deliver_reward(reward)
                    reward_total += reward
                    n_rewards += 1

            # Wall punishment (always active)
            d_wall = arena.radius - np.linalg.norm(body.pos)
            if d_wall < 1.0:
                brain.deliver_reward(-reward_magnitude * 0.5)
                reward_total -= reward_magnitude * 0.5
                n_rewards += 1

            prev_conc = new_conc

            # DishBrain-style: inject chaos on negative outcome
            # Clean sensory = structured (approaching food)
            # Noise blast = chaotic (moving away from food)
            if dishbrain_feedback and dc < -0.0001:
                chaos_remaining = reward_interval

        # 9. Record trajectory
        if tick % 100 == 0:
            trajectory.append(body.pos.copy())
            conc_log.append(conc)
            d = np.sqrt((body.pos[0] - food_x)**2 + (body.pos[1] - food_y)**2)
            dist_log.append(d)

        # 10. Report
        if tick % report_interval == report_interval - 1:
            d = np.sqrt((body.pos[0] - food_x)**2 + (body.pos[1] - food_y)**2)
            elapsed = time.perf_counter() - start_time
            tps = (tick + 1) / elapsed
            print(f"  {tick+1:8,d} | {conc:6.4f} | {d:6.1f} | {motor_spikes:6d} | "
                  f"{sensory_spikes:6d} | {reward_total:+7.1f} | "
                  f"({body.pos[0]:7.2f},{body.pos[1]:7.2f})  [{tps:.0f} t/s]")

    # ── Final report ──
    elapsed = time.perf_counter() - start_time
    d_final = np.sqrt((body.pos[0] - food_x)**2 + (body.pos[1] - food_y)**2)

    print(f"\n{'='*70}")
    print(f"  RESULTS: {ticks:,d} ticks in {elapsed:.1f}s ({ticks/elapsed:.0f} t/s)")
    print(f"  Position: ({body.pos[0]:.2f}, {body.pos[1]:.2f})")
    print(f"  Distance to food: {d0:.1f} -> {d_final:.1f} (change: {d0-d_final:+.1f})")

    if len(dist_log) > 10:
        ci = (dist_log[0] - dist_log[-1]) / max(dist_log[0], 0.1)
        min_d = min(dist_log)
        mean_d = np.mean(dist_log)
        print(f"  Chemotaxis index: {ci:+.4f}")
        print(f"  Min distance: {min_d:.1f}, Mean distance: {mean_d:.1f}")

    print(f"  Spikes: {total_spikes:,d} total, {motor_spikes:,d} motor, {sensory_spikes:,d} sensory")
    print(f"  Rewards delivered: {n_rewards}, net reward: {reward_total:+.1f}")

    # Displacement from body undulation
    disp = float(np.linalg.norm(body.pos - np.array([start_x, start_y])))
    print(f"  Total displacement: {disp:.2f} BL")
    print(f"  Max curvature: {float(np.max(np.abs(body.get_curvature()))):.2f}")

    # Reward weight changes
    if brain.has_reward and len(reward_w0) > 0:
        reward_wf = [brain.synapses[si]['weight'] for si in brain.reward_idx]
        changed = sum(1 for w0, w1 in zip(reward_w0, reward_wf) if abs(w1 - w0) > 0.001)
        maxed = sum(1 for w in reward_wf if w >= 9.9)
        zeroed = sum(1 for w in reward_wf if w < 0.1)
        avg_w = np.mean(reward_wf)
        print(f"\n  Reward-plastic: {len(reward_wf)} synapses")
        print(f"  Changed: {changed}, Maxed: {maxed}, Zeroed: {zeroed}, avg_w: {avg_w:.3f}")

    return {
        'trajectory': np.array(trajectory) if trajectory else np.zeros((0, 2)),
        'distances': np.array(dist_log),
        'concentrations': np.array(conc_log),
        'd0': d0,
        'd_final': d_final,
        'total_spikes': total_spikes,
        'motor_spikes': motor_spikes,
        'sensory_spikes': sensory_spikes,
        'reward_total': reward_total,
        'displacement': disp,
        'brain': brain,
    }


def compare(db_name, ticks=60000, seed=42):
    """Run with and without reward learning, compare."""
    print("\n" + "=" * 70)
    print("  COMPARISON: Reward Learning vs Control")
    print("=" * 70)

    r_learn = run_arena(db_name, ticks=ticks, learn=True, seed=seed)
    print()
    r_ctrl = run_arena(db_name, ticks=ticks, learn=False, seed=seed)

    if r_learn and r_ctrl:
        print(f"\n{'='*70}")
        print(f"  COMPARISON ({ticks:,d} ticks)")
        print(f"{'='*70}")
        print(f"  {'':20s} {'LEARN':>12s} {'CONTROL':>12s}")
        print(f"  {'Distance change':20s} {r_learn['d0']-r_learn['d_final']:+12.1f} "
              f"{r_ctrl['d0']-r_ctrl['d_final']:+12.1f}")
        if len(r_learn['distances']) > 10:
            ci_l = (r_learn['distances'][0] - r_learn['distances'][-1]) / max(r_learn['distances'][0], 0.1)
            ci_c = (r_ctrl['distances'][0] - r_ctrl['distances'][-1]) / max(r_ctrl['distances'][0], 0.1)
            print(f"  {'Chemotaxis index':20s} {ci_l:+12.4f} {ci_c:+12.4f}")
        print(f"  {'Motor spikes':20s} {r_learn['motor_spikes']:12,d} {r_ctrl['motor_spikes']:12,d}")
        print(f"  {'Displacement':20s} {r_learn['displacement']:12.2f} {r_ctrl['displacement']:12.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--brain', default='arena_v1_s42.db')
    p.add_argument('--ticks', type=int, default=60000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-learn', action='store_true')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--sensory-gain', type=float, default=5.0)
    p.add_argument('--reward-mag', type=float, default=1.0)
    p.add_argument('--start-distance', type=float, default=20.0)
    p.add_argument('--surprise-reward', action='store_true',
                   help='Use prediction-error surprise gating for reward delivery')
    p.add_argument('--surprise-alpha', type=float, default=0.25,
                   help='EMA alpha for surprise expectation (0.1=slow, 0.5=fast)')
    p.add_argument('--dishbrain', action='store_true',
                   help='DishBrain-style feedback: inject chaos when moving away from food')
    p.add_argument('--save', action='store_true', help='Save brain state after run')
    p.add_argument('--reward-homeostasis', action='store_true',
                   help='Enable during-reward synaptic homeostasis (default: OFF)')
    args = p.parse_args()

    if args.compare:
        compare(args.brain, args.ticks, args.seed)
    else:
        result = run_arena(args.brain, ticks=args.ticks, learn=not args.no_learn,
                  seed=args.seed, sensory_gain=args.sensory_gain,
                  reward_magnitude=args.reward_mag,
                  start_distance=args.start_distance,
                  surprise_reward=args.surprise_reward,
                  surprise_alpha=args.surprise_alpha,
                  dishbrain_feedback=args.dishbrain,
                  reward_homeostasis=args.reward_homeostasis)
        if args.save and result and 'brain' in result:
            result['brain'].save()
            print(f"\n  Brain state saved to {args.brain}")
