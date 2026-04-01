"""
visualize_arena.py — Generate trajectory plots for arena experiments.

Saves PNG files showing worm path overlaid on concentration gradient heatmap.
No interactive display needed — just saves images for review.

Usage:
    py visualize_arena.py                  # default experiment
    py visualize_arena.py --gain 10        # custom sensory gain
    py visualize_arena.py --learn          # with STDP
    py visualize_arena.py --ticks 60000    # 60 second run
"""
import os, sys, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from arena import Arena, ArenaSimulation

BRAIN_DB = os.path.join(BASE, 'brains', 'elegans_v8_gap_w1.5_s42.db')
OUTPUT_DIR = os.path.join(BASE, 'arena_results')


def plot_trajectory(arena, trajectory, title, filename,
                    food_pos=None, start_pos=None):
    """Save trajectory plot as PNG using matplotlib.

    Shows:
    - Background: concentration heatmap
    - Line: worm trajectory (colored by time: blue=start, red=end)
    - Markers: start (green), end (red), food (yellow star)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # no display needed
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print('  matplotlib not available — skipping plot')
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Concentration heatmap
    r = arena.radius
    extent = [-r * 1.1, r * 1.1, -r * 1.1, r * 1.1]
    grid_n = 200
    xs = np.linspace(extent[0], extent[1], grid_n)
    ys = np.linspace(extent[2], extent[3], grid_n)
    XX, YY = np.meshgrid(xs, ys)
    CC = np.zeros_like(XX)
    for i in range(grid_n):
        for j in range(grid_n):
            CC[i, j] = arena.concentration_at(XX[i, j], YY[i, j])

    # Mask outside arena
    mask = XX ** 2 + YY ** 2 > r ** 2
    CC[mask] = np.nan

    ax.imshow(CC, extent=extent, origin='lower', cmap='YlOrRd',
              alpha=0.5, vmin=0, vmax=CC[~mask].max() if not np.all(mask) else 1.0)

    # Arena boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=1.5, alpha=0.5)

    # Trajectory colored by time
    if len(trajectory) > 1:
        traj = np.array(trajectory)
        n_pts = len(traj)
        for i in range(n_pts - 1):
            t_frac = i / max(n_pts - 1, 1)
            color = (t_frac, 0.2, 1.0 - t_frac)  # blue -> red
            ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-',
                    color=color, linewidth=1.5, alpha=0.8)

    # Markers
    if start_pos is not None:
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=12,
                label='Start', zorder=5)
    if len(trajectory) > 0:
        end = trajectory[-1]
        ax.plot(end[0], end[1], 'rs', markersize=10,
                label='End', zorder=5)
    if food_pos is not None:
        ax.plot(food_pos[0], food_pos[1], 'y*', markersize=20,
                markeredgecolor='orange', label='Food', zorder=5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlabel('X (body-lengths)')
    ax.set_ylabel('Y (body-lengths)')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: %s' % filepath)


def run_and_visualize(sensory_gain=5.0, learn=False, ticks=30000,
                       seed=42, start_angle_deg=90):
    """Run one experiment and generate trajectory plot."""
    arena = Arena(radius=30.0)
    arena.add_food(0.0, 0.0, peak=1.0, sigma=12.0)

    start_distance = 20.0
    angle = np.radians(start_angle_deg)
    sx = start_distance * np.cos(angle)
    sy = start_distance * np.sin(angle)
    heading = angle + np.pi / 2  # perpendicular to food direction

    label = 'gain%.0f_%s_%ds_angle%d' % (
        sensory_gain, 'learn' if learn else 'frozen', ticks // 1000, start_angle_deg)

    print('\nRunning: %s' % label)
    print('  Start: (%.1f, %.1f), heading: %.0f deg, gain: %.1f' % (
        sx, sy, np.degrees(heading), sensory_gain))

    # Sensory run
    sim_sense = ArenaSimulation(
        BRAIN_DB, arena, sensory_gain=sensory_gain, learn=learn,
        seed=seed, start_pos=(sx, sy), start_heading=heading)
    r_sense = sim_sense.run(ticks, quiet=True)

    # Control run
    sim_ctrl = ArenaSimulation(
        BRAIN_DB, arena, sensory_gain=0.0, learn=learn,
        seed=seed, start_pos=(sx, sy), start_heading=heading)
    r_ctrl = sim_ctrl.run(ticks, quiet=True)

    print('  Sensory: d_food %.1f -> %.1f (CI=%.4f)' % (
        r_sense['initial_distance'], r_sense['final_distance'],
        r_sense['chemotaxis_index']))
    print('  Control: d_food %.1f -> %.1f (CI=%.4f)' % (
        r_ctrl['initial_distance'], r_ctrl['final_distance'],
        r_ctrl['chemotaxis_index']))

    # Generate plots
    plot_trajectory(
        arena, r_sense['trajectory'],
        'Sensory (gain=%.0f) — CI=%.4f' % (sensory_gain, r_sense['chemotaxis_index']),
        '%s_sensory.png' % label,
        food_pos=(0.0, 0.0), start_pos=(sx, sy))

    plot_trajectory(
        arena, r_ctrl['trajectory'],
        'Control (no input) — CI=%.4f' % r_ctrl['chemotaxis_index'],
        '%s_control.png' % label,
        food_pos=(0.0, 0.0), start_pos=(sx, sy))

    return r_sense, r_ctrl


def main():
    parser = argparse.ArgumentParser(description='Visualize arena chemotaxis')
    parser.add_argument('--gain', type=float, default=5.0)
    parser.add_argument('--learn', action='store_true')
    parser.add_argument('--ticks', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--angle', type=int, default=90)
    parser.add_argument('--sweep', action='store_true',
                        help='Run multiple angles and gains')
    args = parser.parse_args()

    if not os.path.exists(BRAIN_DB):
        print('Brain not found: %s' % BRAIN_DB)
        return

    if args.sweep:
        # Run a grid of experiments
        for gain in [0.0, 5.0, 10.0, 20.0]:
            for angle in [0, 90, 180, 270]:
                run_and_visualize(
                    sensory_gain=gain, learn=args.learn,
                    ticks=args.ticks, seed=args.seed,
                    start_angle_deg=angle)
    else:
        run_and_visualize(
            sensory_gain=args.gain, learn=args.learn,
            ticks=args.ticks, seed=args.seed,
            start_angle_deg=args.angle)


if __name__ == '__main__':
    main()
