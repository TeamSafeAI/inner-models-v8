"""
arena.py — 2D environment for C. elegans chemotaxis simulation.

The arena is a petri dish (circular boundary) with one or more food sources
that emit chemical gradients. Amphid chemosensory neurons receive current
proportional to the concentration at the worm's head position.

NO behavior is hardcoded here. The arena provides physics (concentration field,
boundary constraints). The brain decides what to do with the sensory input.

Brain-body-environment loop:
  1. Arena: concentration at worm's head position
  2. Sensors: current = gain * concentration → injected into amphid neurons
  3. Brain: existing connectome processes sensory input (AWC→AIY→RIA→AVB etc.)
  4. Body: motor neuron spikes → muscle activation → movement
  5. Arena: worm's new position → go to step 1

Usage:
    arena = Arena(radius=30.0)
    arena.add_food(15.0, 0.0, peak=1.0, sigma=12.0)
    sim = ArenaSimulation('brains/elegans_v8_gap_w1.5_s42.db', arena)
    result = sim.run(ticks=60000)  # 60 seconds
"""
import os, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from worm_body import WormSimulation


class Arena:
    """2D circular arena (petri dish) with chemical concentration field.

    Food sources emit Gaussian concentration gradients. Multiple sources
    add linearly (superposition). The arena has a hard circular boundary.
    """

    def __init__(self, radius=30.0):
        """Create arena.

        Args:
            radius: arena radius in body-lengths (default 30 BL = ~30mm,
                    typical 60mm petri dish is ~60 BL diameter)
        """
        self.radius = radius
        self.food_sources = []  # list of (x, y, peak, sigma)

    def add_food(self, x, y, peak=1.0, sigma=12.0):
        """Place a food source at (x, y).

        Args:
            x, y: position in body-lengths
            peak: maximum concentration at source center (arbitrary units)
            sigma: spread of gradient in body-lengths (controls steepness)
        """
        self.food_sources.append((x, y, peak, sigma))

    def concentration_at(self, x, y):
        """Get chemical concentration at position (x, y).

        Returns scalar concentration (sum of all food source Gaussians).
        """
        c = 0.0
        for fx, fy, peak, sigma in self.food_sources:
            d2 = (x - fx) ** 2 + (y - fy) ** 2
            c += peak * np.exp(-d2 / (2.0 * sigma ** 2))
        return c

    def is_inside(self, x, y):
        """Check if position is inside the arena boundary."""
        return x * x + y * y <= self.radius ** 2

    def clamp_to_boundary(self, pos):
        """Project position back inside the arena if outside.

        Returns clamped position (numpy array). This is a physical constraint
        (wall), not a behavioral rule.
        """
        d = np.linalg.norm(pos)
        if d > self.radius:
            return pos * (self.radius / d)
        return pos


class ArenaSimulation:
    """Worm navigating a 2D arena with sensory feedback.

    Wraps WormSimulation and adds:
    - Sensory input: concentration at head → current to amphid neurons
    - Arena boundary: physical wall constraint
    - Trajectory tracking for analysis
    """

    def __init__(self, brain_path, arena, sensory_gain=5.0,
                 learn=False, seed=42, start_pos=None, start_heading=None,
                 derivative_sensing=False, tau_adapt=500.0,
                 spatial_sensing=False, head_half_width=0.04):
        """Create arena simulation.

        Args:
            brain_path: path to brain .db file
            arena: Arena instance with food sources
            sensory_gain: uA of current per unit concentration (tunable)
            learn: enable STDP learning
            seed: random seed
            start_pos: initial worm position (x, y) or None for origin
            start_heading: initial heading in radians or None for 0
            derivative_sensing: if True, sensors respond to concentration
                CHANGES (dC/dt) rather than absolute levels. This is a
                sensor biophysics property — real amphid neurons adapt via
                receptor desensitization and second-messenger cascades.
            tau_adapt: adaptation time constant in ms (ticks). Controls how
                fast the baseline adapts. ~500ms matches biological sensory
                adaptation rates.
            spatial_sensing: if True, left/right amphid neurons sample
                concentration at different positions offset perpendicular to
                heading. This is physically accurate — amphid sensilla are
                on opposite sides of the head (~80um apart).
            head_half_width: half-width of head in body-lengths (default 0.04
                = 40um, real worm head is ~80um diameter).
        """
        self.sim = WormSimulation(brain_path, learn=learn, seed=seed)
        self.arena = arena
        self.sensory_gain = sensory_gain
        self.derivative_sensing = derivative_sensing
        self.tau_adapt = tau_adapt
        self.spatial_sensing = spatial_sensing
        self.head_half_width = head_half_width

        # Sensory adaptation state (for dC/dt encoding)
        self.adapted_conc = 0.0       # center adaptation
        self.adapted_conc_left = 0.0  # left-side adaptation
        self.adapted_conc_right = 0.0 # right-side adaptation

        # Set initial position/heading
        if start_pos is not None:
            self.sim.body.pos = np.array(start_pos, dtype=float)
        if start_heading is not None:
            self.sim.body.heading = float(start_heading)

        # Build lists of chemosensory neuron indices by type, side
        self.chemical_sensors = []
        self.on_sensors = []
        self.off_sensors = []
        self.tonic_sensors = []
        self.left_sensors = []   # left amphid neurons
        self.right_sensors = []  # right amphid neurons
        sensor_map = self.sim.brain.sensor_map
        for idx, entry in sensor_map.items():
            if entry['modality'] == 'chemical':
                self.chemical_sensors.append(idx)
                rtype = entry.get('response_type', 'tonic')
                if rtype == 'ON':
                    self.on_sensors.append(idx)
                elif rtype == 'OFF':
                    self.off_sensors.append(idx)
                else:
                    self.tonic_sensors.append(idx)
                side = entry.get('side', 'bilateral')
                if side == 'left':
                    self.left_sensors.append(idx)
                elif side == 'right':
                    self.right_sensors.append(idx)

        # Initialize adapted_conc to starting concentration (avoid initial transient)
        if start_pos is not None and self.derivative_sensing:
            c0 = self.arena.concentration_at(start_pos[0], start_pos[1])
            self.adapted_conc = c0
            self.adapted_conc_left = c0
            self.adapted_conc_right = c0

        # Tracking
        self.tick = 0
        self.trajectory = []        # (x, y) every N ticks
        self.concentration_log = [] # concentration at head over time
        self.distance_to_food = []  # distance to nearest food over time

    def step(self):
        """Advance one tick (1ms).

        Returns fired neuron indices.
        """
        # 1. Get concentration at worm's head position
        head = self.sim.body.pos
        heading = self.sim.body.heading
        conc = self.arena.concentration_at(head[0], head[1])

        # 2. Build sensory current
        sensory_I = np.zeros(self.sim.brain.n)

        if self.spatial_sensing and self.derivative_sensing:
            # SPATIAL + dC/dt: Left and right amphid neurons sample different
            # positions AND respond to changes. This combines two real biophysical
            # properties:
            # 1. Amphid sensilla are on opposite sides of the head (~80um apart)
            # 2. Sensory neurons adapt to sustained stimuli (dC/dt encoding)
            #
            # During head oscillation, the left/right offset rotates with the
            # head bend, creating differential dC/dt between left and right.
            # Left amphid neurons (AWCL, AWAL, etc.) project preferentially to
            # left interneurons (AIYL, AIZL), and right to right (AIYR, AIZR).
            # This creates the asymmetric motor drive needed for steering.
            normal_x = -np.sin(heading)  # perpendicular to heading
            normal_y = np.cos(heading)
            offset = self.head_half_width

            # Left/right head positions
            lx = head[0] + normal_x * offset
            ly = head[1] + normal_y * offset
            rx = head[0] - normal_x * offset
            ry = head[1] - normal_y * offset

            conc_left = self.arena.concentration_at(lx, ly)
            conc_right = self.arena.concentration_at(rx, ry)

            alpha = 1.0 / self.tau_adapt
            delta_left = conc_left - self.adapted_conc_left
            delta_right = conc_right - self.adapted_conc_right
            self.adapted_conc_left += alpha * (conc_left - self.adapted_conc_left)
            self.adapted_conc_right += alpha * (conc_right - self.adapted_conc_right)

            # Left amphid neurons get left-side dC/dt (rectified ON)
            left_signal = max(0.0, delta_left) * self.sensory_gain
            for idx in self.left_sensors:
                sensory_I[idx] = left_signal

            # Right amphid neurons get right-side dC/dt (rectified ON)
            right_signal = max(0.0, delta_right) * self.sensory_gain
            for idx in self.right_sensors:
                sensory_I[idx] = right_signal

        elif self.spatial_sensing:
            # SPATIAL only: left/right neurons get different concentrations
            normal_x = -np.sin(heading)
            normal_y = np.cos(heading)
            offset = self.head_half_width

            conc_left = self.arena.concentration_at(
                head[0] + normal_x * offset, head[1] + normal_y * offset)
            conc_right = self.arena.concentration_at(
                head[0] - normal_x * offset, head[1] - normal_y * offset)

            for idx in self.left_sensors:
                sensory_I[idx] = conc_left * self.sensory_gain
            for idx in self.right_sensors:
                sensory_I[idx] = conc_right * self.sensory_gain

        elif self.derivative_sensing:
            # dC/dt only (all sensors same position)
            alpha = 1.0 / self.tau_adapt
            delta = conc - self.adapted_conc
            self.adapted_conc += alpha * (conc - self.adapted_conc)

            signal = max(0.0, delta) * self.sensory_gain
            for idx in self.chemical_sensors:
                sensory_I[idx] = signal
        else:
            # Absolute concentration mode (original behavior)
            for idx in self.chemical_sensors:
                sensory_I[idx] = conc * self.sensory_gain

        # 3. Step worm simulation (brain + body with proprioception)
        fired = self.sim.step(external_I=sensory_I)

        # 4. Arena boundary constraint (physical wall)
        self.sim.body.pos = self.arena.clamp_to_boundary(self.sim.body.pos)

        # 5. Record trajectory
        self.tick += 1
        if self.tick % 100 == 0:  # every 100ms
            self.trajectory.append(self.sim.body.pos.copy())
            self.concentration_log.append(conc)
            # Distance to nearest food
            if self.arena.food_sources:
                min_d = min(
                    np.sqrt((head[0] - fx) ** 2 + (head[1] - fy) ** 2)
                    for fx, fy, _, _ in self.arena.food_sources
                )
                self.distance_to_food.append(min_d)

        return fired

    def run(self, ticks, quiet=False):
        """Run for N ticks.

        Returns summary dict with trajectory, chemotaxis metrics, etc.
        """
        total_spikes = 0
        motor_spikes = 0
        sensory_spikes = 0

        for t in range(ticks):
            fired = self.step()
            total_spikes += len(fired)
            for fi in fired:
                if fi in self.sim.body_map:
                    motor_spikes += 1
                if fi in self.chemical_sensors:
                    sensory_spikes += 1

            if not quiet and (t + 1) % 5000 == 0:
                pos = self.sim.body.pos
                conc = self.arena.concentration_at(pos[0], pos[1])
                d_food = 0.0
                if self.arena.food_sources:
                    fx, fy = self.arena.food_sources[0][:2]
                    d_food = np.sqrt((pos[0] - fx) ** 2 + (pos[1] - fy) ** 2)
                print('  tick %7d | pos (%.2f, %.2f) | conc %.4f | d_food %.2f | '
                      'spikes %d' % (t + 1, pos[0], pos[1], conc, d_food,
                                     total_spikes))

        # Compute metrics
        trajectory = np.array(self.trajectory) if self.trajectory else np.zeros((0, 2))
        distances = np.array(self.distance_to_food) if self.distance_to_food else np.array([])

        result = {
            'ticks': ticks,
            'total_spikes': total_spikes,
            'motor_spikes': motor_spikes,
            'sensory_spikes': sensory_spikes,
            'final_pos': self.sim.body.pos.copy(),
            'final_heading': float(self.sim.body.heading),
            'trajectory': trajectory,
            'concentration_log': np.array(self.concentration_log),
            'distances': distances,
            'n_chemical_sensors': len(self.chemical_sensors),
        }

        # Chemotaxis metrics (if food sources exist and we have distance data)
        if len(distances) > 0:
            result['initial_distance'] = float(distances[0]) if len(distances) > 0 else 0.0
            result['final_distance'] = float(distances[-1]) if len(distances) > 0 else 0.0
            result['mean_distance'] = float(np.mean(distances))
            result['min_distance'] = float(np.min(distances))

            # Chemotaxis index: positive = approaching food, negative = moving away
            # CI = (d_initial - d_final) / d_initial
            d0 = distances[0] if distances[0] > 0 else 1.0
            result['chemotaxis_index'] = float((distances[0] - distances[-1]) / d0)

            # Time in zones (fraction of time spent at various distances)
            if self.arena.food_sources:
                sigma = self.arena.food_sources[0][3]
                result['time_near_food'] = float(np.mean(distances < sigma))
                result['time_far_from_food'] = float(np.mean(distances > 2.0 * sigma))

        return result


def run_chemotaxis_experiment(brain_path, arena, ticks=60000,
                               sensory_gain=5.0, learn=False, seed=42,
                               start_distance=20.0, quiet=False):
    """Run a standard chemotaxis experiment.

    Places worm at start_distance from food, runs for ticks,
    returns result dict with trajectory and metrics.

    Also runs a control (no sensory input) for comparison.
    """
    if not arena.food_sources:
        raise ValueError("Arena must have at least one food source")

    fx, fy = arena.food_sources[0][:2]

    # Start worm at start_distance from food, heading random-ish
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, 2 * np.pi)
    start_x = fx + start_distance * np.cos(angle)
    start_y = fy + start_distance * np.sin(angle)
    start_heading = rng.uniform(0, 2 * np.pi)

    if not quiet:
        print('=== Chemotaxis Experiment ===')
        print('  Food at (%.1f, %.1f), worm starts at (%.1f, %.1f)' % (
            fx, fy, start_x, start_y))
        print('  Start distance: %.1f BL, heading: %.1f deg' % (
            start_distance, np.degrees(start_heading)))
        print('  Sensory gain: %.1f, learning: %s' % (sensory_gain, learn))
        print('  Chemical sensors: loading...')

    # Run with sensory input
    if not quiet:
        print('\n--- With sensory input ---')
    sim_sensory = ArenaSimulation(
        brain_path, arena, sensory_gain=sensory_gain,
        learn=learn, seed=seed,
        start_pos=(start_x, start_y), start_heading=start_heading)
    if not quiet:
        print('  Chemical sensors: %d neurons' % sim_sensory.n_chemical_sensors
              if hasattr(sim_sensory, 'n_chemical_sensors')
              else '  Chemical sensors: %d neurons' % len(sim_sensory.chemical_sensors))
    result_sensory = sim_sensory.run(ticks, quiet=quiet)

    # Run control (no sensory input = sensory_gain=0)
    if not quiet:
        print('\n--- Control (no sensory input) ---')
    sim_control = ArenaSimulation(
        brain_path, arena, sensory_gain=0.0,
        learn=learn, seed=seed,
        start_pos=(start_x, start_y), start_heading=start_heading)
    result_control = sim_control.run(ticks, quiet=quiet)

    return {
        'sensory': result_sensory,
        'control': result_control,
    }


if __name__ == '__main__':
    brain_path = os.path.join(BASE, 'brains', 'elegans_v8_gap_w1.5_s42.db')
    if not os.path.exists(brain_path):
        print('Brain not found: %s' % brain_path)
        print('Run: py elegans_import.py')
        sys.exit(1)

    # Set up arena: petri dish with food at center-right
    arena = Arena(radius=30.0)
    arena.add_food(15.0, 0.0, peak=1.0, sigma=12.0)

    results = run_chemotaxis_experiment(
        brain_path, arena, ticks=30000, sensory_gain=5.0,
        learn=False, seed=42, start_distance=20.0)

    print('\n=== Results ===')
    for label, r in results.items():
        print('\n%s:' % label)
        print('  Final pos: (%.2f, %.2f)' % (r['final_pos'][0], r['final_pos'][1]))
        if 'initial_distance' in r:
            print('  Distance: %.2f -> %.2f (change: %.2f)' % (
                r['initial_distance'], r['final_distance'],
                r['initial_distance'] - r['final_distance']))
            print('  Chemotaxis index: %.4f' % r['chemotaxis_index'])
            print('  Mean distance: %.2f, Min distance: %.2f' % (
                r['mean_distance'], r['min_distance']))
        print('  Spikes: %d total, %d motor, %d sensory' % (
            r['total_spikes'], r['motor_spikes'], r['sensory_spikes']))
