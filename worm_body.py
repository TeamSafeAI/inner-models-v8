"""
worm_body.py — 24-segment rigid-chain C. elegans body simulation.

Physics from ElegansBot (Chung et al., eLife 2024): angular spring+damper joints,
anisotropic drag (40:1 perpendicular:parallel on agar), tapered radius profile,
muscle time constant ~100ms for spike-to-force smoothing.

The body moves in 2D (top-down view, worm on agar plate). Each joint has 1 DOF
(bend angle). Dorsal muscle contraction = positive torque, ventral = negative.
Anisotropic drag converts undulation into net forward/backward thrust.

Brain-body coupling:
  - Motor neuron spikes -> muscle activation via body_map table
  - Body bend angles -> proprioceptive feedback current to B-type motor neurons
  - WormSimulation orchestrates the closed loop at 1ms per tick

Usage:
    sim = WormSimulation('brains/elegans_v8_gap_w1.5_s42.db')
    result = sim.run(ticks=10000)  # 10 seconds
    print(result['displacement'])
"""
import os, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ── Physical constants (normalized units) ──
# Lengths in body-lengths (1 BL = 1.0), time in seconds.
# Absolute force values are arbitrary — only RATIOS matter:
#   - Drag ratio 40:1 (perpendicular:parallel) — critical for crawling
#   - Spring restores body to straight in ~200ms
#   - Full muscle activation produces ~20-30 degree bending
# The worm is overdamped (low Reynolds number), so dynamics are first-order:
#   dtheta/dt = (muscle_torque - spring * theta) / rotational_drag

N_SEGMENTS = 24
SEG_LENGTH = 1.0 / N_SEGMENTS            # in body-lengths
N_JOINTS = N_SEGMENTS - 1                # 23 joints

# Drag coefficients per segment (Boyle 2012, ElegansBot-scaled)
# Ratio 40:1 is canonical for agar. Absolute scale from ElegansBot:
# they divide Boyle's raw drag by ~100 to match real trajectories.
# We scale KAPPA by the same factor to keep tau_rot (~0.014s) unchanged.
DRAG_PAR = 150.0                         # parallel (along body axis)
DRAG_PERP = 40.0 * DRAG_PAR              # perpendicular — exact 40:1 ratio

# Joint spring stiffness — scaled with drag to preserve tau_rot
# tau_rot = gamma_rot / kappa ≈ 0.014s (unchanged from original 5.0/0.069)
KAPPA = 750.0                            # = 5.0 * 150 (same scale as drag)

# Rotational drag per joint (resists angular velocity)
GAMMA_ROT = DRAG_PERP * SEG_LENGTH ** 2  # ~10.4

# Muscle activation dynamics
TAU_MUSCLE = 0.100                       # 100ms spike-to-force smoothing

# Maximum bend amplitude at full activation (radians).
# 0.8 rad ≈ 46 degrees — slightly under the pi/3 clamp, feels natural.
MAX_BEND_AMPLITUDE = 0.8

# Activation gain: converts low-pass filtered spike rate to 0-1 range.
# Real brain: ~0.005 spikes/ms per motor neuron → activation ~0.3-0.6
# under command drive. Gain of 12 maps background to ~0.15, command to ~0.5.
ACTIVATION_GAIN = 12.0

# Head-to-tail muscle gain gradient (head muscles ~2.5x stronger)
MUSCLE_GAIN = np.array([
    1.0 + 1.5 * (1.0 - i / (N_SEGMENTS - 1)) for i in range(N_SEGMENTS)
])

# Tapered radius: thickest at ~40% from head, thins at both ends
RADIUS_PROFILE = np.array([
    0.4 + 0.6 * np.exp(-((i / N_SEGMENTS - 0.4) / 0.25)**2)
    for i in range(N_SEGMENTS)
])
RADIUS_PROFILE /= RADIUS_PROFILE.max()

# Proprioceptive feedback gain (uA per radian of bend angle).
# Combined feedback model: 30% local restorative + 70% anterior propagating.
# Local restorative = stretch receptor on opposite side (prevents saturation).
# Anterior propagating = sense bend 1-3 segments ahead (wave propagation).
# gain=25 produces: head oscillation ~1.2 Hz (natural, no external drive),
# traveling wave with ~1.5 wavelengths, speed ~0.2 BL/s.
PROPRIO_GAIN = 25.0

# Fraction of gain for local restorative vs anterior propagating feedback.
# Restorative opposes current bend (drives opposite side muscles).
# Propagating reproduces anterior bend (drives same side muscles).
PROPRIO_LOCAL_FRAC = 0.3   # local restorative (same segment, opposite sign)
PROPRIO_ANT_FRAC = 0.7     # anterior propagating (1-3 seg ahead, same sign)


class WormBody:
    """24-segment rigid-chain worm body with anisotropic drag.

    State:
        theta[23]:       joint angles (radians). 0 = straight. + = dorsal bend.
        dtheta[23]:      joint angular velocities.
        pos[2]:          head position (x, y) in world coordinates.
        heading:         body heading angle (radians).
        dorsal_act[24]:  dorsal muscle activation per segment (0-1).
        ventral_act[24]: ventral muscle activation per segment (0-1).
    """

    def __init__(self):
        self.theta = np.zeros(N_JOINTS)
        self.dtheta = np.zeros(N_JOINTS)
        self.pos = np.array([0.0, 0.0])
        self.heading = 0.0

        # Smooth muscle activations (low-pass filtered from spikes)
        self.dorsal_act = np.zeros(N_SEGMENTS)
        self.ventral_act = np.zeros(N_SEGMENTS)

        # Per-tick spike accumulators (reset each step)
        self._dorsal_input = np.zeros(N_SEGMENTS)
        self._ventral_input = np.zeros(N_SEGMENTS)

    def apply_motor_spike(self, segment, side, effect):
        """Register a motor neuron spike for this tick.

        Args:
            segment: body segment index (0-23)
            side: 'dorsal' or 'ventral'
            effect: 'excitatory' or 'inhibitory'
        """
        if effect == 'excitatory':
            if side == 'dorsal':
                self._dorsal_input[segment] += 1.0
            else:
                self._ventral_input[segment] += 1.0
        else:  # inhibitory (cross-inhibitor)
            if side == 'dorsal':
                self._dorsal_input[segment] -= 0.5
            else:
                self._ventral_input[segment] -= 0.5

    def step(self, dt=0.001):
        """Advance body physics by dt seconds. Call once per brain tick (1ms).

        Returns displacement vector (how much the head moved).
        """
        # 1. Update muscle activation (exponential low-pass filter on spike input)
        # Then apply activation gain to convert sparse spikes into 0-1 range.
        alpha = dt / TAU_MUSCLE
        target_d = np.clip(self._dorsal_input, 0.0, 5.0)
        target_v = np.clip(self._ventral_input, 0.0, 5.0)
        self.dorsal_act += (target_d - self.dorsal_act) * alpha
        self.ventral_act += (target_v - self.ventral_act) * alpha

        # Reset spike accumulators
        self._dorsal_input[:] = 0.0
        self._ventral_input[:] = 0.0

        # 2. Control-angle muscle torque (ElegansBot / Boyle 2012 style)
        # Instead of torque = activation * kappa * scale (where kappa cancels
        # at steady state), we compute a TARGET bend angle from activation and
        # let the spring pull toward it. Steady-state bend = theta_target.
        #   theta_target = (dorsal - ventral) * gain * MAX_BEND_AMPLITUDE
        #   torque = kappa * (theta_target - theta)
        # This gives realistic 30-60° bends at full activation.
        theta_target = np.zeros(N_JOINTS)
        for j in range(N_JOINTS):
            d_act = 0.5 * (self.dorsal_act[j] + self.dorsal_act[j + 1])
            v_act = 0.5 * (self.ventral_act[j] + self.ventral_act[j + 1])
            gain = 0.5 * (MUSCLE_GAIN[j] + MUSCLE_GAIN[j + 1])
            # Scale activations and compute target angle
            d_scaled = min(d_act * ACTIVATION_GAIN, 1.0)
            v_scaled = min(v_act * ACTIVATION_GAIN, 1.0)
            theta_target[j] = (d_scaled - v_scaled) * gain * MAX_BEND_AMPLITUDE

        # 3. Overdamped joint dynamics (first-order, no inertia)
        # Spring pulls toward target angle instead of toward zero.
        # dtheta/dt = kappa * (theta_target - theta) / gamma_rot
        net_torque = KAPPA * (theta_target - self.theta)
        self.dtheta = net_torque / GAMMA_ROT
        self.theta += self.dtheta * dt

        # Clamp to prevent extreme bending
        self.theta = np.clip(self.theta, -np.pi / 3, np.pi / 3)

        # 4. Compute locomotion from body undulation + anisotropic drag
        displacement = self._compute_locomotion(dt)
        self.pos += displacement

        return displacement

    def _compute_locomotion(self, dt):
        """Compute net displacement AND heading change from undulation.

        The drag ratio (40:1 perp:par) is what makes crawling work — lateral
        undulation creates forward thrust because sideways motion is penalized
        much more than forward motion.

        Heading change comes from net torque of drag forces about the head.
        With co-scaled KAPPA and DRAG, the torque calculation is physically
        self-consistent — no arbitrary multiplier needed.
        """
        net_force = np.array([0.0, 0.0])
        net_torque = 0.0
        I_rot = 1e-20  # avoid division by zero

        # Precompute segment positions relative to head (local frame)
        seg_pos = np.zeros((N_SEGMENTS, 2))
        cum_angle = 0.0
        cx, cy = 0.0, 0.0
        for i in range(N_SEGMENTS):
            seg_pos[i] = [cx, cy]
            if i < N_JOINTS:
                cum_angle += self.theta[i]
            cx += SEG_LENGTH * np.cos(cum_angle)
            cy += SEG_LENGTH * np.sin(cum_angle)

        c_h = np.cos(self.heading)
        s_h = np.sin(self.heading)

        for i in range(N_SEGMENTS):
            # Segment orientation in world frame
            angle = self.heading + np.sum(self.theta[:i]) if i > 0 else self.heading
            tangent = np.array([np.cos(angle), np.sin(angle)])
            normal = np.array([-np.sin(angle), np.cos(angle)])

            # Segment velocity from joint angular velocities
            seg_vel = np.array([0.0, 0.0])
            for j in range(min(i, N_JOINTS)):
                lever = (i - j) * SEG_LENGTH
                perp_dir = np.array([
                    -np.sin(self.heading + np.sum(self.theta[:j+1])),
                     np.cos(self.heading + np.sum(self.theta[:j+1]))
                ])
                seg_vel += self.dtheta[j] * lever * perp_dir

            # Decompose velocity into parallel and perpendicular components
            v_par = np.dot(seg_vel, tangent)
            v_perp = np.dot(seg_vel, normal)

            # Anisotropic drag force on this segment
            r = RADIUS_PROFILE[i]
            f_par = -DRAG_PAR * r * v_par * tangent
            f_perp = -DRAG_PERP * r * v_perp * normal
            f_total = f_par + f_perp

            net_force += f_total

            # Torque about head: 2D cross product r × F (world frame)
            rx = c_h * seg_pos[i, 0] - s_h * seg_pos[i, 1]
            ry = s_h * seg_pos[i, 0] + c_h * seg_pos[i, 1]
            net_torque += rx * f_total[1] - ry * f_total[0]

            # Rotational drag about head
            dist2 = seg_pos[i, 0]**2 + seg_pos[i, 1]**2
            I_rot += DRAG_PERP * r * dist2

        # Net displacement via proper anisotropic drag resistance matrix.
        # Each segment contributes a 2x2 drag tensor based on its orientation.
        # This correctly handles perpendicular drag being 40x parallel.
        A = np.zeros((2, 2))
        for i in range(N_SEGMENTS):
            angle = self.heading + np.sum(self.theta[:i]) if i > 0 else self.heading
            c_a, s_a = np.cos(angle), np.sin(angle)
            t = np.array([c_a, s_a])
            n = np.array([-s_a, c_a])
            r = RADIUS_PROFILE[i]
            A += DRAG_PAR * r * np.outer(t, t) + DRAG_PERP * r * np.outer(n, n)
        displacement = np.linalg.solve(A, net_force) * dt

        # Heading change from net torque (rotation)
        self.heading += net_torque / I_rot * dt

        return displacement

    def get_segment_positions_local(self):
        """Get segment center positions relative to head. Shape (N_SEGMENTS, 2)."""
        positions = np.zeros((N_SEGMENTS, 2))
        angle = 0.0
        x, y = 0.0, 0.0
        for i in range(N_SEGMENTS):
            positions[i] = [x, y]
            if i < N_JOINTS:
                angle += self.theta[i]
            x += SEG_LENGTH * np.cos(angle)
            y += SEG_LENGTH * np.sin(angle)
        return positions

    def get_segment_positions_world(self):
        """Get segment center positions in world coordinates. Shape (N_SEGMENTS, 2)."""
        local = self.get_segment_positions_local()
        c, s = np.cos(self.heading), np.sin(self.heading)
        rot = np.array([[c, -s], [s, c]])
        return (rot @ local.T).T + self.pos

    def get_bend_angles(self):
        """Get current joint angles (radians). Shape (N_JOINTS,)."""
        return self.theta.copy()

    def get_curvature(self):
        """Get body curvature (theta/seg_length). + = dorsal, - = ventral."""
        return self.theta / SEG_LENGTH


class WormSimulation:
    """Coupled brain + body simulation for C. elegans.

    Runs BrainState and WormBody in lockstep at 1ms ticks.
    Motor neuron spikes drive muscles via body_map.
    Body bend angles feed back as proprioceptive current.
    """

    def __init__(self, brain_path, learn=False, seed=42):
        from simulate import load_brain, BrainState

        brain = load_brain(brain_path)
        self.brain = BrainState(brain, learn=learn, seed=seed)
        self.body = WormBody()
        self.body_map = self.brain.body_map  # {neuron_idx: {segment, side, effect}}

        # Build proprio targets: body_map entries that are excitatory
        # B-type motor neurons (forward drive) get proprioceptive stretch feedback
        # This propagates the undulation wave along the body
        self.proprio_targets = {}  # {neuron_idx: segment}
        for idx, entry in self.body_map.items():
            if entry['effect'] == 'excitatory':
                self.proprio_targets[idx] = entry['segment']

        # Recording
        self.tick = 0
        self.position_history = []
        self.shape_history = []

    def step(self, external_I=None):
        """Advance one tick (1ms). Returns fired neuron indices."""
        n = self.brain.n

        # 1. Proprioceptive feedback: body bend angles -> motor neuron current
        proprio_I = np.zeros(n)
        angles = self.body.get_bend_angles()

        for idx, seg in self.proprio_targets.items():
            entry = self.body_map[idx]

            # Component 1: LOCAL restorative feedback (stretch receptor model).
            # Bend stretches the OPPOSITE side -> drives opposite muscles.
            # Prevents saturation, creates head oscillation at seg 0.
            j_local = min(seg, N_JOINTS - 1)
            local_gain = PROPRIO_GAIN * PROPRIO_LOCAL_FRAC
            if entry['side'] == 'dorsal':
                proprio_I[idx] += -angles[j_local] * local_gain  # dorsal bend -> inhibit dorsal
            else:
                proprio_I[idx] += angles[j_local] * local_gain   # dorsal bend -> excite ventral

            # Component 2: ANTERIOR wave-propagating feedback.
            # Sense bends 1-3 segments ahead, weighted by distance.
            # Models B-type dendrites extending anteriorly (Wen et al. 2012).
            # Bridges motor neuron gaps at segments 8, 11, 13.
            weighted_angle = 0.0
            total_weight = 0.0
            for offset in range(1, 4):  # 1, 2, 3 segments anterior
                j = seg - offset
                if j < 0 or j >= N_JOINTS:
                    break
                w = 1.0 / offset  # 1.0, 0.5, 0.33
                weighted_angle += angles[j] * w
                total_weight += w
            if total_weight > 0:
                weighted_angle /= total_weight
                ant_gain = PROPRIO_GAIN * PROPRIO_ANT_FRAC
                if entry['side'] == 'dorsal':
                    proprio_I[idx] += weighted_angle * ant_gain   # same bend propagates
                else:
                    proprio_I[idx] += -weighted_angle * ant_gain  # same bend propagates

        # 2. Combine with external stimulus
        total_I = proprio_I
        if external_I is not None:
            total_I = total_I + external_I

        # 3. Brain step
        fired_idx = self.brain.step(total_I)

        # 4. Motor neuron spikes -> body
        for fi in fired_idx:
            if fi in self.body_map:
                entry = self.body_map[fi]
                self.body.apply_motor_spike(entry['segment'], entry['side'], entry['effect'])

        # 5. Body step
        self.body.step(dt=0.001)

        # 6. Record periodically
        self.tick += 1
        if self.tick % 100 == 0:
            self.position_history.append(self.body.pos.copy())
            self.shape_history.append(self.body.get_segment_positions_world())

        return fired_idx

    def run(self, ticks, external_I=None, quiet=False):
        """Run for N ticks with optional constant external stimulus.

        Returns summary dict with displacement, positions, motor spike counts.
        """
        total_spikes = 0
        motor_spikes = 0

        for t in range(ticks):
            fired = self.step(external_I=external_I)
            total_spikes += len(fired)
            for fi in fired:
                if fi in self.body_map:
                    motor_spikes += 1

            if not quiet and (t + 1) % 2000 == 0:
                pos = self.body.pos
                print('  tick %7d | spikes %6d | motor %5d | pos (%.6f, %.6f)' % (
                    t + 1, total_spikes, motor_spikes, pos[0], pos[1]))

        return {
            'ticks': ticks,
            'total_spikes': total_spikes,
            'motor_spikes': motor_spikes,
            'final_pos': self.body.pos.copy(),
            'displacement': float(np.linalg.norm(self.body.pos)),
            'heading': float(self.body.heading),
            'positions': self.position_history,
            'shapes': self.shape_history,
            'max_curvature': float(np.max(np.abs(self.body.get_curvature()))),
        }

    def make_stimulus(self, neuron_indices, current=15.0):
        """Create external_I array for specific neurons.

        Args:
            neuron_indices: list of neuron array indices to stimulate
            current: uA to inject
        Returns:
            numpy array (shape n) for use with step() or run()
        """
        I = np.zeros(self.brain.n)
        for idx in neuron_indices:
            I[idx] = current
        return I

    def find_neurons_by_type_and_position(self, neuron_type, x_min=None, x_max=None):
        """Find neuron indices by Izhikevich type and X position range.

        Uses the brain's stored neuron types and positions — no names needed.
        Useful for finding command interneurons (IB type, head position)
        or sensory neurons (RS type, specific position).
        """
        # Need access to raw neuron data. Load from brain dict.
        from simulate import load_brain
        brain_data = load_brain(self.brain.body_map and
                                 os.path.join(BASE, 'brains', 'elegans_v8_gap_w1.5_s42.db') or '')
        # Actually, we have neuron_types in brain already
        # But positions aren't in BrainState. Let's load from DB directly.
        import sqlite3
        conn = sqlite3.connect(os.path.join(BASE, 'brains', 'elegans_v8_gap_w1.5_s42.db'))
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT id, neuron_type, pos_x FROM neurons ORDER BY id').fetchall()
        conn.close()

        # Build id->idx map
        id_to_idx = {rows[i]['id']: i for i in range(len(rows))}

        results = []
        for row in rows:
            if row['neuron_type'] != neuron_type:
                continue
            x = row['pos_x']
            if x_min is not None and x < x_min:
                continue
            if x_max is not None and x > x_max:
                continue
            results.append(id_to_idx[row['id']])
        return results


if __name__ == '__main__':
    # Quick sanity check
    print('Creating WormSimulation...')
    sim = WormSimulation(os.path.join(BASE, 'brains', 'elegans_v8_gap_w1.5_s42.db'))
    print('  Body map: %d motor neurons' % len(sim.body_map))
    print('  Proprio targets: %d' % len(sim.proprio_targets))
    print()
    print('Running 1000 ticks (1 second)...')
    result = sim.run(1000, quiet=True)
    print('  Total spikes: %d' % result['total_spikes'])
    print('  Motor spikes: %d' % result['motor_spikes'])
    print('  Position: (%.8f, %.8f)' % (result['final_pos'][0], result['final_pos'][1]))
    print('  Displacement: %.8f m' % result['displacement'])
    print('  Max curvature: %.4f' % result['max_curvature'])
    print()
    print('Done.')
