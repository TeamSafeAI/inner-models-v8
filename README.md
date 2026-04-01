# V8 Inner Models Engine

Modular spiking neural network engine. Izhikevich neurons, 8 synapse types, composable building blocks, JSON brain recipes. NumPy-optimized.

The idea: neurons and synapses are self-contained **parts** with intrinsic behavior. The engine just ticks. Topology creates function.

## What's here

```
engine/
  neurons/     5 Izhikevich types (RS, FS, IB, CH, LTS)
  paths/       8 synapse types (fixed, plastic, facilitating, depressing,
               gated, gap_junction, reward_plastic, developmental)
  runner.py    Brain class - tick(), deliver_reward()
  loader.py    Load brain DB
  recorder.py  Spike/weight recording
  encoder.py   Signal encoder (audio/tone -> neuron currents)

blocks/
  components/       11 circuit components (JSON)
  sub_components/   21 sub-circuit components (JSON)

recipes/           JSON brain recipes
schema.py          SQLite brain DB schema
brain_generator.py Recipe -> DB compiler
build_arena_brain.py  Arena brain + body/sensor map population
run_arena.py       Full brain-body-environment loop
arena.py           2D arena (Gaussian food gradients, circular boundary)
worm_body.py       24-segment rigid chain body (anisotropic drag, proprioception)
```

## Quick start

Requires: Python 3.10+, NumPy

```bash
# Generate a brain from recipe
py brain_generator.py --recipe recipes/arena_v1.json

# Populate body/sensor maps for arena
py build_arena_brain.py --brain brains/zoo/arena_v1_s42.db

# Run chemotaxis (brain + body + environment)
py run_arena.py --brain arena_v1_s42.db --ticks 30000

# Run developmental (blank slate) brain
py brain_generator.py --recipe recipes/arena_dev_v1.json
py build_arena_brain.py --brain brains/zoo/arena_dev_v1_s42.db
py run_arena.py --brain arena_dev_v1_s42.db --ticks 30000
```

## Building blocks

Each component is a JSON file defining neurons, internal synapses, and terminals (named I/O ports). Meaning comes from **wiring**, not labels.

Example: `binary_decision` is a generic 2-option circuit. Two bistable CH rings, cross-kill FS neurons, RS integrators with STDP. Wire mechanosensory to sensory_a_in/sensory_b_in and it decides forward/backward. Wire emotional_state to modulator_a_in/modulator_b_in and emotion biases the choice.

Example: `sensory_bank` is 3 neurons (2 RS + gap junction for noise reduction, 1 RS output with depressing adaptation). Modality-agnostic. What it senses depends on what you wire to channel_in.

Example: `working_memory_cell` is a bistable CH ring + RS trigger + FS kill + 2 RS association neurons with fast STDP (lr=0.5). Sets a 1-bit memory. Whatever was active when the memory set can later recall it.

## Synapse types

| Type | Behavior |
|------|----------|
| fixed | Static weight. Reflexes, structural connections. |
| plastic | STDP. Weight changes based on pre/post timing. |
| facilitating | Gets stronger with repeated use. Short-term potentiation. |
| depressing | Gets weaker with repeated use. Adaptation/habituation. |
| gated | Only transmits when modulator neuron is active. Attention. |
| gap_junction | Electrical coupling. Continuous, bidirectional. |
| reward_plastic | 3-factor: pre x post x reward. Reinforcement learning. |
| developmental | STDP + Fisher Information pruning. Blank slate. |

## Recipes

JSON files that declare components, counts, layer positions, and inter-layer projections with synapse types. The brain_generator places components in 3D space and wires them by proximity.

**arena_v1.json**: Hardwired animal brain. ~8K neurons, 21 component types, 5 layers. Fixed sensory-decision-motor spine. Reward_plastic on emotion->decision. Works immediately (CI=+0.43 in 30K ticks).

**arena_dev_v1.json**: Blank slate brain. Same components but inter-layer connections are developmental synapses. Only the reflex spine (sensory->decision->motor) is fixed. Everything else starts dense and gets pruned by Fisher Information during a 10K-tick critical period. Needs extended experience to develop.

## Neurons

All 5 types use the same Izhikevich model with different parameters:

| Type | Role | Key property |
|------|------|-------------|
| RS | Standard excitatory | Steady spiking |
| FS | Inhibitory interneuron | Fast, precise |
| IB | Motor, sustainers | Initial burst then steady |
| CH | Oscillators, bistable rings | Rhythmic bursting |
| LTS | Suppressor inhibitory | Low threshold |

## Performance

8070N brain @ 82 t/s with full body physics (24-segment worm, proprioception, arena).
50K brain @ 49-209 t/s (pure brain, no body).

## The developmental approach

Human brains overproduce synapses ~2x (peaking around age 2), then prune through adolescence. The topology that survives is shaped by experience.

The `developmental` synapse type implements this:
1. Start with dense, random connections (overproduction)
2. Track Fisher Information: coincidence_rate = (pre-post coincidences) / (source fires)
3. Every eval_interval ticks during critical period: prune synapses where coincidence rate < threshold
4. Surviving synapses continue as regular STDP plastic
5. After critical period: topology is set, no more pruning

This is the blank slate. The animal brain (arena_v1) works from birth because pathways are pre-wired. The developmental brain (arena_dev_v1) must discover its own wiring through experience.

## Design principles

1. **Architecture > quantity.** Topology matters more than neuron count.
2. **Memory density predicts learning.** Dense WMC cross-wiring = more STDP surface.
3. **Each part owns its own rules.** The engine doesn't know what "emotional" means.
4. **Meaning from wiring, not labels.** binary_decision doesn't know it's deciding forward/backward.
5. **Projection:crosswire ratio > 1.5** needed for learning at scale.
