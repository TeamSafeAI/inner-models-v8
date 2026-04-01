"""
Brain Generator — compose building blocks into brain DBs at any scale.

Instantiates N copies of components, places them in 3D space, and auto-wires
between terminals based on proximity and compatibility.

Usage:
  py brain_generator.py --list
  py brain_generator.py --components novelty_detector:5,emotional_state:5 --output brains/gen.db
  py brain_generator.py --recipe recipes/basic.json
  py brain_generator.py --components emotional_state:3,command_decision_v2:2 --seed 99 --max-dist 25

Wiring rules:
  - Output terminals connect to input terminals of DIFFERENT instances
  - kill_in, calm_in are excluded (need specific control sources)
  - Connection probability scales with proximity (closer = more likely)
  - One random output neuron -> one random input neuron per connection (sparse)
  - Weight based on input terminal type (trigger=10, sensory=8, modulator=3, etc.)
"""
import os, sys, json, argparse, math, random

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from schema import create_brain_db, add_neuron, add_synapse

BLOCKS_DIR = os.path.join(BASE, 'blocks')

# Inputs that need specific control sources, not random wiring
CONTROL_INPUTS = {'kill_in', 'calm_in', 'error_in', 'dopamine_in', 'tonic_in', 'satisfy_in', 'clear_in'}

# Base weight when auto-wiring to each input type
INPUT_WEIGHTS = {
    'trigger_in': 10.0, 'trigger_a': 10.0, 'trigger_b': 10.0, 'trigger_c': 10.0,
    'stimulus_in': 8.0, 'fwd_sensory_in': 8.0, 'bwd_sensory_in': 8.0,
    'sensory_in': 8.0, 'anterior_in': 5.0, 'posterior_in': 5.0,
    'command_in': 5.0, 'drive_in': 5.0, 'inhibit_in': 5.0,
    'modulator_in': 3.0, 'modulator_fwd_in': 3.0, 'modulator_bwd_in': 3.0,
    'modulator_a_in': 3.0, 'modulator_b_in': 3.0,
    'association_in': 3.0,
    'input_a': 5.0, 'input_b': 5.0, 'sequence_a': 5.0, 'sequence_b': 5.0,
    # Brain region terminals
    'sensory_a_in': 8.0, 'sensory_b_in': 8.0,
    'mossy_in': 8.0, 'ec_in': 8.0, 'cortical_in': 8.0,
    'thalamic_in': 8.0, 'gate_in': 6.0,
    'error_in': 10.0, 'reward_in': 10.0, 'prediction_in': 5.0,
    'dopamine_in': 6.0, 'tonic_in': 5.0,
    # PFC + hypothalamus terminals
    'context_in': 8.0, 'clear_in': 10.0,
    'need_in': 5.0, 'satisfy_in': 10.0,
}

# Auto-timer rules: when a component has both a trigger/input and a kill/calm,
# add a delayed synapse so bistable states self-terminate.
# Format: (trigger_label, kill_label, weight, delay)
AUTO_TIMERS = {
    'emotional_state': ('trigger_in', 'calm_in', 20.0, 400),
    'working_memory_cell': ('trigger_in', 'kill_in', 20.0, 500),
    'bistable_motif': ('trigger_in', 'kill_in', 20.0, 500),
}


def find_component(name):
    """Find component JSON by name. Searches sub_components/ then components/."""
    # Exact path (with or without .json)
    for path in [name, name + '.json']:
        if os.path.exists(os.path.join(BLOCKS_DIR, path)):
            return path
    # Search by basename
    for folder in ['sub_components', 'components']:
        d = os.path.join(BLOCKS_DIR, folder)
        if os.path.exists(d):
            candidate = os.path.join(folder, name + '.json')
            if os.path.exists(os.path.join(BLOCKS_DIR, candidate)):
                return candidate
            candidate = os.path.join(folder, name)
            if os.path.exists(os.path.join(BLOCKS_DIR, candidate)):
                return candidate
    raise FileNotFoundError(f"Component not found: {name}")


def load_component(path):
    """Load a component JSON file."""
    with open(os.path.join(BLOCKS_DIR, path)) as f:
        return json.load(f)


def list_components():
    """Print all available components with terminal info."""
    for folder in ['sub_components', 'components']:
        d = os.path.join(BLOCKS_DIR, folder)
        if not os.path.exists(d):
            continue
        print(f"\n{folder}/")
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json'):
                continue
            with open(os.path.join(d, f)) as fh:
                data = json.load(fh)
            n = len(data.get('nodes', []))
            terms = data.get('terminals', [])
            ins = [str(t.get('label', t['id'])) for t in terms if t['bind_type'] == 'input']
            outs = [str(t.get('label', t['id'])) for t in terms if t['bind_type'] == 'output']
            print(f"  {f.replace('.json',''):25s} {n:2d}N  in:{ins}  out:{outs}")


def generate(recipe):
    """Generate a brain DB from a recipe dict."""
    seed = recipe.get('seed', 42)
    random.seed(seed)

    output_db = recipe.get('output', 'brains/generated.db')
    if not os.path.isabs(output_db):
        output_db = os.path.join(BASE, output_db)

    wiring = recipe.get('wiring', recipe.get('global_params', {}))
    max_distance = wiring.get('max_distance', 20.0)
    prob = wiring.get('probability', 0.3)
    weight_scale = wiring.get('weight_scale', 1.0)
    exclude = set(wiring.get('exclude_inputs', list(CONTROL_INPUTS)))
    inhib_frac = wiring.get('inhibition_fraction', 0.3)  # fraction of cross-wires that are inhibitory

    # Build instance list
    instances = []
    next_id = 1
    spacing = 12.0
    total = sum(c.get('count', 1) for c in recipe['components'])
    cols = max(1, int(math.ceil(math.sqrt(total))))
    idx = 0

    layer_spacing = recipe.get('layer_spacing', 50.0)

    for spec in recipe['components']:
        path = find_component(spec.get('path') or spec.get('type'))
        comp = load_component(path)
        count = spec.get('count', 1)
        layer = spec.get('layer', None)

        for copy in range(count):
            row, col = idx // cols, idx % cols
            ix, iz = col * spacing, row * spacing
            if layer is not None:
                iz = layer * layer_spacing + (copy % max(1, cols)) * spacing
                ix = (copy // max(1, cols)) * spacing

            id_map = {}
            for node in comp['nodes']:
                id_map[node['id']] = next_id
                next_id += 1

            terminals = []
            for t in comp.get('terminals', []):
                terminals.append({
                    'bind_type': t['bind_type'],
                    'label': str(t.get('label', t['id'])),
                    'neuron_ids': [id_map[nid] for nid in t['neuron_ids']],
                })

            instances.append({
                'name': comp['name'], 'copy': copy,
                'id_map': id_map, 'nodes': comp['nodes'],
                'paths': comp['paths'], 'terminals': terminals,
                'x': ix, 'z': iz,
            })
            idx += 1

    # Create DB
    if os.path.exists(output_db):
        os.remove(output_db)
    conn = create_brain_db(output_db)

    n_neurons = n_internal = n_cross = 0

    # Neurons + internal synapses
    for inst in instances:
        for node in inst['nodes']:
            gid = inst['id_map'][node['id']]
            add_neuron(conn, node['type'],
                       pos_x=node.get('x', 0) + inst['x'],
                       pos_z=node.get('z', 0) + inst['z'],
                       neuron_id=gid)
            params = node.get('params')
            if params:
                for k, v in params.items():
                    conn.execute(f"UPDATE neurons SET {k}=? WHERE id=?", (v, gid))
            n_neurons += 1

        for path in inst['paths']:
            add_synapse(conn, inst['id_map'][path['source']], inst['id_map'][path['target']],
                        path['weight'], path['delay'], path['synapse_type'],
                        params_override=path.get('params'))
            n_internal += 1

    # Auto-timers: add trigger→calm/kill delayed synapses for bistable components
    n_timers = 0
    if wiring.get('auto_timers', True):
        for inst in instances:
            timer_rule = AUTO_TIMERS.get(inst['name'])
            if not timer_rule:
                continue
            trig_label, kill_label, tw, td = timer_rule
            trig_neurons = kill_neurons = None
            for t in inst['terminals']:
                if t['label'] == trig_label:
                    trig_neurons = t['neuron_ids']
                if t['label'] == kill_label:
                    kill_neurons = t['neuron_ids']
            if trig_neurons and kill_neurons:
                # First trigger neuron → first kill neuron (one timer synapse)
                add_synapse(conn, trig_neurons[0], kill_neurons[0], tw, td, 'fixed')
                n_timers += 1

    # --- H1: Structured projections (layer-to-layer directed wiring) ---
    n_proj = 0
    for proj in recipe.get('projections', []):
        if 'from_type' not in proj:
            continue  # skip comment-only entries
        from_type = proj['from_type']
        to_type = proj['to_type']
        pw = proj.get('weight', 8.0) * weight_scale
        pp = proj.get('probability', 0.8)
        pdelay = proj.get('delay', 1)
        psyntype = proj.get('synapse_type', 'fixed')
        from_label = proj.get('from_terminal')  # optional: specific terminal
        to_label = proj.get('to_terminal')      # optional: specific terminal

        from_insts = [inst for inst in instances if inst['name'] == from_type]
        to_insts = [inst for inst in instances if inst['name'] == to_type]

        for fi in from_insts:
            for ti in to_insts:
                if random.random() > pp:
                    continue
                # Find output terminals (optionally filtered by label)
                out_terms = [t for t in fi['terminals'] if t['bind_type'] == 'output']
                if from_label:
                    out_terms = [t for t in out_terms if t['label'] == from_label]
                if to_label:
                    # Explicit terminal target — bypass exclude (exclude is for random wiring)
                    in_terms = [t for t in ti['terminals'] if t['bind_type'] == 'input'
                                and t['label'] == to_label]
                else:
                    in_terms = [t for t in ti['terminals'] if t['bind_type'] == 'input'
                                and t['label'] not in exclude]

                for ot in out_terms:
                    for it in in_terms:
                        src = random.choice(ot['neuron_ids'])
                        tgt = random.choice(it['neuron_ids'])
                        add_synapse(conn, src, tgt, pw, pdelay, psyntype,
                                    params_override=proj.get('params'))
                        n_proj += 1

    # --- Cross-wiring: output terminals → input terminals of nearby instances ---
    # H4: type-aware inhibition mode
    type_aware = wiring.get('type_aware_inhibition', False)

    # H5: per-type terminal protection — block specific terminals on specific
    # component types from receiving random cross-wires.
    # Format: {'component_name': ['terminal_label', ...]}
    protected_triggers = wiring.get('protected_triggers', {})

    for i, a in enumerate(instances):
        for j, b in enumerate(instances):
            if i == j:
                continue
            dist = math.sqrt((a['x'] - b['x'])**2 + (a['z'] - b['z'])**2)
            if dist > max_distance:
                continue

            for out_t in a['terminals']:
                if out_t['bind_type'] != 'output':
                    continue
                for in_t in b['terminals']:
                    if in_t['bind_type'] != 'input':
                        continue
                    if in_t['label'] in exclude:
                        continue
                    # H5: per-type terminal protection
                    if b['name'] in protected_triggers and in_t['label'] in protected_triggers[b['name']]:
                        continue

                    # Probability scales with proximity
                    p = prob * (1.0 - dist / max_distance)
                    if random.random() > p:
                        continue

                    # One random output neuron → one random input neuron (sparse)
                    src = random.choice(out_t['neuron_ids'])
                    tgt = random.choice(in_t['neuron_ids'])
                    base_w = INPUT_WEIGHTS.get(in_t['label'], 5.0) * weight_scale

                    # H4: type-aware inhibition
                    # Same type = competition (inhibitory)
                    # Different type = cooperation (excitatory)
                    if type_aware:
                        if a['name'] == b['name']:
                            w = -abs(base_w)  # same type: inhibit
                        else:
                            w = abs(base_w)    # different type: excite
                    else:
                        # Original: random fraction inhibitory
                        if random.random() < inhib_frac:
                            w = -abs(base_w)
                        else:
                            w = abs(base_w)
                    add_synapse(conn, src, tgt, w, 1, 'fixed')
                    n_cross += 1

    conn.commit()
    conn.close()

    proj_str = f" + {n_proj} projections" if n_proj else ""
    print(f"Generated: {output_db}")
    print(f"  {n_neurons} neurons, {n_internal} internal + {n_timers} timers + {n_cross} cross{proj_str} = {n_internal + n_timers + n_cross + n_proj} total synapses")
    print(f"  {len(instances)} instances from {len(recipe['components'])} component types")
    print(f"  Seed={seed}, max_dist={max_distance}, prob={prob}, weight_scale={weight_scale}")
    return output_db


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate brain from components')
    p.add_argument('--recipe', help='Recipe JSON file')
    p.add_argument('--components', help='name:count pairs, comma-separated')
    p.add_argument('--output', default='brains/generated.db')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-dist', type=float, default=20.0)
    p.add_argument('--probability', type=float, default=0.3)
    p.add_argument('--weight-scale', type=float, default=1.0)
    p.add_argument('--list', action='store_true', help='List available components')
    args = p.parse_args()

    if args.list:
        list_components()
        sys.exit(0)

    if args.recipe:
        rpath = args.recipe if os.path.isabs(args.recipe) else os.path.join(BASE, args.recipe)
        with open(rpath) as f:
            recipe = json.load(f)
    elif args.components:
        comps = []
        for spec in args.components.split(','):
            parts = spec.strip().rsplit(':', 1)
            comps.append({'path': parts[0].strip(), 'count': int(parts[1])})
        recipe = {
            'components': comps, 'seed': args.seed, 'output': args.output,
            'wiring': {
                'max_distance': args.max_dist,
                'probability': args.probability,
                'weight_scale': args.weight_scale,
            }
        }
    else:
        p.print_help()
        sys.exit(1)

    generate(recipe)
