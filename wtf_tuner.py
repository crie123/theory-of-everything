"""
WTF Universe — Hyperparameter Tuner
=====================================
Goal: find node configs where simulation naturally produces
the correct cosmic epoch sequence:
  0 → Big Bang (high energy, only nu/e/p)
  1 → Nucleosynthesis (H, He appear)
  2 → Recombination (neutral H dominates)
  3 → First Stars (star kind appears)
  4 → Reionization (star UV ionizes H→p)
  5 → First Galaxies (bubbles with 3+ stars)
  6 → Cosmic Noon (peak star count)
  7 → Modern Universe (stable, low BH rate)

Score = how closely the sim profile matches the TARGET timeline.
"""

import numpy as np
import copy
import json
import itertools
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple
import sys

# ----------------------------------------------------------------
# TARGET EPOCH PROFILE
# Each entry: (iter_window_start, iter_window_end, metric, target_value, weight)
# metric can be: "frac_H", "frac_He", "star_count", "galaxy_count",
#                "frac_nu", "bh_count", "bubble_count", "frac_p"
# ----------------------------------------------------------------
TARGET_PROFILE = [
    # Epoch 0: Big Bang — mostly neutrinos, no stars, no BH yet
    (0,   25,  "frac_nu",    0.85, 2.0),
    (0,   25,  "star_count", 0.0,  1.5),

    # Epoch 1: Nucleosynthesis — H and He should start appearing
    (25,  70,  "frac_H",     0.30, 2.0),
    (25,  70,  "frac_He",    0.05, 1.5),
    (25,  70,  "frac_nu",    0.40, 1.0),

    # Epoch 2: Recombination — neutral H dominant, p fraction drops
    (70,  140, "frac_H",     0.55, 2.5),
    (70,  140, "frac_p",     0.10, 1.5),

    # Epoch 3: First Stars — stars appear inside bubbles
    (140, 210, "star_count", 3.0,  3.0),
    (140, 210, "bubble_count", 1.0, 2.0),

    # Epoch 4: Reionization — some p appear again (UV ionization)
    (210, 280, "frac_p",     0.15, 2.0),
    (210, 280, "star_count", 5.0,  2.0),

    # Epoch 5: First Galaxies
    (280, 380, "galaxy_count", 1.0, 3.0),
    (280, 380, "star_count",  8.0,  2.0),

    # Epoch 6: Cosmic Noon — peak everything
    (380, 500, "star_count",  12.0, 2.5),
    (380, 500, "bubble_count", 3.0, 2.0),
    (380, 500, "bh_count",     2.0, 1.5),

    # Epoch 7: Modern — settling down
    (500, 650, "frac_He",    0.20, 2.0),
    (500, 650, "frac_H",     0.40, 1.5),
]

TOTAL_STEPS  = 660  # Full simulation run
SAMPLE_EVERY = 5    # Sample metrics every N steps (speed)
MAX_PARTICLES = 400 # Hard cap — kill excess to keep each run fast

import multiprocessing as mp

def _worker(args):
    """Worker for parallel evaluation"""
    cfg, steps, sample_every = args
    try:
        hist = run_simulation(cfg, steps, sample_every)
        sc   = score_history(hist, TARGET_PROFILE)
        return sc, cfg, hist
    except Exception as e:
        return 1e9, cfg, {}


# ═══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER SPACE — parameter grids & search configuration
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class NodeConfig:
    """Initial node layout config"""
    count: int          # number of initial nodes
    positions: str      # "center", "ring", "random", "grid"
    strength: float     # base strength multiplier
    spread: float       # spatial spread

@dataclass
class HyperConfig:
    # Node layout
    node_count: int
    node_positions: str   # center / ring / random / grid
    node_strength: float  # multiplier on NODE_STRENGTH_BASE
    node_spread: float    # spatial spread (multiplier on 1.0)
    node_radius: float    # NODE_RADIUS

    # Energy
    feed_energy: float    # FEED_ENERGY
    feed_rate: float      # FEED_RATE
    feed_particles: int   # FEED_PARTICLES_PER_TICK
    framework_reservoir: float  # as fraction of 1e7

    # Physics
    g_const: float        # G multiplier
    h_const: float        # H (expansion) multiplier
    critical_work_density: float
    bubble_formation_cost: float
    internal_node_rate: float
    universe_feedback: float

    # Label (set after creation)
    label: str = ""

    def describe(self):
        return (f"nodes={self.node_count}@{self.node_positions}(str={self.node_strength:.1f},"
                f"r={self.node_radius:.1f}) | feed={self.feed_energy:.0f}x{self.feed_particles}"
                f"@{self.feed_rate:.1f} | G={self.g_const:.3f} H={self.h_const:.4f}"
                f" BWD={self.critical_work_density:.0f} | bub_cost={self.bubble_formation_cost:.0f}"
                f" fb={self.universe_feedback:.3f}")


# ----------------------------------------------------------------
# PARAMETER GRID
# ----------------------------------------------------------------
PARAM_GRID = {
    "node_count":            [1, 3, 6, 12],
    "node_positions":        ["center", "ring", "random", "grid"],
    "node_strength":         [0.5, 1.0, 2.0, 3.5],
    "node_spread":           [0.5, 1.5, 3.0],
    "node_radius":           [1.0, 1.5, 2.5, 4.0],
    "feed_energy":           [50.0, 100.0, 200.0, 400.0],
    "feed_rate":             [0.3, 0.6, 0.8, 1.0],
    "feed_particles":        [1, 2, 3, 5],
    "framework_reservoir":   [0.5, 1.0, 2.0, 5.0],
    "g_const":               [0.02, 0.05, 0.10, 0.20],
    "h_const":               [0.0005, 0.002, 0.005, 0.01],
    "critical_work_density": [40.0, 80.0, 120.0, 200.0],
    "bubble_formation_cost": [500, 2000, 5000, 10000],
    "internal_node_rate":    [0.0005, 0.002, 0.008, 0.02],
    "universe_feedback":     [0.005, 0.02, 0.05, 0.1],
}

# Total search space size
total_space = 1
for v in PARAM_GRID.values():
    total_space *= len(v)
print(f"Total search space: {total_space:,} configs")


# ----------------------------------------------------------------
# SIMULATION RUNNER (headless, no streamlit)
# ----------------------------------------------------------------
def make_node_positions(count, layout, spread):
    """Generate node positions based on layout strategy"""
    positions = []
    if layout == "center":
        for _ in range(count):
            positions.append(np.random.randn(3) * 0.1 * spread)
    elif layout == "ring":
        for i in range(count):
            angle = 2 * np.pi * i / max(count, 1)
            r = 2.0 * spread
            positions.append(np.array([r * np.cos(angle), r * np.sin(angle), 0]))
    elif layout == "random":
        for _ in range(count):
            positions.append(np.random.randn(3) * 3.0 * spread)
    elif layout == "grid":
        side = max(1, int(np.ceil(count ** (1/3))))
        pts = list(itertools.product(range(side), repeat=3))[:count]
        for p in pts:
            positions.append((np.array(p, float) - side/2) * 2.0 * spread)
        while len(positions) < count:
            positions.append(np.random.randn(3) * spread)
    return positions


def run_simulation(cfg: HyperConfig, steps: int, sample_every: int) -> Dict:
    """
    Run simulation with given config. Returns time-series of metrics.
    Pure Python/numpy, no streamlit.
    """
    import importlib
    import wtf_model as M
    importlib.reload(M)  # Fresh state

    # Override globals
    M.G = 0.05 * cfg.g_const / 0.05
    M.G = cfg.g_const
    M.H = cfg.h_const
    M.FEED_ENERGY = cfg.feed_energy
    M.FEED_RATE = cfg.feed_rate
    M.FEED_PARTICLES_PER_TICK = cfg.feed_particles
    M.FRAMEWORK_RESERVOIR = cfg.framework_reservoir * 1e7
    M._framework_reservoir = cfg.framework_reservoir * 1e7
    M.CRITICAL_WORK_DENSITY = cfg.critical_work_density
    M.BUBBLE_FORMATION_COST = cfg.bubble_formation_cost
    M.INTERNAL_NODE_RATE = cfg.internal_node_rate
    M.UNIVERSE_WORK_FEEDBACK = cfg.universe_feedback
    M.NODE_RADIUS = cfg.node_radius
    M.NODE_STRENGTH_BASE = 18.0 * cfg.node_strength

    # Create world
    world, white, _ = M.create_universe()

    # Override nodes with config layout
    node_positions = make_node_positions(cfg.node_count, cfg.node_positions, cfg.node_spread)
    nodes = []
    for pos in node_positions:
        nodes.append(M.WorkNode(np.array(pos, float), 18.0 * cfg.node_strength))

    bubbles = []
    dt = 0.05

    # Metrics time series
    history = {
        "frac_nu": [], "frac_H": [], "frac_He": [], "frac_p": [],
        "star_count": [], "galaxy_count": [], "bh_count": [], "bubble_count": [],
        "total_particles": [], "epoch": [],
    }

    for step_i in range(steps):
        # Rebuild spatial hash
        M.spatial_hash.build(world)

        M.framework_feed(world)
        white.process(world)

        for b in world[:]:
            if isinstance(b, M.WhiteHole):
                continue
            if isinstance(b, M.BlackHole):
                b.update_bh(world)
            else:
                b.update(world, dt, nodes, use_spatial=True)
                M.phase_walls(b)

        epoch = M.get_current_epoch(step_i)
        for b in world:
            if not isinstance(b, (M.WhiteHole, M.BlackHole)):
                M.assemble(b, nodes, epoch=epoch)

        M.accretion_fast(world)
        world[:] = M.collide_all_fast(world)
        M.merge_black_holes(world)
        M.spawn_black_holes(world, white)
        M.spawn_multiverse(bubbles, world, nodes)
        M.internal_nodes(bubbles, nodes)
        M.update_nodes(nodes, world)
        M.node_interactions_in_bubbles(bubbles, nodes)
        M.universe_evolution(bubbles, world, nodes)
        for u in bubbles:
            M.decay_dead_universe(u, nodes)
        M.cosmic_epoch_events(world, bubbles, nodes, step_i)
        M.universe_feedback(bubbles, world)
        M.interact_universes(bubbles, world)
        M.space_decay(world)

        # Hard particle cap — keeps each trial fast
        non_special = [b for b in world if not isinstance(b, (M.WhiteHole, M.BlackHole))]
        if len(non_special) > MAX_PARTICLES:
            excess = non_special[MAX_PARTICLES:]
            for b in excess:
                if b in world and b.kind != "star":
                    world.remove(b)

        # Sample metrics
        if step_i % sample_every == 0:
            particles = [b for b in world if hasattr(b, 'el') and not isinstance(b, (M.WhiteHole, M.BlackHole))]
            n = max(len(particles), 1)
            counts = {}
            for b in particles:
                counts[b.el] = counts.get(b.el, 0) + 1

            history["frac_nu"].append((step_i, counts.get("nu", 0) / n))
            history["frac_H"].append((step_i, counts.get("H", 0) / n))
            history["frac_He"].append((step_i, counts.get("He", 0) / n))
            history["frac_p"].append((step_i, counts.get("p", 0) / n))
            history["star_count"].append((step_i, sum(1 for b in world if b.kind == "star")))
            bh_c = sum(1 for b in world if isinstance(b, M.BlackHole))
            history["bh_count"].append((step_i, bh_c))
            history["bubble_count"].append((step_i, len(bubbles)))
            for b in bubbles:
                local_stars = [p for p in world
                               if p.kind == "star" and np.linalg.norm(p.pos - b.center) < b.radius]
                b.star_count = len(local_stars)
            gal_c = sum(1 for b in bubbles if not b.dead and b.star_count >= 3)
            history["galaxy_count"].append((step_i, gal_c))
            history["total_particles"].append((step_i, len(world)))
            history["epoch"].append((step_i, epoch))

    return history


# ----------------------------------------------------------------
# SCORING
# ----------------------------------------------------------------
def score_history(history: Dict, target_profile: list) -> float:
    """
    Compute how well history matches target profile.
    Returns score (lower = better, 0 = perfect).
    """
    total_penalty = 0.0
    total_weight = 0.0

    for (t_start, t_end, metric, target_val, weight) in target_profile:
        if metric not in history:
            total_penalty += weight * target_val**2
            total_weight += weight
            continue

        # Get samples in the iteration window
        samples = [(t, v) for (t, v) in history[metric]
                   if t_start <= t < t_end]
        if not samples:
            total_penalty += weight * target_val**2
            total_weight += weight
            continue

        avg_val = np.mean([v for _, v in samples])

        # Normalized squared error
        denom = max(abs(target_val), 0.5)
        err = ((avg_val - target_val) / denom) ** 2
        total_penalty += weight * err
        total_weight += weight

    return total_penalty / total_weight if total_weight > 0 else 1e9


# ----------------------------------------------------------------
# RANDOM SEARCH (much faster than grid for high-dim spaces)
# ----------------------------------------------------------------
def random_config(rng=None) -> HyperConfig:
    if rng is None:
        rng = np.random.default_rng()

    def pick(key):
        vals = PARAM_GRID[key]
        return vals[rng.integers(len(vals))]

    return HyperConfig(
        node_count=pick("node_count"),
        node_positions=pick("node_positions"),
        node_strength=pick("node_strength"),
        node_spread=pick("node_spread"),
        node_radius=pick("node_radius"),
        feed_energy=pick("feed_energy"),
        feed_rate=pick("feed_rate"),
        feed_particles=pick("feed_particles"),
        framework_reservoir=pick("framework_reservoir"),
        g_const=pick("g_const"),
        h_const=pick("h_const"),
        critical_work_density=pick("critical_work_density"),
        bubble_formation_cost=pick("bubble_formation_cost"),
        internal_node_rate=pick("internal_node_rate"),
        universe_feedback=pick("universe_feedback"),
    )


def neighbor_config(cfg: HyperConfig, rng, n_perturb=3) -> HyperConfig:
    """Perturb N random params from a known good config"""
    d = asdict(cfg)
    keys = [k for k in PARAM_GRID.keys()]
    chosen = rng.choice(keys, size=min(n_perturb, len(keys)), replace=False)
    for k in chosen:
        vals = PARAM_GRID[k]
        d[k] = vals[rng.integers(len(vals))]
    d.pop("label", None)
    return HyperConfig(**d)


# ----------------------------------------------------------------
# MAIN SEARCH LOOP
# ----------------------------------------------------------------
def run_search(n_random=40, n_hillclimb=30, steps=TOTAL_STEPS,
               sample_every=SAMPLE_EVERY, seed=42):

    rng = np.random.default_rng(seed)
    results = []

    print(f"\n{'='*60}")
    print(f"  WTF Universe Hyperparameter Search")
    print(f"  Random trials: {n_random} | Hill-climb: {n_hillclimb}")
    print(f"  Steps per run: {steps} | Sample every: {sample_every}")
    print(f"{'='*60}\n")

    # Phase 1: Random search
    print("[ Phase 1: Random Search ]")
    for i in range(n_random):
        cfg = random_config(rng)
        t0 = time.time()
        try:
            hist = run_simulation(cfg, steps, sample_every)
            sc = score_history(hist, TARGET_PROFILE)
        except Exception as e:
            sc = 1e9
            hist = {}
            print(f"  [{i+1:2d}/{n_random}] ERROR: {e}")
            continue
        elapsed = time.time() - t0
        cfg.label = f"rand_{i}"
        results.append((sc, cfg, hist))
        print(f"  [{i+1:2d}/{n_random}] score={sc:.4f}  ({elapsed:.1f}s)  {cfg.describe()}")

    results.sort(key=lambda x: x[0])
    print(f"\n  Top random: score={results[0][0]:.4f}")

    # Phase 2: Hill-climbing from top-5 random configs
    print(f"\n[ Phase 2: Hill-Climbing from top-5 ]")
    top5 = results[:5]
    hc_results = []

    for rank, (base_score, base_cfg, _) in enumerate(top5):
        print(f"\n  Climbing from rank-{rank+1} (score={base_score:.4f})")
        current_score = base_score
        current_cfg = base_cfg

        for j in range(n_hillclimb):
            n_p = rng.integers(1, 5)  # perturb 1-4 params
            candidate = neighbor_config(current_cfg, rng, n_perturb=int(n_p))
            t0 = time.time()
            try:
                hist = run_simulation(candidate, steps, sample_every)
                sc = score_history(hist, TARGET_PROFILE)
            except Exception as e:
                print(f"    [{j+1:2d}] ERROR: {e}")
                continue
            elapsed = time.time() - t0

            improvement = current_score - sc
            if sc < current_score:
                current_score = sc
                current_cfg = candidate
                marker = "✓ improved"
            else:
                marker = ""

            candidate.label = f"hc_r{rank}_{j}"
            hc_results.append((sc, candidate, hist))
            print(f"    [{j+1:2d}/{n_hillclimb}] score={sc:.4f} Δ={improvement:+.4f}  "
                  f"({elapsed:.1f}s) {marker}")

        hc_results.append((current_score, current_cfg, None))

    # Combine & sort
    all_results = results + hc_results
    all_results.sort(key=lambda x: x[0])

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — Top 10 Configurations")
    print(f"{'='*60}")
    for rank, (sc, cfg, hist) in enumerate(all_results[:10]):
        print(f"\n  #{rank+1}  score={sc:.4f}  [{cfg.label}]")
        print(f"       {cfg.describe()}")

    return all_results


# ----------------------------------------------------------------
# ANALYSIS: What metrics drove each epoch in best config
# ----------------------------------------------------------------
def analyze_best(results, top_n=3):
    print(f"\n{'='*60}")
    print(f"  Epoch Analysis — Top {top_n} Configs")
    print(f"{'='*60}")

    for rank, (sc, cfg, hist) in enumerate(results[:top_n]):
        if hist is None:
            continue
        print(f"\n  Config #{rank+1}  score={sc:.4f}")
        print(f"  {cfg.describe()}\n")

        epoch_labels = {
            0: "Big Bang", 1: "Nucleosynthesis", 2: "Recombination",
            3: "First Stars", 4: "Reionization", 5: "First Galaxies",
            6: "Cosmic Noon", 7: "Modern"
        }
        epoch_starts = {0: 0, 1: 30, 2: 80, 3: 150, 4: 220, 5: 300, 6: 450, 7: 650}
        epoch_ends   = {k: v for k, v in list(zip(
            list(epoch_starts.keys()),
            list(epoch_starts.values())[1:] + [TOTAL_STEPS]
        ))}

        rows = []
        for eid in range(8):
            t0 = epoch_starts[eid]
            t1 = epoch_ends[eid]

            def avg_in(metric):
                s = [(t, v) for (t, v) in hist.get(metric, []) if t0 <= t < t1]
                return np.mean([v for _, v in s]) if s else 0.0

            rows.append({
                "Epoch": f"{eid}: {epoch_labels[eid]}",
                "nu%":    f"{avg_in('frac_nu')*100:.1f}",
                "H%":     f"{avg_in('frac_H')*100:.1f}",
                "He%":    f"{avg_in('frac_He')*100:.1f}",
                "p%":     f"{avg_in('frac_p')*100:.1f}",
                "Stars":  f"{avg_in('star_count'):.1f}",
                "Galaxies": f"{avg_in('galaxy_count'):.1f}",
                "BH":     f"{avg_in('bh_count'):.1f}",
                "Bubbles": f"{avg_in('bubble_count'):.1f}",
            })

        # Print table
        headers = ["Epoch", "nu%", "H%", "He%", "p%", "Stars", "Galaxies", "BH", "Bubbles"]
        widths = [25, 6, 6, 6, 6, 7, 9, 6, 8]
        header_line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        print(header_line)
        print("  " + "-" * (sum(widths) + len(widths) * 2))
        for row in rows:
            line = "  " + "  ".join(row[h].ljust(w) for h, w in zip(headers, widths))
            print(line)

    return results[:top_n]


# ----------------------------------------------------------------
# SAVE RESULTS TO JSON
# ----------------------------------------------------------------
def save_results(results, path="wtf_tuner_results.json"):
    output = []
    for rank, (sc, cfg, hist) in enumerate(results[:20]):
        entry = {
            "rank": rank + 1,
            "score": float(sc),
            "label": cfg.label,
            "config": {k: v for k, v in asdict(cfg).items() if k != "label"},
            "has_history": hist is not None,
        }
        # Save summary metrics per epoch window
        if hist:
            epoch_summary = {}
            for metric in ["frac_nu", "frac_H", "frac_He", "frac_p",
                           "star_count", "galaxy_count", "bh_count", "bubble_count"]:
                epoch_summary[metric] = [(int(t), float(v)) for t, v in hist.get(metric, [])]
            entry["epoch_summary"] = epoch_summary
        output.append(entry)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved results to {path}")


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--random",    type=int, default=30,   help="Random search trials")
    parser.add_argument("--hillclimb", type=int, default=20,   help="Hill-climb steps per seed")
    parser.add_argument("--steps",     type=int, default=660,  help="Sim steps per trial")
    parser.add_argument("--sample",    type=int, default=5,    help="Sample every N steps")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       type=str, default="wtf_tuner_results.json")
    args = parser.parse_args()

    t_start = time.time()
    results = run_search(
        n_random=args.random,
        n_hillclimb=args.hillclimb,
        steps=args.steps,
        sample_every=args.sample,
        seed=args.seed,
    )
    top = analyze_best(results, top_n=3)
    save_results(results, path=args.out)

    print(f"\n  Total search time: {time.time() - t_start:.1f}s")
    print(f"\n  Best config:")
    best_sc, best_cfg, _ = results[0]
    print(f"    score = {best_sc:.4f}")
    print(f"    {best_cfg.describe()}")
    print(f"\n  Copy these into wtf_model.py / create_universe():")
    print(f"    G                    = {best_cfg.g_const}")
    print(f"    H                    = {best_cfg.h_const}")
    print(f"    FEED_ENERGY          = {best_cfg.feed_energy}")
    print(f"    FEED_RATE            = {best_cfg.feed_rate}")
    print(f"    FEED_PARTICLES_PER_TICK = {best_cfg.feed_particles}")
    print(f"    FRAMEWORK_RESERVOIR  = {best_cfg.framework_reservoir * 1e7:.0f}")
    print(f"    CRITICAL_WORK_DENSITY = {best_cfg.critical_work_density}")
    print(f"    BUBBLE_FORMATION_COST = {best_cfg.bubble_formation_cost}")
    print(f"    INTERNAL_NODE_RATE   = {best_cfg.internal_node_rate}")
    print(f"    UNIVERSE_WORK_FEEDBACK = {best_cfg.universe_feedback}")
    print(f"    NODE_RADIUS          = {best_cfg.node_radius}")
    print(f"    NODE_STRENGTH_BASE   = {18.0 * best_cfg.node_strength:.1f}")
    print(f"\n    # Nodes in create_universe():")
    print(f"    #   count={best_cfg.node_count}, layout='{best_cfg.node_positions}',")
    print(f"    #   spread={best_cfg.node_spread}, strength_mult={best_cfg.node_strength}")