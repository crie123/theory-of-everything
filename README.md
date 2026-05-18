# WTF — Work Field Theory Simulation Engine

> *A cosmological simulation engine modelling universe formation from neutrino plasma through galaxy assembly, based on SPH (Smoothed Particle Hydrodynamics) macro-particles and work-field node mechanics.*

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Cosmic Epochs](#cosmic-epochs)
5. [Classes](#classes)
6. [Physics Functions](#physics-functions)
7. [Hyperparameters](#hyperparameters)
8. [Step Loop](#step-loop)
9. [Collapse Detection](#collapse-detection)
10. [Known Constraints & Observations](#known-constraints--observations)

---

## Overview

WTF simulates cosmological evolution from first principles using two coupled systems:

- **SPH Clouds** — macro-particles representing gas clouds with collective thermodynamics (temperature, density, pressure). Each cloud represents N micro-particles.
- **Work Nodes** — abstract field sources that drive chemistry and structure formation. Analogous to gravitational potential wells / dark matter halos.

The simulation proceeds through 8 cosmic epochs automatically based on iteration count. Structure formation — neutrinos → hydrogen → stars → galaxies → black holes — emerges from the interaction of these two systems without hardcoded outcomes.

**Key insight from development:** without precisely tuned parameters, random formation is near zero. Stable structures require a controlled initial energy burst (node flythrough ignition) that locks material inside the newly formed universe bubble.

---

## Core Concepts

### Clouds vs Particles

The engine uses **Cloud** objects rather than point particles. Each Cloud is an SPH macro-particle with:

| Field | Description |
|-------|-------------|
| `pos` | Centre of mass position (3-vector) |
| `vel` | Bulk drift velocity |
| `mass` | Total enclosed mass |
| `T` | Temperature — internal energy proxy |
| `rho` | SPH density (recomputed each step) |
| `P` | Pressure = (γ−1)·ρ·T |
| `cs` | Sound speed = √(γ·P/ρ) |
| `el` | Dominant element: `nu e p n H He C O Fe Ni` |
| `kind` | `cloud \| star \| whitehole \| bh` |
| `N` | Number of micro-particles represented |

This means chemistry and Jeans collapse happen at the **cloud level**, not per-particle. A single cloud of H with sufficient mass and density collapses into a star.

### Work Nodes

Work nodes are field sources that catalyse chemistry and attract matter. They are not particles — they have no mass and don't interact with SPH forces. Two types exist:

- **Dominant nodes** (`primary`) — strong (strength ×6), mutually repulsive, act as proto-galactic dark matter halos. Separate to different spatial regions on creation.
- **Satellite nodes** (`secondary/tertiary/exotic`) — weaker, seeded by bubble interiors, eventually absorbed by nearest dominant.

Node strength decays slowly. Nodes deposit accumulated work into nearby clouds, lowering effective assembly thresholds.

### Energy Budget

The simulation has a finite **Framework Reservoir** (default 20M units). It feeds new neutrino clouds at a set rate. When the reservoir empties, no new material enters. Existing bubbles can still evolve. This models the finite energy of a closed system.

---

## Architecture

```
wtf_model.py          — physics engine (1400+ lines)
wtf_app.py            — Streamlit UI, step loop, visualisation
```

### Data flow per step

```
spatial_hash.build()          → O(1) neighbour queries for all subsequent ops
sph_density_pass()            → set rho, P, cs for all clouds (vectorised)
framework_feed()              → inject new nu clouds from reservoir
white.process()               → white hole processes nearby clouds
cloud.update()                → SPH forces + Hubble expansion + cooling
assemble()                    → epoch-aware chemistry: nu→e→p→H→He→...
accretion_fast()              → black holes consume nearby clouds
collide_all_fast()            → cloud–cloud collisions + fusion
merge_clouds()                → overlapping clouds merge (conserve mass/momentum)
split_cloud()                 → Jeans fragmentation of massive clouds
spawn_multiverse()            → node flythrough ignition → new universe bubble
internal_nodes()              → seed new work nodes inside existing bubbles
update_nodes()                → node drift (epoch-scaled) + decay
node_interactions_in_bubbles()→ dominant repulsion + satellite absorption
universe_evolution()          → bubble radius/energy evolution
cosmic_epoch_events()         → epoch-dispatched: recombination, stars, galaxies
universe_feedback()           → bubbles heat internal clouds
interact_universes()          → bubble gravity attracts nearby clouds
node_gravity()                → dominant nodes attract assembled matter
phase_walls() / space_decay() → boundary conditions
```

---

## Cosmic Epochs

Epochs advance automatically based on iteration count. Each epoch enables specific physics:

| ID | Start iter | Name | Key physics |
|----|-----------|------|-------------|
| 0 | 0 | 💥 Big Bang / QGP | Hot plasma, nodes at max drift speed, nu dominates |
| 1 | 30 | ⚛️ Nucleosynthesis | H→He rate ∝ ρ² (density-dependent); Hubble cooling ~a⁻² |
| 2 | 80 | 🌑 Recombination | p+e⁻→H enhanced by density; Hubble cooling ~a⁻¹ |
| 3 | 150 | 🌟 First Stars (Pop III) | Massive metal-free stars near dominant nodes |
| 4 | 220 | ☀️ Reionization | Star UV ionizes surrounding H clouds (H→p) |
| 5 | 300 | 🌌 First Galaxies | Galaxy assembly tracking begins; stability metric |
| 6 | 450 | 🔥 Cosmic Noon | Peak star formation; extra node seeding in bubbles |
| 7 | 650 | 🪐 Modern Universe | Slow evolution; dark energy dominance |

Node drift speed (`NODE_DRIFT_BY_EPOCH`) drops by ~20× from epoch 0 to epoch 3, modelling the transition from radiation-pressure-dominated to matter-dominated node anchoring.

---

## Classes

### `Cloud`

Base macro-particle. Represents a gas/plasma cloud.

```python
Cloud(pos, mass, radius, kind="cloud", el="nu", temperature=None, N=1)
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `update(others, dt, nodes, epoch)` | Apply SPH forces, Hubble drift, cooling |
| `cool(dt, epoch)` | Hubble + radiative cooling; γ switches at epoch 2 |
| `update_thermo(epoch)` | Recompute P, cs from current rho, T |
| `set_density_from_neighbors(pos_arr, mass_arr)` | Vectorised SPH density |
| `_pressure_visc_force(neighbors)` | SPH pressure gradient + Monaghan viscosity |
| `_gravity_force(neighbors)` | Long-range gravity from SPH neighbourhood |

`energy` is a property alias for `T` — kept for backward compatibility.

---

### `WhiteHole(Cloud)`

Central energy source. Processes clouds that enter `WHITE_RADIUS`:
- Ejects them outward with velocity boost
- Heats them (`T += 5` per contact)
- Converts black holes into cloud explosions (`explode_bh`)

```python
WhiteHole(pos=[0,0,0])
```

---

### `BlackHole(Cloud)`

Singularity object. Accretes nearby clouds, may explode if overheated.

```python
BlackHole(pos)
```

| Method | Description |
|--------|-------------|
| `accrete(b)` | Absorb cloud: add mass + thermal energy |
| `update_bh(world)` | Inner/outer work balance; trigger explosion at `BH_EXPLODE_E` |

Spawned automatically when local work density exceeds `CRITICAL_WORK_DENSITY` (default 120).

---

### `WorkNode`

Abstract field source. No mass, not an SPH particle.

```python
WorkNode(pos, strength, node_type="primary", origin="void")
```

| Field | Description |
|-------|-------------|
| `strength` | Field intensity; grows slightly with accumulated work |
| `node_type` | `primary` (dominant halo) / `secondary` / `tertiary` / `exotic` |
| `life` | Decay factor 0→1; primary nodes decay slowest |
| `drift_vel` | Spatial drift velocity; epoch-scaled |
| `_epoch` | Current epoch injected by `update_nodes()` |

Nodes are not updated by SPH. They drift independently, seeding chemistry fields wherever they pass through cloud clusters.

---

### `UniverseBubble`

A pocket universe formed by node flythrough ignition.

```python
UniverseBubble(center, btype="H")  # btype: "H" | "He" | "exotic"
```

| Field | Description |
|-------|-------------|
| `radius` | Grows when star_count ≥ 3, shrinks when energy depleted |
| `energy` | Fuel; drained by internal work + radius cost |
| `star_count` | Stars within `max(radius, 6.0)` |
| `stability` | 0–1; drops on star loss, rises on galaxy assembly |
| `peak_work_density` | Highest work density observed inside (collapse indicator) |
| `bh_id` | `id()` of central BH if collapse produced one |
| `collapse_count` | How many collapse events occurred |

Maximum simultaneous active bubbles: `MAX_BUBBLES = 12`.

---

### `SpatialHash`

Uniform grid for O(1) neighbour queries. Rebuilt every step.

```python
spatial_hash.build(bodies)
spatial_hash.query_radius(pos, radius)  → list of bodies
```

Cell size: 4.0 units. All SPH, gravity, collision, and accretion operations use this instead of O(n²) all-pairs search.

---

## Physics Functions

### SPH

| Function | Description |
|----------|-------------|
| `sph_density_pass(world)` | Global vectorised density estimation for all clouds. Sets `rho`, `P`, `cs`. Must be called once per step before `cloud.update()`. |
| `_sph_kernel_w_vec(dist, h)` | Cubic spline kernel W(r,h) — vectorised over distance array |
| `_sph_kernel_dw_vec(dist, h)` | dW/dq gradient term for pressure force calculation |

SPH smoothing length `SPH_H = 3.5`. Adiabatic index switches: `γ = 4/3` (epochs 0–1, radiation-dominated) → `γ = 5/3` (epoch 2+, monatomic gas).

---

### Chemistry

| Function | Description |
|----------|-------------|
| `assemble(b, nodes, epoch)` | Epoch-aware cloud chemistry. Advances element along chain `nu→e→p→H→He→...` based on temperature T and proximity to work nodes. Uses grace period (`ASSEMBLY_GRACE_STEPS`) to prevent instant dissociation. Heavy elements (H+) never decay back to nu outside node field. |
| `fuse(c1, c2)` | Nuclear fusion on cloud collision: advance c1 one step up the element chain |
| `collide(c1, c2)` | Cloud–cloud collision check; triggers fusion + thermal mixing |
| `collide_all_fast(world)` | Spatial-hash accelerated collision pass over all clouds |

**Assembly thresholds** (MeV-analogue, divided by node strength factor):

| Transition | Threshold |
|------------|-----------|
| nu → e | 0.782 |
| e → p | 1.293 |
| p → H | 9.592 |
| H → He | 8.7 × 0.8 (density-dependent rate) |

Burnout (cold dissociation): `e` decays if `T < 0.39`, `p` decays if `T < 0.65`. Hot plasma doesn't burn out — only cold matter below formation threshold.

---

### Cloud Lifecycle

| Function | Description |
|----------|-------------|
| `merge_clouds(world)` | Merge overlapping clouds (separation < `CLOUD_MERGE_FRAC × (r1+r2)`). Conserves mass, momentum, thermal energy. Immunity window: clouds with `_el_age < 0` are protected. |
| `split_cloud(world, epoch)` | Jeans fragmentation: clouds exceeding `CLOUD_SPLIT_MASS = 5000` split into two daughters at 50% mass each, with slight cooling. |
| `evaporate_small_clouds(world)` | Remove clouds below `CLOUD_MIN_MASS = 5`. |

---

### Structure Formation

| Function | Description |
|----------|-------------|
| `star_formation(world, bubbles, nodes, epoch)` | Form stars from H clouds near dominant nodes (wide capture radius, lower Jeans threshold) or satellite nodes (standard threshold). Pop III enabled at epoch ≥ 3. |
| `pop3_star_formation(world, bubbles, nodes, epoch)` | First massive metal-free stars: mass > 40, rho > 0.01, near any node. 1.5× mass boost, high ignition temperature. |
| `reionization_feedback(world, bubbles, epoch)` | Epoch 4+: stars emit UV heating to nearby H/He clouds. Photo-ionization: H → p when T exceeds threshold. |
| `galaxy_formation(world, bubbles, nodes, epoch, step)` | Track star clusters per bubble. Gravitationally bind stars to bubble centre. Detect collapse events. Seed extra nodes at Cosmic Noon (epoch 6). |

---

### Node System

| Function | Description |
|----------|-------------|
| `update_nodes(nodes, world, epoch)` | Update all nodes: inject epoch for drift scaling, compute local work density, decay life. |
| `node_interactions_in_bubbles(bubbles, nodes)` | Dominant nodes repel each other (exclusion radius 8). Satellite nodes within `NODE_RADIUS` of dominant are absorbed (+strength, -life). |
| `node_gravity(world, nodes)` | Dominant nodes gravitationally attract assembled matter (H, He, stars) within radius 12. H clouds very close to dominant node slowly accrete mass. |
| `work_field_fast(b, nodes)` | Vectorised work field force on cloud b from all alive nodes within `NODE_RADIUS`. |
| `local_work_density_fast(b, R=2.0)` | Work density at cloud position: sum of nearby `work` values / volume. Used for BH spawn threshold. |

---

### Universe / Multiverse

| Function | Description |
|----------|-------------|
| `spawn_multiverse(bubbles, world, nodes, step)` | **Node flythrough ignition**: fast-moving node sweeping through cloud cluster deposits work energy. If `node_strength × speed × nearby_clouds > BUBBLE_FORMATION_COST`, a bubble ignites. Captures all nearby clouds (energy lock-in), seeds 4 internal nodes. Cooldown: 1 bubble per 8 steps, max `MAX_BUBBLES = 12`. |
| `internal_nodes(bubbles, nodes)` | Each bubble probabilistically seeds new work nodes in its interior (`INTERNAL_NODE_RATE`). Exotic bubbles produce exotic nodes. |
| `universe_evolution(bubbles, world, nodes)` | Bubble energy budget: drain from internal work + radius cost. Grow if energy > 80, shrink otherwise. Mark dead at energy < 5. |
| `universe_feedback(bubbles, world)` | Bubble radiates heat into internal clouds, enabling chemistry even far from external nodes. |
| `interact_universes(bubbles, world)` | Internal clouds: light thermal pressure outward. External nearby clouds: gravitational attraction inward (up to 2.5× bubble radius). |
| `decay_dead_universe(u, nodes)` | Dead bubbles decay internal nodes at type-specific rates. |
| `framework_feed(world)` | Inject new neutrino clouds from the global reservoir at `FEED_RATE`. Reservoir depletes over time (`FRAMEWORK_DECAY = 0.00001`). |

---

### Black Holes

| Function | Description |
|----------|-------------|
| `spawn_black_holes(world, white)` | Check all clouds for BH spawn condition: `local_work_density > 120` AND `|vel| > 2.0`. Spawn one BH per step maximum. |
| `accretion_fast(world)` | All BHs accrete clouds within `bh.radius × 2` using spatial hash. |
| `merge_black_holes(world)` | BHs within `BH_MERGE_DIST = 1.2` merge (mass + thermal energy). |
| `explode_bh(bh, world)` | BH overheated (`T > BH_EXPLODE_E = 200`): explodes into 60 nu clouds. |

---

### Collapse Detection

```python
COLLAPSE_LOG  # global list of collapse event dicts
log_collapse(step, ctype, pos, mass, peak_density, galaxy_size)
```

Three collapse types detected automatically in `galaxy_formation()`:

| Type | Condition | Physical meaning |
|------|-----------|-----------------|
| `galactic_bh` | `peak_work_density ≥ 0.6 × CRITICAL` on galaxy dissolution | Central supermassive BH formed during collapse |
| `direct_collapse` | ≥ 2 stars lost, low peak density | Tidal disruption / direct mass loss without BH stage |
| `agn_flare` | Central BH `T > 0.7 × BH_EXPLODE_E` | AGN episode: BH accretes and flares |

Each event record contains: `step`, `type`, `pos`, `mass`, `peak_density`, `galaxy_size`.

---

### Boundary Conditions

| Function | Description |
|----------|-------------|
| `phase_walls(b)` | Elastic reflection at `PHASE_WALL_R = 25`: `vel *= -0.6` |
| `space_decay(world)` | Hard remove at `SPACE_DECAY_R = 35` |

---

## Hyperparameters

### Critical — changing these fundamentally alters behaviour

| Parameter | Default | Effect |
|-----------|---------|--------|
| `G` | 0.2 | Gravitational constant. Higher → faster collapse, more BHs |
| `H` | 0.0005 | Hubble expansion. Higher → clouds disperse faster, fewer structures |
| `FEED_ENERGY` | 400.0 | Initial cloud temperature. Too low: assembly stalls. Too high: everything dissociates |
| `CRITICAL_WORK_DENSITY` | 120.0 | BH formation threshold. Lower → more BHs, earlier collapse |
| `BUBBLE_FORMATION_COST` | 150 | Node flythrough energy for universe ignition. Lower → more bubbles |
| `NODE_STRENGTH_BASE` | 63.0 | Base node field strength. Scales all chemistry rates |
| `N_DOMINANT_NODES` | 4 | Number of proto-galactic halos. Equals max simultaneous galaxies |

### Tunable — affects timing and rates

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SPH_H` | 3.5 | SPH smoothing length. Larger → more diffuse pressure, slower collapse |
| `SPH_GAMMA` | 5/3 | Adiabatic index (matter era). Lower → softer gas, more compressible |
| `CLOUD_COOL_RATE` | 0.0008 | Cooling rate per step. Higher → faster temperature drop → earlier H, stars |
| `CLOUD_SPLIT_MASS` | 5000 | Jeans mass for fragmentation |
| `NODE_RADIUS` | 2.5 | Node field capture radius. Larger → more clouds assembled per step |
| `INTERNAL_NODE_RATE` | 0.002 | Probability of new node per bubble per step |
| `MAX_BUBBLES` | 12 | Hard cap on simultaneous universe bubbles |
| `ASSEMBLY_GRACE_STEPS` | e:8 p:15 H:40 He:80 | Minimum lifetime before element can dissociate |

### Epoch timing

```python
COSMIC_EPOCHS[1]["iteration_start"] = 30   # Nucleosynthesis
COSMIC_EPOCHS[2]["iteration_start"] = 80   # Recombination
COSMIC_EPOCHS[3]["iteration_start"] = 150  # First Stars
COSMIC_EPOCHS[4]["iteration_start"] = 220  # Reionization
COSMIC_EPOCHS[5]["iteration_start"] = 300  # First Galaxies
COSMIC_EPOCHS[6]["iteration_start"] = 450  # Cosmic Noon
COSMIC_EPOCHS[7]["iteration_start"] = 650  # Modern Universe
```

Node drift speed by epoch:
```python
NODE_DRIFT_BY_EPOCH = {0: 0.08, 1: 0.05, 2: 0.012, 3: 0.004,
                       4: 0.002, 5: 0.001, 6: 0.001, 7: 0.001}
```

---

## Step Loop

Minimal headless step loop (no Streamlit):

```python
import wtf_model as M
import numpy as np

world, white, nodes = M.create_universe()
bubbles = []
dt = 0.05

for step in range(1000):
    epoch = M.get_current_epoch(step)
    M._current_epoch = epoch

    # 1. Spatial index
    M.spatial_hash.build(world)

    # 2. SPH density (must come before update)
    M.sph_density_pass(world)

    # 3. Feed + white hole
    M.framework_feed(world)
    white.process(world)

    # 4. Dynamics
    for b in world[:]:
        if isinstance(b, M.WhiteHole): continue
        if isinstance(b, M.BlackHole): b.update_bh(world)
        else: b.update(world, dt, nodes, use_spatial=True, epoch=epoch)

    # 5. Chemistry
    for b in world:
        if not isinstance(b, (M.WhiteHole, M.BlackHole)):
            M.assemble(b, nodes, epoch=epoch)

    # 6. Collisions + cloud lifecycle
    M.accretion_fast(world)
    world[:] = M.collide_all_fast(world)
    M.merge_clouds(world)
    M.split_cloud(world, epoch)
    M.evaporate_small_clouds(world)

    # 7. Black holes
    M.merge_black_holes(world)
    M.spawn_black_holes(world, white)

    # 8. Universe / multiverse
    M.spawn_multiverse(bubbles, world, nodes, step=step)
    M.internal_nodes(bubbles, nodes)
    M.update_nodes(nodes, world, epoch=epoch)
    M.node_interactions_in_bubbles(bubbles, nodes)
    M.universe_evolution(bubbles, world, nodes)
    for u in bubbles:
        M.decay_dead_universe(u, nodes)

    # 9. Epoch-specific events (stars, galaxies, reionization)
    M.cosmic_epoch_events(world, bubbles, nodes, step)

    # 10. Feedback + gravity
    M.universe_feedback(bubbles, world)
    M.interact_universes(bubbles, world)
    M.node_gravity(world, nodes)

    # 11. Boundary conditions
    for b in world:
        M.phase_walls(b)
    M.space_decay(world)
```

---

## Collapse Detection

The engine logs galaxy collapse events automatically. Access at any time:

```python
import wtf_model as M

# All events
for event in M.COLLAPSE_LOG:
    print(event["step"], event["type"], event["peak_density"])

# Filter by type
bh_collapses = [e for e in M.COLLAPSE_LOG if e["type"] == "galactic_bh"]
direct       = [e for e in M.COLLAPSE_LOG if e["type"] == "direct_collapse"]
agn          = [e for e in M.COLLAPSE_LOG if e["type"] == "agn_flare"]
```

The peak work density at collapse is the key diagnostic:
- `peak_density ≥ 72` (60% of critical 120) → `galactic_bh` — central BH confirmed
- `peak_density < 72` + rapid star loss → `direct_collapse` — tidal disruption or material loss without BH stage
- This distinction matches the observed behaviour at iteration ~1488: small singularity peak at galaxy dissolution = galactic BH, not full singularity event

---

## Known Constraints & Observations

### What the model captures well
- Thermal death / heat death of the universe (confirmed, stable without new energy input)
- Epoch-sequenced structure formation (neutrinos → H → He → stars → galaxies in correct order)
- Galaxy instability and disruption with central BH formation
- Inflationary-like expansion followed by contraction when internal nodes consume all material
- Edge-preferential node distribution after expansion (consistent with cosmic web observations)
- Bubble overlap regions with differing internal structure (proto-multiverse)

### Fundamental limitations

**Earth-based physics only.** All thresholds, element chains, and interaction constants derive from observed particle physics in our local universe. The model cannot represent:

- Alternative particle physics in overlapping bubble universes where physical constants may differ
- Sub-Planck-scale behaviour
- Strong/electroweak unification physics
- Quantum effects (the model is classical SPH)

**Parameter sensitivity.** Structure formation is strongly parameter-dependent. Without the 4-dominant-node hierarchy and energy lock-in on bubble ignition, random formation probability approaches zero. This may reflect a real fine-tuning problem, or may indicate missing physics that provides robustness.

**Galaxy stability.** Galaxies of 7–12 stars form and dissolve cyclically. The disruption mechanism produces work density peaks consistent with central BH formation. Whether this represents real galactic merger dynamics or a model artefact is not definitively resolved.

**Scale.** One simulation "unit" is not calibrated to physical units (AU, parsec, MeV). The element thresholds (`TH_NU_E = 0.782`, etc.) are set to MeV-analogue values but the spatial and temporal scales are arbitrary. Calibration to physical units would require matching the epoch timing to actual cosmological timescales.

---

## File Structure

```
wtf_model.py    Physics engine — all classes, functions, constants
wtf_app.py      Streamlit UI — step loop, visualisation, state management
README.md       This file
```

### Streamlit UI features
- **Step / Run / Batch / Live** modes
- **Epoch banner** with progress to next epoch
- **6-panel graph**: total work, BH count, work density threshold, node evolution, reservoir level, stars & galaxies per epoch
- **💥 Collapses tab**: timeline of all collapse events with type, peak density, galaxy size
- **State tree**: full simulation state readable at any step
- **3D view**: clouds, nodes, bubbles in 3D space
- **Save / Load** snapshots via pickle