import numpy as np
from collections import defaultdict

# ============================================================
# GLOBAL PHYSICS CONSTANTS
# ============================================================
G      = 0.2       # Gravitational constant
H      = 0.0005    # Hubble expansion rate
C_CRIT = 0.98      # Coherence threshold → frozen state

# Energy thresholds (MeV-analogue)
TH_NU_E  = 0.782
TH_E_P   = 1.293
TH_PN    = 9.592
TH_STAB  = 8.7

# Framework feeding
FRAMEWORK_FEED          = True
FEED_RATE               = 0.3
FEED_ENERGY             = 400.0
FEED_PARTICLES_PER_TICK = 1

WHITE_RADIUS = 3.0

# Black hole
CRITICAL_WORK_DENSITY = 120.0
COLLAPSE_SPEED        = 0.25
BH_MERGE_DIST         = 1.2
BH_EXPLODE_E          = 200
BH_COLLAPSE_W         = 50
WORK_GRADIENT_MIN     = 2.0

# Multiverse / space
MULTIVERSE_RATE = 0.0005
PHASE_WALL_R    = 25
SPACE_DECAY_R   = 35

# Bubble / thermodynamics
BUBBLE_FORMATION_COST  = 150
BUBBLE_MIN_ENERGY      = 300
INTERNAL_NODE_RATE     = 0.002
INTERNAL_NODE_STRENGTH = 1.0

# Node params
NODE_BURN_RATE     = 0.0005
NODE_MAX_AGE       = 4000
NODE_BASE_COUNT    = 4
NODE_RADIUS        = 2.5   # raised: ensure nu clouds find node field
NODE_STRENGTH_BASE = 63.0
NODE_DRIFT         = 0.0015

# Node drift speed per epoch:
# epoch 0-1 : fast outward (radiation pressure / ionization front)
# epoch 2   : slowing (recombination)
# epoch 3+  : nearly frozen (dark-matter halo anchoring)
# Drift scaled to new epoch timing (peaks ~1500 iterations)
# Epoch 0-1 (0-500 iter): fast drift for universe ignition
# Epoch 2-3 (500-1200):   slowing as matter dominates
# Epoch 4+  (1200+):      nearly frozen, anchored in halos
NODE_DRIFT_BY_EPOCH = {0: 0.08, 1: 0.04, 2: 0.008, 3: 0.003,
                       4: 0.001, 5: 0.001, 6: 0.0005, 7: 0.0005}

# How many steps a newly formed element survives before the burnout
# check can destroy it.  Prevents the "instant dissociation" loop.
ASSEMBLY_GRACE_STEPS = {"e": 8, "p": 15, "H": 40, "He": 80}

UNIVERSE_WORK_FEEDBACK = 0.02

BUBBLE_TYPES = {
    "H":     {"base_el": "H",  "node_bias": 1.0, "energy": 120, "node_rate": 0.002},
    "He":    {"base_el": "He", "node_bias": 1.8, "energy": 200, "node_rate": 0.006},
    "exotic":{"base_el": "C",  "node_bias": 0.6, "energy": 300, "node_rate": 0.01},
}

FRAMEWORK_RESERVOIR = 1_000_000
FRAMEWORK_DECAY     = 0.00001

ELEMENTS = ["nu", "e", "p", "n", "H", "He", "C", "O", "Fe", "Ni"]
# n = free neutron (unstable, beta-decays to H within a few steps)
# Ni = iron-group endpoint (binding energy maximum, no further fusion)

# ============================================================
# SPH CLOUD PARAMETERS
# ============================================================
# Each Cloud = macro-particle representing a gas cloud of N micro-particles.
# Internal physics: temperature T, SPH density rho, pressure P, sound speed cs.
#
# SPH_H      — kernel smoothing length (neighbourhood radius = 2*SPH_H)
# SPH_GAMMA  — adiabatic index (5/3 monatomic, 4/3 radiation-dominated)
# Viscosity α,β prevent cloud interpenetration (Monaghan 1992)

SPH_GAMMA      = 5.0 / 3.0   # monatomic ideal gas (matter era)
SPH_GAMMA_RAD  = 4.0 / 3.0   # radiation-dominated (epochs 0-1)
SPH_H          = 3.5          # smoothing length
SPH_ALPHA_VISC = 1.0          # artificial viscosity α
SPH_BETA_VISC  = 2.0          # artificial viscosity β
SPH_C_MIN      = 0.1          # minimum sound speed
CLOUD_COOL_RATE  = 0.0008     # base cooling rate per step
CLOUD_MERGE_FRAC = 0.5        # merge overlap threshold (fraction of sum-radii)
CLOUD_SPLIT_MASS = 5000       # Jeans mass: fragment if cloud exceeds this
CLOUD_MIN_MASS   = 5          # evaporation threshold


# -------- SPH cubic spline kernel (vectorized) --------

def _sph_kernel_w_vec(dist_arr, h):
    """W(r,h) for array of distances."""
    q     = dist_arr / h
    sigma = 1.0 / (np.pi * h**3)
    W = np.where(q < 1.0,  sigma * (1 - 1.5*q**2 + 0.75*q**3),
        np.where(q < 2.0,  sigma * 0.25*(2 - q)**3,
                           0.0))
    return W


def _sph_kernel_dw_vec(dist_arr, h):
    """dW/dq scalar for array of distances (for gradient)."""
    q     = dist_arr / h
    sigma = 1.0 / (np.pi * h**3)
    dWdq  = np.where(q < 1.0,  sigma * (-3*q + 2.25*q**2),
             np.where(q < 2.0,  sigma * (-0.75*(2 - q)**2),
                                0.0))
    return dWdq


# ============================================================
# COSMIC EPOCH SYSTEM
# ============================================================
COSMIC_EPOCHS = {
    0: {"name": "Big Bang / Quark-Gluon Plasma",  "emoji": "💥",
        "description": "Extreme energy, only quarks and leptons", "color": "#FF4500"},
    1: {"name": "Primordial Nucleosynthesis",      "emoji": "⚛️",
        "description": "Protons & neutrons fuse into H, He-4, Li", "color": "#FF8C00",
        "iteration_start": 200},
    2: {"name": "Recombination / Dark Ages",       "emoji": "🌑",
        "description": "Universe cools, neutral atoms form, photons decouple", "color": "#4B0082",
        "iteration_start": 500},
    3: {"name": "First Stars (Pop III)",           "emoji": "🌟",
        "description": "Massive, metal-free stars ignite — UV floods the universe", "color": "#FFD700",
        "iteration_start": 900},
    4: {"name": "Reionization",                    "emoji": "☀️",
        "description": "First stars re-ionize surrounding gas", "color": "#00BFFF",
        "iteration_start": 1200},
    5: {"name": "First Galaxies",                  "emoji": "🌌",
        "description": "Protogalaxies assemble around dark matter halos", "color": "#7B68EE",
        "iteration_start": 1500},
    6: {"name": "Cosmic Noon",                     "emoji": "🔥",
        "description": "Peak star formation rate, quasar activity", "color": "#FF6347",
        "iteration_start": 2000},
    7: {"name": "Modern Universe",                 "emoji": "🪐",
        "description": "Galaxies mature, dark energy dominates", "color": "#20B2AA",
        "iteration_start": 2500},
}

_framework_reservoir = FRAMEWORK_RESERVOIR
_framework_drain     = 0.0   # total energy permanently lost to framework
_current_epoch       = 0

# Minimum viable energy for a cloud to exist as matter.
# Below this, the cloud "falls through the framework" — provalivaetsya v karkas.
# This is the permanent sink: no resurrection, no revolution.
FRAMEWORK_FLOOR_T    = 0.05   # minimum temperature before provál
FRAMEWORK_FLOOR_MASS = 3.0    # minimum mass before provál

# ── Collapse event log ────────────────────────────────────────────────────
# Each entry: {"step": int, "type": str, "pos": array, "mass": float,
#              "peak_density": float, "galaxy_size": int}
# type: "galactic_bh"  — galaxy collapse → central supermassive BH
#        "direct_collapse" — mass lost without BH stage (below Chandrasekhar)
#        "agn_flare"       — existing BH accretes enough to flare
COLLAPSE_LOG = []

def log_collapse(step, ctype, pos, mass, peak_density, galaxy_size=0):
    COLLAPSE_LOG.append({
        "step": step, "type": ctype,
        "pos": pos.copy(), "mass": float(mass),
        "peak_density": float(peak_density),
        "galaxy_size": int(galaxy_size),
    })


def get_current_epoch(iteration):
    epoch = 0
    for eid, edata in COSMIC_EPOCHS.items():
        if eid == 0:
            continue
        if iteration >= edata.get("iteration_start", 9999):
            epoch = eid
    return epoch


def get_epoch_info(iteration):
    return COSMIC_EPOCHS[get_current_epoch(iteration)]


# ============================================================
# SPATIAL HASH
# ============================================================
class SpatialHash:
    def __init__(self, cell_size=4.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def _key(self, pos):
        cs = self.cell_size
        return (int(pos[0]//cs), int(pos[1]//cs), int(pos[2]//cs))

    def build(self, bodies):
        self.grid.clear()
        for b in bodies:
            self.grid[self._key(b.pos)].append(b)

    def query_radius(self, pos, radius):
        r_cells = int(radius / self.cell_size) + 1
        cx, cy, cz = self._key(pos)
        candidates = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for dz in range(-r_cells, r_cells + 1):
                    candidates.extend(self.grid.get((cx+dx, cy+dy, cz+dz), []))
        if not candidates:
            return []
        cpos  = np.array([b.pos for b in candidates])
        diff  = cpos - np.array(pos)
        dsq   = (diff*diff).sum(axis=1)
        mask  = dsq <= radius*radius
        return [b for b, m in zip(candidates, mask) if m]


spatial_hash = SpatialHash(cell_size=4.0)


# ============================================================
# CLOUD  —  SPH macro-particle
# ============================================================
class Cloud:
    """
    A Cloud is one SPH macro-particle representing a gas/plasma cloud.

    Physical state:
      pos    — centre of mass  [3-vector]
      vel    — bulk drift velocity
      mass   — total enclosed mass
      T      — temperature  (internal energy per unit mass proxy)
      rho    — SPH-estimated mass density (recomputed each step)
      P      — thermal pressure  = (γ-1)·ρ·T
      cs     — sound speed       = sqrt(γ·P/ρ)
      el     — dominant element  (chemistry label)
      kind   — "cloud" | "star" | "whitehole" | "bh"
      N      — number of micro-particles represented
    """

    __slots__ = [
        "pos","vel","mass","radius","kind","el","N",
        "T","rho","P","cs",
        "work","state","coherence","cluster_id",
        "_sph_ready","_el_age",
    ]

    def __init__(self, pos, mass, radius, kind="cloud", el="nu",
                 temperature=None, N=1):
        self.pos    = np.array(pos, float)
        self.vel    = np.random.randn(3) * 0.01
        self.mass   = float(mass)
        self.radius = float(radius)
        self.kind   = kind
        self.el     = el
        self.N      = int(N)

        self.T      = float(temperature) if temperature is not None else FEED_ENERGY
        self.rho    = mass / max(4/3*np.pi*radius**3, 1e-6)
        self.P      = max(0.0, (SPH_GAMMA-1)*self.rho*self.T)
        self.cs     = max(SPH_C_MIN, np.sqrt(abs(SPH_GAMMA*self.P/max(self.rho,1e-9))))

        self.work       = 0.0
        self.state      = "material"
        self.coherence  = np.random.rand()
        self.cluster_id = None
        self._sph_ready = False
        # grace counter per element: prevents instant burnout-loop
        self._el_age    = 0   # steps since last element change

    # --- legacy energy alias ---
    @property
    def energy(self):
        return self.T
    @energy.setter
    def energy(self, v):
        self.T = float(v)

    # --- thermodynamics ---

    def update_thermo(self, epoch=0):
        gamma = SPH_GAMMA_RAD if epoch <= 1 else SPH_GAMMA
        self.P  = max(0.0, (gamma-1)*self.rho*self.T)
        self.cs = max(SPH_C_MIN,
                      np.sqrt(abs(gamma*self.P/max(self.rho, 1e-9))))

    def cool(self, dt, epoch=0):
        """Hubble + radiative cooling."""
        hub = (2 if epoch <= 1 else 1) * H * self.T
        self.T = max(0.01, self.T - (CLOUD_COOL_RATE + hub)*dt*self.T)
        self.update_thermo(epoch)

    # --- SPH density (vectorized, called from sph_density_pass) ---

    def set_density_from_neighbors(self, neighbor_pos, neighbor_mass):
        """Called by the global SPH pass with pre-built arrays."""
        dist = np.linalg.norm(neighbor_pos - self.pos, axis=1)
        W    = _sph_kernel_w_vec(dist, SPH_H)
        self.rho      = max(float(np.dot(neighbor_mass, W)), 1e-6)
        self._sph_ready = True

    # --- forces ---

    def _pressure_visc_force(self, neighbors):
        """
        Combined SPH pressure gradient + Monaghan artificial viscosity.
        F_i = -m_i Σ_j m_j [ (P_i/ρ_i² + P_j/ρ_j²) + Π_ij ] ∇W_ij
        """
        if not neighbors:
            return np.zeros(3)

        nb_pos   = np.array([n.pos  for n in neighbors])
        nb_mass  = np.array([n.mass for n in neighbors])
        nb_P     = np.array([n.P    for n in neighbors])
        nb_rho   = np.array([n.rho  for n in neighbors])
        nb_vel   = np.array([n.vel  for n in neighbors])
        nb_cs    = np.array([n.cs   for n in neighbors])

        rvec   = self.pos - nb_pos          # (N,3)  r_i - r_j
        dist   = np.linalg.norm(rvec, axis=1)  # (N,)
        safe_d = np.maximum(dist, 1e-9)

        # Kernel gradient direction: ∇W = (dW/dq)(1/h) * r̂
        dWdq   = _sph_kernel_dw_vec(dist, SPH_H)
        rhat   = rvec / safe_d[:, None]
        gradW  = (dWdq / SPH_H)[:, None] * rhat   # (N,3)

        # Pressure term
        pi_i   = self.P / max(self.rho**2, 1e-12)
        pj     = nb_P / np.maximum(nb_rho**2, 1e-12)
        pterm  = (pi_i + pj)                        # (N,)

        # Artificial viscosity
        vrel   = self.vel - nb_vel                  # (N,3)
        rv     = (vrel * rvec).sum(axis=1)          # (N,)
        approaching = rv < 0
        mu     = np.where(approaching,
                          SPH_H * rv / (dist**2 + 0.01*SPH_H**2),
                          0.0)
        rho_avg = 0.5*(self.rho + nb_rho)
        cs_avg  = 0.5*(self.cs  + nb_cs)
        pi_visc = np.where(approaching,
                           (-SPH_ALPHA_VISC*cs_avg*mu + SPH_BETA_VISC*mu**2) / rho_avg,
                           0.0)

        coeff  = nb_mass * (pterm + pi_visc)        # (N,)
        F      = -self.mass * (coeff[:, None] * gradW).sum(axis=0)
        return F

    def _gravity_force(self, neighbors):
        """Long-range gravity (SPH neighborhood)."""
        if not neighbors:
            return np.zeros(3)
        nb_pos  = np.array([n.pos  for n in neighbors])
        nb_mass = np.array([n.mass for n in neighbors])
        rvec    = nb_pos - self.pos
        dist    = np.linalg.norm(rvec, axis=1) + 1e-6
        F       = G * self.mass * ((nb_mass / dist**3)[:, None] * rvec).sum(axis=0)
        return F

    # --- master update ---

    def update(self, others, dt, nodes=None, use_spatial=True, epoch=0):
        # 1. Hubble flow
        self.vel += self.pos * H * dt

        # 2. Neighbours
        nb = (spatial_hash.query_radius(self.pos, SPH_H*2)
              if (use_spatial and spatial_hash.grid) else others)
        nb = [n for n in nb if n is not self]

        # 3. SPH density (if global pass skipped this cloud)
        if not self._sph_ready and nb:
            nb_pos  = np.array([n.pos  for n in nb])
            nb_mass = np.array([n.mass for n in nb])
            self.set_density_from_neighbors(nb_pos, nb_mass)

        # 4. Forces
        F = self._pressure_visc_force(nb) + self._gravity_force(nb)
        if nodes:
            F += work_field_fast(self, nodes)

        a = F / max(self.mass, 1e-6)
        a_mag = np.linalg.norm(a)
        if a_mag > 5.0:
            a = a * 5.0 / a_mag
        self.vel += a * dt
        self.vel *= 0.985
        self.pos += self.vel * dt
        self.work += np.linalg.norm(a) * dt

        # 5. Viscous heating + cooling
        visc_heat = np.linalg.norm(F) * 0.0001 * dt
        self.T    = max(0.01, self.T + visc_heat)
        self.cool(dt, epoch)

        # 6. Coherence
        self.coherence = min(self.coherence + 0.001*dt, 1.0)
        if self.coherence > C_CRIT:
            self.state = "frozen"

        self._sph_ready = False

        # 7. Star ageing
        if self.kind == "star":
            self.mass += 0.02
            self.T     = min(self.T * 1.001, 1e5)
            if self.mass > 600:
                self._supernova()

    def _supernova(self):
        self.kind   = "cloud"
        self.mass  *= 0.25
        self.radius *= 0.4
        self.el     = np.random.choice(ELEMENTS[3:])
        self.T      = TH_STAB * 10
        self.N      = max(self.N // 4, 1)


# Keep 'Body' alias so old code (WhiteHole/BlackHole) still compiles
Body = Cloud


# ============================================================
# GLOBAL SPH DENSITY PASS  (call once per step, before update)
# ============================================================
def sph_density_pass(world):
    """
    Vectorized density estimation for all Cloud bodies.
    Sets c.rho, c.P, c.cs for every cloud in O(N·k) where k=avg neighbors.
    """
    global _current_epoch
    clouds   = [b for b in world
                if isinstance(b, Cloud) and b.kind not in ("whitehole","bh")]
    if not clouds:
        return

    pos_arr  = np.array([c.pos  for c in clouds])
    mass_arr = np.array([c.mass for c in clouds])

    for i, ci in enumerate(clouds):
        diff  = pos_arr - ci.pos
        dist  = np.linalg.norm(diff, axis=1)
        W     = _sph_kernel_w_vec(dist, SPH_H)
        ci.rho        = max(float(np.dot(mass_arr, W)), 1e-6)
        ci._sph_ready = True
        ci.update_thermo(_current_epoch)


# ============================================================
# CLOUD MERGING & SPLITTING  (Jeans physics)
# ============================================================
def merge_clouds(world):
    """
    Merge overlapping diffuse clouds.  Conserves mass, momentum, thermal energy.
    Stars and BHs are never consumed.
    """
    rem = set()
    for ci in world:
        if id(ci) in rem or ci.kind in ("bh","whitehole"):
            continue
        if not isinstance(ci, Cloud):
            continue
        nbs = spatial_hash.query_radius(ci.pos, (ci.radius + 2.0)*1.5)
        for cj in nbs:
            if cj is ci or id(cj) in rem:
                continue
            if not isinstance(cj, Cloud) or cj.kind in ("bh","whitehole","star"):
                continue
            # Don't merge immunity-window seeds
            if getattr(ci,'_el_age',-1) < 0 or getattr(cj,'_el_age',-1) < 0:
                continue
            d = np.linalg.norm(ci.pos - cj.pos)
            if d >= CLOUD_MERGE_FRAC*(ci.radius + cj.radius):
                continue
            mt = ci.mass + cj.mass
            ci.vel = (ci.mass*ci.vel + cj.mass*cj.vel) / mt
            ci.T   = (ci.mass*ci.T  + cj.mass*cj.T)  / mt
            ci.pos = (ci.mass*ci.pos + cj.mass*cj.pos) / mt
            ci.mass = mt
            ci.N   += cj.N
            ci.radius = (ci.radius**3 + cj.radius**3)**(1/3)
            ep = {k: i for i, k in enumerate(ELEMENTS)}
            if ep.get(cj.el,0) > ep.get(ci.el,0):
                ci.el = cj.el
            rem.add(id(cj))
    world[:] = [b for b in world if id(b) not in rem]


def split_cloud(world, epoch=0):
    """
    Fragment clouds that exceed the Jeans mass.
    Two daughters get half the mass, conserving momentum + slight cooling.
    """
    new_clouds = []
    for c in world:
        if not isinstance(c, Cloud) or c.kind in ("bh","whitehole","star"):
            continue
        if c.mass < CLOUD_SPLIT_MASS:
            continue
        d = np.random.randn(3); d /= np.linalg.norm(d) + 1e-9
        off = d * c.radius * 0.4
        m2  = c.mass * 0.5
        for sign in (+1, -1):
            nc = Cloud(c.pos + sign*off, m2, c.radius*0.63,
                       kind=c.kind, el=c.el,
                       temperature=c.T*0.9, N=max(c.N//2,1))
            nc.vel = c.vel + sign*d*0.1
            new_clouds.append(nc)
        c.mass = -1
    world[:] = [b for b in world if b.mass > 0]
    world.extend(new_clouds)


def evaporate_small_clouds(world):
    world[:] = [b for b in world
                if not (isinstance(b, Cloud) and b.kind == "cloud"
                        and b.mass < CLOUD_MIN_MASS)]


def framework_collapse(world, step=0):
    """
    Dip in carcass — permanent energy sink.

    A cloud that has exhausted both its mass and thermal energy falls
    through the framework floor. Two outcomes:

    1. Wandering micro-BH (high |vel|, low T, low mass):
       Brief Hawking-like flash — tiny BH that radiates and vanishes
       instantly. No singularity, no accretion. Just a luminosity spike
       and permanent removal.

    2. Cold dead cloud (low |vel|, low T):
       Silent removal. Energy drains to framework permanently.

    In both cases: no resurrection, no revolution, no recycling.
    Energy conservation: drained energy added to _framework_drain.
    """
    global _framework_drain
    to_remove = set()

    for b in world:
        if not isinstance(b, Cloud): continue
        if b.kind in ("whitehole", "bh", "star"): continue

        # Check floor conditions: both T and mass must be depleted
        t_depleted    = b.T    < FRAMEWORK_FLOOR_T
        mass_depleted = b.mass < FRAMEWORK_FLOOR_MASS * 2

        if not (t_depleted and mass_depleted): continue

        speed = np.linalg.norm(b.vel)

        if speed > 0.5 and b.mass > FRAMEWORK_FLOOR_MASS:
            # Wandering micro-BH: high velocity + minimal energy
            # Brief flash (logged as collapse event), then permanent removal
            log_collapse(step, "framework_drain_bh", b.pos,
                         b.mass, b.T * b.mass, galaxy_size=0)

        # Permanent drain — energy exits the system
        _framework_drain += b.T * b.mass
        to_remove.add(id(b))

    if to_remove:
        world[:] = [b for b in world if id(b) not in to_remove]


# ============================================================
# WHITE HOLE
# ============================================================
class WhiteHole(Cloud):
    def __init__(self, pos):
        super().__init__(pos, 18000, 0.7, "whitehole", "nu",
                         temperature=FEED_ENERGY*10)

    def process(self, world):
        for b in world[:]:
            if b is self:
                continue
            d = np.linalg.norm(b.pos - self.pos)
            if d < WHITE_RADIUS:
                r = b.pos - self.pos
                r /= np.linalg.norm(r) + 1e-6
                b.T   += 5
                b.vel += r * 0.4
                if b.kind == "bh":
                    self._explode_bh(b, world)
                else:
                    self._revolution(b)

    def _explode_bh(self, bh, world):
        """White hole processes BH: conserved mass split into N fragments."""
        n_fragments = 20
        frag_mass   = max(bh.mass / n_fragments, 5.0)
        frag_T      = max(bh.T   / n_fragments, 1.0)   # energy conserved
        for _ in range(n_fragments):
            p = Cloud(bh.pos + np.random.randn(3)*0.4, frag_mass, 0.15,
                      "cloud", "nu", temperature=frag_T, N=3)
            p.vel = np.random.randn(3) * min(0.6, np.sqrt(frag_T * 0.01))
            world.append(p)
        if bh in world:
            world.remove(bh)

    def _revolution(self, b):
        """Recycle cloud back to nu — conserve mass, shed most energy to framework."""
        global _framework_drain
        # Energy shed: white hole absorbs most of the thermal energy
        # Only a small fraction returns as kinetic heat of the nu cloud
        energy_shed = b.T * b.mass * 0.85
        _framework_drain += energy_shed
        b.el     = "nu"
        b.kind   = "cloud"
        b.T      = max(b.T * 0.15, FEED_ENERGY * 0.1)   # retain 15% as heat
        b.radius = max(b.radius * 0.5, 0.2)
        # mass conserved — white hole doesn't create or destroy mass

    # legacy method names
    def explode_bh(self, bh, world): self._explode_bh(bh, world)
    def revolution(self, b):         self._revolution(b)


# ============================================================
# ASSEMBLY  —  epoch-aware chemistry on clouds
# ============================================================
def assemble(b, nodes=None, epoch=0):
    """
    Cloud chemistry driven by temperature T and SPH density rho.

    Key fixes vs original:
    - 'in_field' only resets to nu if cloud has NOT been in a field for
      ASSEMBLY_GRACE_STEPS — prevents instant dissociation on field exit.
    - Burnout thresholds are higher and gated by grace period: a freshly
      formed electron won't vanish in the same step it appeared.
    - non-nu clouds outside node radius drift until grace expires, not
      instantly destroyed — models finite mean-free-path of radiation.
    """
    b._el_age += 1

    # ── Free neutron beta-decay — unconditional, node-independent ─────
    # n → p + e⁻ + ν̄ₑ → H (or p if cold). Half-life ~15min real.
    # Must run before any node-field checks.
    if b.el == "n":
        if np.random.rand() < 0.15:
            b.el = "H" if b.T > TH_NU_E else "p"
            b._el_age = 0
        return   # neutrons skip all other assembly logic


    if not nodes:
        # decay back to nu only if grace period is over
        if b.el != "nu" and b._el_age > ASSEMBLY_GRACE_STEPS.get(b.el, 5):
            b.el = "nu"; b._el_age = 0
        return

    # Find nearest node and its strength
    best_dist = 1e9; best_strength = 0.0
    for n in nodes:
        if not n.is_alive(): continue
        d = np.linalg.norm(b.pos - n.pos)
        if d < best_dist:
            best_dist = d; best_strength = n.strength * n.life

    # If cloud is inside a bubble with internal nodes, count those too
    # This ensures chemistry works inside the bubble even if external nodes are far
    from_bubble_node = (best_dist > NODE_RADIUS * 3 and best_strength > 0)

    # NODE_RADIUS for nu clouds (must be attracted to nodes to assemble)
    # but already-assembled clouds have a WIDER tolerance — they can
    # survive further from nodes (Debye shielding / molecular binding)
    el_tolerance = {"nu": 1.0, "e": 1.8, "p": 2.5, "H": 4.0,
                    "He": 5.0, "C": 6.0, "O": 6.0, "Fe": 6.0}
    effective_radius = NODE_RADIUS * el_tolerance.get(b.el, 1.0)

    in_field = best_dist < effective_radius

    if not in_field:
        # Outside field: only light elements (e, p) decay back to nu
        # Heavy elements (H+) are gravitationally/chemically stable
        grace = ASSEMBLY_GRACE_STEPS.get(b.el, 5)
        if b.el != "nu" and b._el_age > grace:
            if b.el not in ("H","He","C","O","Fe","Ni"):
                b.el = "nu"; b._el_age = 0
        return

    T   = b.T
    rho = max(b.rho, 0.05)  # floor density so newly formed clouds aren't rate=0

    # ── Burnout: dissociation only in epoch >= 1 and after grace period ──────
    # In epoch 0 (QGP/Big Bang): everything is plasma, no burnout.
    # Even in later epochs, H/He/heavy elements are stable — only e/p can
    # dissociate back, and only when the cloud has cooled below a threshold.
    grace = ASSEMBLY_GRACE_STEPS.get(b.el, 5)
    if epoch >= 1 and b._el_age > grace and b._el_age >= 0:
        # Electron dissociates only if T drops below formation threshold
        # (i.e. no longer enough energy to maintain e state) — NOT if too hot
        # Hot plasma just means more energetic electrons, not their destruction
        if b.el == "e" and T < TH_NU_E * 0.5:    # too cold to stay as e
            b.el = "nu"; b._el_age = 0; return
        if b.el == "p" and T < TH_E_P * 0.5:      # too cold to stay as p
            b.el = "e"; b._el_age = 0; return
    # H can ionize at extreme temperatures (stellar interior)
    if epoch >= 3 and b.el == "H" and T > TH_STAB * 20 and b._el_age > grace:
        b.el = "p"; b.T *= 0.6; b._el_age = 0; return

    # ── Progressive assembly ───────────────────────────────────────────────
    # Threshold scaled by node strength: stronger nodes → lower effective
    # barrier (like higher local photon / work density catalysing reactions)
    strength_factor = min(best_strength / (NODE_STRENGTH_BASE * 3.5), 1.5)
    eff_nu_e = TH_NU_E / strength_factor
    eff_e_p  = TH_E_P  / strength_factor
    eff_p_h  = TH_PN   / strength_factor

    prev_el = b.el
    # Sequential assembly — each 'if' can trigger in same call if T crosses
    # multiple thresholds. This means nu→H can happen in one hot step.
    if b.el == "nu" and T > eff_nu_e:  b.el = "e"
    if b.el == "e"  and T > eff_e_p:   b.el = "p"
    if b.el == "p"  and T > eff_p_h:   b.el = "H"
    if b.el != prev_el:
        b._el_age = 0

    # Epoch 1: BBN nucleosynthesis — density-dependent H→He
    # Three-body rate ∝ ρ²  →  more likely in dense clouds
    if epoch >= 1 and b.el == "H" and T > TH_STAB * 0.8:
        rate = 0.04 * min(rho / 100.0, 3.0)
        if np.random.rand() < rate:
            b.el = "He"; b.T *= 0.8

    # Free neutron beta-decay → always becomes H (or p if very cold)
    # Rate: ~1/15min real, here scaled to a few steps
    if b.el == "n":
        if np.random.rand() < 0.15:   # fast decay
            b.el = "H" if T > TH_NU_E else "p"
            b._el_age = 0
        return   # neutrons don't follow normal assembly

    # Epoch 2: recombination — p + e⁻ → H  (density-enhanced)
    if epoch >= 2 and b.el == "p" and T < TH_E_P * 1.5:
        if np.random.rand() < 0.005 * (1 + rho / 50.0):
            b.el = "H"

    # Epoch 3+: stellar nucleosynthesis
    if epoch >= 3 and b.kind == "star":
        if   b.el == "He" and np.random.rand() < 0.01:  b.el = "C";  b.T *= 1.05
        elif b.el == "C"  and np.random.rand() < 0.005: b.el = "O";  b.T *= 1.02
        elif b.el == "O"  and np.random.rand() < 0.002: b.el = "Fe"


# ============================================================
# FUSION & COLLISION
# ============================================================
def fuse(c1, c2):
    """
    Physical nuclear fusion ladder.
    Skip 'n' (neutron) as a stable intermediate — neutrons are unstable
    free particles (half-life 15min), immediately beta-decay to proton.
    Ni-62 is the binding energy peak — endpoint, no further exothermic fusion.
    Ni accumulation beyond mass cap → direct collapse (framework drain path).
    """
    order = {k: i for i, k in enumerate(ELEMENTS)}

    # Only fuse if c1 is lighter element
    if order.get(c1.el, 0) >= order.get(c2.el, 0):
        c1.T *= 1.05   # elastic collision heating only
        return

    # Physical fusion ladder — skip neutron as stable product
    FUSION_NEXT = {
        "nu": "e",    # QGP
        "e":  "p",    # pair production
        "p":  "H",    # p+p → deuterium → H  (skip free neutron)
        "n":  "H",    # free neutron beta-decays to H immediately
        "H":  "He",   # stellar: H burning
        "He": "C",    # stellar: helium flash → C
        "C":  "O",    # carbon burning
        "O":  "Fe",   # oxygen/silicon burning → Fe-group
        "Fe": "Ni",   # Fe + alpha → Ni (only in extreme conditions)
        "Ni": None,   # endpoint — no exothermic fusion beyond Ni-62
    }

    next_el = FUSION_NEXT.get(c1.el)

    if next_el is None:
        # Ni endpoint: mass accumulation → instability → framework drain path
        # Mark cloud as overloaded — will be caught by framework_collapse
        c1.T   *= 0.3    # Ni fusion is endothermic — COOLS the cloud
        c1.mass = min(c1.mass, CLOUD_SPLIT_MASS * 0.9)  # cap mass
    else:
        c1.el = next_el
        c1.T *= 1.1      # exothermic fusion heating


def collide(c1, c2):
    d = np.linalg.norm(c1.pos - c2.pos)
    if d < (c1.radius + c2.radius) * 0.6:
        fuse(c1, c2)
        mt    = c1.mass + c2.mass
        c1.T  = (c1.T*c1.mass + c2.T*c2.mass) / max(mt,1e-6)
        c1.vel= (c1.mass*c1.vel + c2.mass*c2.vel) / max(mt,1e-6)
        c1.mass = mt; c1.N += c2.N
        return True
    return False


def collide_all_fast(world):
    rem = set()
    for b in world:
        if id(b) in rem or b.kind in ("bh","whitehole"):
            continue
        nbs = spatial_hash.query_radius(b.pos, b.radius*2+0.5)
        for o in nbs:
            if o is b or id(o) in rem or o.kind in ("bh","whitehole"):
                continue
            if collide(b, o):
                rem.add(id(o))
    return [b for b in world if id(b) not in rem]


# ============================================================
# FRAMEWORK FEED
# ============================================================
def framework_feed(world):
    global _framework_reservoir
    if not FRAMEWORK_FEED: return
    # Hard floor: reservoir cannot feed below zero
    if _framework_reservoir <= 0:
        _framework_reservoir = 0.0
        return
    _framework_reservoir *= (1 - FRAMEWORK_DECAY)
    cost = FEED_ENERGY * 10 * FEED_PARTICLES_PER_TICK
    if np.random.rand() < FEED_RATE and _framework_reservoir >= cost:
        for _ in range(FEED_PARTICLES_PER_TICK):
            pos = np.random.randn(3) * (PHASE_WALL_R * 0.8)
            c   = Cloud(pos, 50, 0.5, "cloud", "nu",
                        temperature=FEED_ENERGY, N=10)
            c.rho = 0.01
            world.append(c)
            _framework_reservoir -= FEED_ENERGY * 10


def total_material_energy(world):
    return sum(b.T * b.mass for b in world if b.kind != "whitehole")


def get_framework_reservoir():
    return _framework_reservoir

def get_framework_drain():
    """Total energy permanently lost to framework (dip sink)."""""
    return _framework_drain


# ============================================================
# SUPERNOVA (standalone)
# ============================================================
def supernova(b, world):
    """Stellar explosion: conserve total mass and energy."""""
    if b.mass > 800:
        n_ejecta  = 20
        ejecta_mass = b.mass * 0.7 / n_ejecta   # 70% ejected
        ejecta_T    = b.T * 2.0 / n_ejecta       # energy split (not multiplied)
        for _ in range(n_ejecta):
            p = Cloud(b.pos + np.random.randn(3)*0.3, ejecta_mass, 0.3,
                      "cloud", "H", temperature=ejecta_T, N=3)
            p.vel = np.random.randn(3)*0.4
            world.append(p)
        b.mass *= 0.3; b.el = "Fe"; b.T *= 0.3   # remnant cools too


# ============================================================
# BLACK HOLE
# ============================================================
class BlackHole(Cloud):
    def __init__(self, pos):
        super().__init__(pos, 50000, 1.0, "bh", "bh", temperature=0)
        self.inner_work = 0
        self.outer_work = 0
        self.resource   = 0

    def accrete(self, b):
        self.mass       += b.mass
        self.outer_work += b.work
        self.T          += b.T * b.mass / max(self.mass, 1)

    def update_bh(self, world):
        self.inner_work += self.mass * COLLAPSE_SPEED
        if self.inner_work - self.outer_work > BH_COLLAPSE_W:
            self.resource += self.mass * 0.4
            self.mass     *= 0.7
        if self.T > BH_EXPLODE_E:
            explode_bh(self, world)


def spawn_black_holes(world, white):
    for b in world:
        if np.linalg.norm(b.pos - white.pos) < WHITE_RADIUS * 1.5:
            continue
        rho  = local_work_density_fast(b)
        grad = np.linalg.norm(b.vel)
        if rho > CRITICAL_WORK_DENSITY and grad > WORK_GRADIENT_MIN:
            world.append(BlackHole(b.pos.copy()))
            return


def merge_black_holes(world):
    bhs = [b for b in world if isinstance(b, BlackHole)]
    for i in range(len(bhs)):
        for j in range(i+1, len(bhs)):
            if np.linalg.norm(bhs[i].pos - bhs[j].pos) < BH_MERGE_DIST:
                bhs[i].mass += bhs[j].mass
                bhs[i].T    += bhs[j].T
                if bhs[j] in world: world.remove(bhs[j])
                return


def explode_bh(bh, world):
    """BH self-explosion (overheated): conserved mass/energy fragments."""""
    n_fragments = 30
    frag_mass   = max(bh.mass / n_fragments, 5.0)
    frag_T      = max(bh.T   / n_fragments, 1.0)
    for _ in range(n_fragments):
        p = Cloud(bh.pos + np.random.randn(3)*0.5, frag_mass, 0.3,
                  "cloud", "nu", temperature=frag_T, N=3)
        p.vel = np.random.randn(3) * min(1.0, np.sqrt(frag_T * 0.01))
        world.append(p)
    if bh in world: world.remove(bh)


# ============================================================
# WORK NODE
# ============================================================
class WorkNode:
    def __init__(self, pos, strength, node_type="primary", origin="void"):
        self.pos       = np.array(pos, float)
        self.strength  = float(strength)
        self.node_type = node_type
        self.origin    = origin
        self.life      = 1.0
        self.age       = 0
        self.work_accumulated = 0.0
        self.energy    = 0.0
        self._epoch    = 0          # updated externally each step
        # Strong initial outward burst — radiation-pressure ionization front
        outward = np.array(pos, float) / (np.linalg.norm(pos) + 1e-9)
        speed   = NODE_DRIFT_BY_EPOCH[0] * 12.0
        self.drift_vel = outward * speed + np.random.randn(3) * speed * 0.4

    def update(self, local_work_density=0, dt=0.016):
        self.age += 1
        self.work_accumulated += local_work_density * dt
        decay = {"primary":0.00001,"secondary":0.00003,
                 "tertiary":0.0001, "exotic":0.0005}.get(self.node_type, 0.00003)
        self.life -= decay
        self.life  = max(self.life, 0.1 if self.node_type=="primary" else 0.05)
        we = min(self.work_accumulated*0.00001, 0.5)
        self.strength *= (1 + we*0.01)
        drift_speed = NODE_DRIFT_BY_EPOCH.get(self._epoch, NODE_DRIFT)
        # Damping: free-streaming in epoch 0-1, strong drag later
        damp = 0.99 if self._epoch <= 1 else 0.92
        self.drift_vel *= damp
        self.drift_vel += np.random.randn(3) * drift_speed * 0.3
        # clamp only in later epochs
        if self._epoch >= 2:
            spd = np.linalg.norm(self.drift_vel)
            if spd > drift_speed * 2:
                self.drift_vel *= drift_speed * 2 / spd
        self.pos += self.drift_vel * dt

    def is_alive(self):
        return self.life > 0.01


# ============================================================
# WORK FIELD  —  vectorized
# ============================================================
def work_field_fast(b, nodes):
    if not nodes: return np.zeros(3)
    mods  = {"primary":1.0,"secondary":0.7,"tertiary":0.4,"exotic":1.5}
    alive = [n for n in nodes if n.is_alive()]
    if not alive: return np.zeros(3)
    npos  = np.array([n.pos for n in alive])
    rv    = npos - b.pos
    dsq   = (rv*rv).sum(axis=1)
    mask  = dsq <= NODE_RADIUS**2
    if not mask.any(): return np.zeros(3)
    rv    = rv[mask]; dsq = dsq[mask]+1e-12
    am    = [n for n,m in zip(alive,mask) if m]
    s     = np.array([n.strength*n.life*mods.get(n.node_type,1.0) for n in am])
    return (s[:,None]*rv/dsq[:,None]).sum(axis=0)


def work_field(b, nodes):
    return work_field_fast(b, nodes)


# ============================================================
# LOCAL WORK DENSITY
# ============================================================
def local_work_density_fast(b, R=2.0):
    nbs = spatial_hash.query_radius(b.pos, R)
    V   = 4/3*np.pi*R**3
    return sum(o.work for o in nbs) / V if V > 0 else 0


def local_work_density(b, world, R=2.0):
    return local_work_density_fast(b, R)


# ============================================================
# UNIVERSE BUBBLE
# ============================================================
class UniverseBubble:
    def __init__(self, center, btype="H"):
        self.center          = np.array(center, float)
        self.radius          = 3.0
        self.energy          = 100.0
        self.type            = btype
        self.age             = 0
        self.dead            = False
        self.has_first_stars   = False
        self.galaxy_mass       = 0.0
        self.star_count        = 0
        self.peak_work_density = 0.0   # max work density ever seen inside
        self.bh_id             = None  # id of galactic central BH (if any)
        self.collapse_count    = 0     # how many collapse events happened
        self.stability         = 1.0   # 1=stable, 0=dissolving


def find_node_clusters(nodes, cluster_radius=3.0):
    if not nodes: return []
    visited  = set()
    clusters = []
    for i, n1 in enumerate(nodes):
        if i in visited or not n1.is_alive(): continue
        members = [n1]; visited.add(i)
        for j, n2 in enumerate(nodes):
            if j in visited or not n2.is_alive(): continue
            for m in members:
                if np.linalg.norm(n2.pos-m.pos) < cluster_radius:
                    members.append(n2); visited.add(j); break
        if len(members) >= 2:
            center = np.mean([n.pos for n in members], axis=0)
            ts     = sum(n.strength*n.life for n in members)
            clusters.append((center, len(members), ts))
    return clusters


MAX_BUBBLES = 12   # maximum simultaneous universe bubbles
_last_bubble_step = -99  # cooldown tracker

def spawn_multiverse(bubbles, world, nodes, step=0):
    """
    Bubble formation via NODE FLYTHROUGH model.
    A fast-moving node sweeping through a cloud cluster deposits work energy.
    When node_strength * speed * nearby_clouds exceeds threshold -> bubble ignites.
    This matches the observed: node flies into nu-cloud cluster, flashes, exits.
    """
    global _last_bubble_step
    if not FRAMEWORK_FEED or not nodes: return
    # Hard cap + cooldown: max 1 bubble per 8 steps, max MAX_BUBBLES total
    active = [b for b in bubbles if not b.dead]
    if len(active) >= MAX_BUBBLES: return
    if step - _last_bubble_step < 8: return
    clouds = [b for b in world if b.kind == "cloud"
              and not isinstance(b, (WhiteHole, BlackHole))]
    if len(clouds) < 2: return

    for n in nodes:
        if not n.is_alive(): continue
        speed = np.linalg.norm(n.drift_vel)
        if speed < 0.05: continue   # stationary nodes don't trigger

        # Nearby clouds within SPH kernel
        nearby = [c for c in clouds if np.linalg.norm(c.pos - n.pos) < SPH_H * 1.5]
        if len(nearby) < 2: continue

        # Work = strength × speed × cloud_count (flythrough energy deposit)
        work = n.strength * n.life * speed * len(nearby)

        # Ignition threshold scales with cost constant
        if work < BUBBLE_FORMATION_COST: continue

        # Consume clouds (max 15% of total)
        max_burn = max(2, int(len(clouds) * 0.15))
        consumed = []
        for c in sorted(nearby, key=lambda x: x.mass):
            if len(consumed) >= max_burn: break
            consumed.append(c)
        for c in consumed:
            if c in world: world.remove(c)

        # Bubble type: exotic if node is exotic/primary with high work
        if n.node_type == "exotic" or work > BUBBLE_FORMATION_COST * 5:
            btype = "exotic"
        elif work > BUBBLE_FORMATION_COST * 2:
            btype = "He"
        else:
            btype = "H"

        em  = {"exotic": 2.0, "He": 1.5, "H": 1.0}[btype]
        bub = UniverseBubble(n.pos.copy(), btype)
        bub.energy = BUBBLE_TYPES[btype]["energy"] * em + work * 0.005
        bub.radius = max(2.5, SPH_H)
        bubbles.append(bub)
        _last_bubble_step = step

        # Seed bubble: hot nu clouds drawn from bubble's own energy budget
        # Energy is NOT free — it's debited from bub.energy (locked-in ignition)
        n_seeds     = 3
        seed_mass   = 80
        seed_T      = min(FEED_ENERGY * 3, bub.energy / (n_seeds * 2))
        seed_cost   = seed_mass * seed_T * n_seeds
        if bub.energy >= seed_cost:
            bub.energy -= seed_cost
            for _ in range(n_seeds):
                seed_pos = n.pos + np.random.randn(3) * bub.radius * 0.3
                seed = Cloud(seed_pos, mass=seed_mass, radius=0.6,
                             kind="cloud", el="nu",
                             temperature=seed_T, N=20)
                seed._el_age = -20
                world.append(seed)

        # Place 4 internal nodes spread across bubble radius
        for i in range(4):
            angle = i * np.pi / 2
            npos = n.pos + np.array([
                np.cos(angle) * bub.radius * 0.4,
                np.sin(angle) * bub.radius * 0.4,
                np.random.randn() * bub.radius * 0.2
            ])
            internal = WorkNode(npos, NODE_STRENGTH_BASE * 2.0,
                                node_type="secondary", origin="bubble_seed")
            internal._epoch = _current_epoch
            # Slow inward drift — sweep through bubble contents
            internal.drift_vel = (n.pos - npos) / (np.linalg.norm(n.pos - npos) + 1e-9) * 0.05
            nodes.append(internal)

        # Node recoil after depositing work
        n.drift_vel *= 0.2
        return   # one bubble per step max


def internal_nodes(bubbles, nodes):
    for bub in bubbles:
        bub.age += 1
        cfg  = BUBBLE_TYPES.get(bub.type, BUBBLE_TYPES["H"])
        if np.random.rand() < INTERNAL_NODE_RATE*cfg["node_bias"]:
            dep = np.random.rand()
            p   = bub.center + np.random.randn(3)*(bub.radius*dep*0.8)
            if bub.type == "exotic":
                nt = "exotic"; s = NODE_STRENGTH_BASE*1.5*(0.8+dep*0.4)
            elif bub.type == "He":
                nt = "secondary" if np.random.rand()<0.7 else "exotic"
                s  = NODE_STRENGTH_BASE*1.2*(0.8+dep*0.2)
            else:
                nt = "secondary"; s = NODE_STRENGTH_BASE*(0.7+dep*0.3)
            n = WorkNode(p, s, node_type=nt, origin=f"bubble_{bub.type}")
            # Internal bubble nodes start with inward/random drift
            # so they actively sweep through the bubble contents
            n.drift_vel = np.random.randn(3) * 0.15
            n._epoch = _current_epoch
            nodes.append(n)
    for n in nodes[:]:
        if not n.is_alive(): nodes.remove(n)


def inside_any_bubble(pos, bubbles):
    for b in bubbles:
        if np.linalg.norm(pos-b.center) < b.radius: return b
    return None


# ============================================================
# EPOCH-SPECIFIC EVENTS
# ============================================================

def pop3_star_formation(world, bubbles, nodes, epoch):
    """Pop III: first massive metal-free stars near any work node."""
    if epoch < 3: return
    for b in world:
        if b.el != "H" or b.kind == "star": continue
        near_node = any(np.linalg.norm(b.pos - n.pos) < NODE_RADIUS * 4
                        for n in nodes if n.is_alive())
        if not near_node: continue
        if b.mass > 40 and b.rho > 0.01:
            b.kind = "star"; b.radius = 0.6
            b.mass *= 1.5;   b.T = TH_STAB * 5
            bub = inside_any_bubble(b.pos, bubbles)
            if bub: bub.has_first_stars = True
            break


def star_formation(world, bubbles, nodes, epoch=0):
    """Stars form near work nodes (dark matter halos).
    Dominant nodes have lower Jeans threshold — deeper potential well.
    """
    dominant  = [n for n in nodes if n.node_type=="primary" and n.is_alive()]
    satellite = [n for n in nodes if n.node_type!="primary" and n.is_alive()]

    for b in world:
        if b.el != "H" or b.kind == "star": continue

        # Check nearest dominant node
        d_dom = min((np.linalg.norm(b.pos-n.pos) for n in dominant), default=999)
        d_sat = min((np.linalg.norm(b.pos-n.pos) for n in satellite), default=999)

        near_dominant  = d_dom < NODE_RADIUS * 4   # wide capture radius
        near_satellite = d_sat < NODE_RADIUS * 2

        if not (near_dominant or near_satellite): continue

        # Dominant node: lower mass + density threshold (deep halo potential)
        if near_dominant:
            jeans_ok = b.mass > 40 and b.rho > 0.01 and b.T < TH_STAB * 12
        else:
            jeans_ok = b.mass > 60 and b.rho > 0.02 and b.T < TH_STAB * 8

        if jeans_ok:
            b.kind   = "star"
            b.radius = 0.4
            b.T      = max(b.T, TH_STAB * 2)

    if epoch >= 3:
        pop3_star_formation(world, bubbles, nodes, epoch)


def reionization_feedback(world, bubbles, epoch):
    if epoch < 4: return
    stars = [b for b in world if b.kind == "star"]
    for star in stars:
        nbs = spatial_hash.query_radius(star.pos, 4.0)
        for o in nbs:
            if o is star: continue
            if o.el in ("H","He"):
                # UV photon energy deposit scales with star T
                dT = 1.5 * star.T / max(star.mass, 1)
                o.T = min(o.T + dT, star.T*0.5)
                if o.el == "H" and o.T > TH_E_P*2:
                    o.el = "p"   # photo-ionization


def galaxy_formation(world, bubbles, nodes, epoch, step=0):
    """
    Galaxy assembly, stability tracking, and collapse detection.

    Collapse types observed:
    - peak_work_density approaches 120 (CRITICAL_WORK_DENSITY) → galactic BH
    - galaxy dissolves WITHOUT peak → direct collapse / tidal disruption
    """
    if epoch < 5: return
    for bub in bubbles:
        if bub.dead: continue

        # Stars inside this bubble OR within 6 units of bubble center
        # (galaxy extends beyond the formal bubble boundary)
        ls = [b for b in world if b.kind == "star"
              and np.linalg.norm(b.pos - bub.center) < max(bub.radius, 6.0)]
        prev_count      = bub.star_count
        bub.star_count  = len(ls)
        bub.galaxy_mass = sum(b.mass for b in ls)

        # Track peak work density inside bubble
        local_clouds = [b for b in world
                        if not isinstance(b, (WhiteHole, BlackHole))
                        and np.linalg.norm(b.pos - bub.center) < bub.radius]
        if local_clouds:
            wd = max(local_work_density_fast(b) for b in local_clouds)
            if wd > bub.peak_work_density:
                bub.peak_work_density = wd

        # ── Stability: galaxy is stable if star count doesn't drop fast ──
        if bub.star_count >= 3:
            bub.stability = min(1.0, bub.stability + 0.01)
            bub.radius   += 0.005
            bub.energy   += 2.0

            # Gravitational binding: pull stars toward bubble center
            for star in ls:
                direction = bub.center - star.pos
                d = np.linalg.norm(direction)
                if d > 0.1:
                    star.vel += direction / d * 0.005 * bub.stability

        elif bub.star_count < prev_count and prev_count >= 3:
            # Galaxy is losing stars — destabilising
            bub.stability = max(0.0, bub.stability - 0.05)
            star_loss = prev_count - bub.star_count

            # ── COLLAPSE DETECTION ─────────────────────────────────────
            if bub.peak_work_density >= CRITICAL_WORK_DENSITY * 0.6:
                # Significant work density peak → galactic central BH formed
                # (even if it briefly dipped, the peak tells us what happened)
                bh = BlackHole(bub.center.copy())
                bh.mass = bub.galaxy_mass * 0.1   # BH is ~10% of galaxy mass
                world.append(bh)
                bub.bh_id = id(bh)
                log_collapse(step, "galactic_bh", bub.center,
                             bh.mass, bub.peak_work_density, prev_count)
                bub.collapse_count += 1
                bub.peak_work_density = 0.0   # reset for next cycle
            elif star_loss >= 2:
                # Galaxy dissolves without BH — direct collapse / tidal disruption
                log_collapse(step, "direct_collapse", bub.center,
                             bub.galaxy_mass, bub.peak_work_density, prev_count)
                bub.collapse_count += 1
                bub.peak_work_density = 0.0

        # ── Cosmic Noon: enhanced star formation ──
        if epoch >= 6 and np.random.rand() < 0.003:
            pos = bub.center + np.random.randn(3) * bub.radius * 0.5
            nodes.append(WorkNode(pos, NODE_STRENGTH_BASE * 2.0,
                                  node_type="primary", origin="cosmic_noon"))

        # ── AGN flare: existing BH accretes and flares ────────────────
        if bub.bh_id is not None:
            bhs = [b for b in world if isinstance(b, BlackHole)
                   and id(b) == bub.bh_id]
            if bhs:
                central_bh = bhs[0]
                if central_bh.T > BH_EXPLODE_E * 0.7:
                    log_collapse(step, "agn_flare", central_bh.pos,
                                 central_bh.mass, central_bh.T, bub.star_count)


def cosmic_epoch_events(world, bubbles, nodes, iteration):
    global _current_epoch
    epoch = get_current_epoch(iteration)
    _current_epoch = epoch

    if epoch >= 2:
        for b in world:
            if isinstance(b, Cloud) and b.el == "p" and b.T < TH_E_P:
                if np.random.rand() < 0.005*(1 + b.rho/30.0):
                    b.el = "H"

    star_formation(world, bubbles, nodes, epoch)
    reionization_feedback(world, bubbles, epoch)
    galaxy_formation(world, bubbles, nodes, epoch, step=iteration)
    return epoch


# ============================================================
# UNIVERSE FEEDBACK
# ============================================================
def universe_feedback(bubbles, world):
    for bub in bubbles:
        if bub.dead: continue
        local = [b for b in world
                 if np.linalg.norm(b.pos - bub.center) < bub.radius]
        W = sum(b.work for b in local)
        bub.energy += W * UNIVERSE_WORK_FEEDBACK
        # Bubble radiates energy into internal clouds, heating them
        # This enables chemistry: cold nu clouds inside get heated to TH_NU_E
        if local and bub.energy > 10:
            heat = min(bub.energy * 0.001, 5.0) / len(local)
            for b in local:
                b.T = max(b.T, b.T + heat)


def interact_universes(bubbles, world):
    for b in bubbles:
        if b.dead: continue
        for o in world:
            if isinstance(o, (WhiteHole, BlackHole)): continue
            d = np.linalg.norm(o.pos - b.center)
            if d < b.radius:
                o.T   += 1.0
                o.vel += (o.pos - b.center) * 0.005
            elif d < b.radius * 2.5:
                direction = (b.center - o.pos) / (d + 1e-9)
                strength  = 0.02 * (b.radius / max(d, 0.1)) ** 2
                o.vel    += direction * strength


def node_gravity(world, nodes):
    """Dominant nodes act as dark matter halos — gravitationally attract
    nearby H/He/star clouds.  This concentrates matter for galaxy formation.
    Stronger than bubble attraction and epoch-independent."""
    dominant = [n for n in nodes
                if n.node_type == "primary" and n.is_alive()]
    if not dominant: return

    for o in world:
        if isinstance(o, (WhiteHole, BlackHole)): continue
        if o.kind == "cloud" and o.el not in ("H","He","C","O","Fe","e","p"):
            continue   # only attract assembled matter + stars

        for dom in dominant:
            r_vec = dom.pos - o.pos
            d     = np.linalg.norm(r_vec) + 1e-9
            if d > 12.0: continue   # limited range

            # Gravitational pull ∝ strength / d²
            pull = dom.strength / NODE_STRENGTH_BASE * 0.008 / (d * d)
            pull = min(pull, 0.05)   # cap
            o.vel += r_vec / d * pull

            # Bonus: H clouds very close to dominant node get mass boost
            # (gas accretes onto proto-galactic halo)
            if o.el == "H" and d < NODE_RADIUS * 2:
                o.mass   = min(o.mass * 1.002, CLOUD_SPLIT_MASS * 0.8)
                o.radius = (o.mass / (4/3*np.pi)) ** (1/3) * 0.1


def phase_walls(b):
    if np.linalg.norm(b.pos) > PHASE_WALL_R:
        b.vel *= -0.6


def space_decay(world):
    world[:] = [b for b in world if np.linalg.norm(b.pos) <= SPACE_DECAY_R]


# ============================================================
# UNIVERSE EVOLUTION
# ============================================================
def universe_evolution(bubbles, world, nodes):
    for u in bubbles:
        u.age += 1
        matter = [b for b in world if np.linalg.norm(b.pos-u.center) < u.radius]
        work   = sum(b.work for b in matter)*0.002
        u.energy -= work + 0.02*u.radius
        if u.energy > 80: u.radius += 0.001
        else:             u.radius *= 0.999
        if u.energy < 5:  u.dead = True


def decay_dead_universe(u, nodes):
    if not getattr(u,'dead',False): return
    dr = {"primary":0.97,"secondary":0.95,"tertiary":0.92,"exotic":0.85}
    for n in nodes[:]:
        if np.linalg.norm(n.pos-u.center) < u.radius:
            n.life *= dr.get(n.node_type,0.95)
            if n.life < 0.01: nodes.remove(n)


def node_interactions_in_bubbles(bubbles, nodes):
    """Node interactions: dominant nodes repel each other (halo exclusion),
    satellites get absorbed by nearest dominant (hierarchical clustering)."""
    dominant  = [n for n in nodes if n.node_type == "primary" and n.is_alive()]
    satellite = [n for n in nodes if n.node_type != "primary" and n.is_alive()]

    # Dominant ↔ dominant: strong repulsion — halos don't overlap
    for i, n1 in enumerate(dominant):
        for n2 in dominant[i+1:]:
            r_vec = n1.pos - n2.pos
            d     = np.linalg.norm(r_vec) + 1e-9
            if d < 8.0:   # exclusion radius between dominant nodes
                push = r_vec / d * (8.0 - d) * 0.005
                n1.drift_vel += push
                n2.drift_vel -= push

    # Satellite → dominant: absorbed if very close, else weak attraction
    for sat in satellite[:]:
        if not sat.is_alive(): continue
        dists = [(np.linalg.norm(sat.pos - dom.pos), dom) for dom in dominant]
        if not dists: continue
        d_min, nearest = min(dists, key=lambda x: x[0])
        if d_min < NODE_RADIUS:
            # Absorption: dominant gets stronger, satellite fades
            nearest.strength = min(nearest.strength * 1.002,
                                   NODE_STRENGTH_BASE * 12.0)
            sat.life *= 0.97
        elif d_min < 5.0:
            # Weak gravitational pull toward dominant
            direction = (nearest.pos - sat.pos) / (d_min + 1e-9)
            sat.drift_vel += direction * 0.002

    # Within bubbles: light exchange between same-type nodes
    for u in bubbles:
        bn = [n for n in nodes if np.linalg.norm(n.pos-u.center) < u.radius]
        for i, n1 in enumerate(bn):
            for n2 in bn[i+1:]:
                if n1.node_type == n2.node_type == "secondary":
                    d = np.linalg.norm(n1.pos - n2.pos)
                    if d < NODE_RADIUS * 0.5:
                        ex = min(n1.strength, n2.strength) * 0.001
                        n1.strength += ex; n2.strength += ex


# ============================================================
# NODES UPDATE
# ============================================================
def update_nodes(nodes, world=None, epoch=0):
    dead = []
    for n in nodes:
        n._epoch = epoch          # inject current epoch for drift scaling
        lw = 0
        if world and spatial_hash.grid:
            nb = spatial_hash.query_radius(n.pos, NODE_RADIUS)
            lw = sum(b.work for b in nb)/(len(nb)+1)
        n.update(local_work_density=lw, dt=0.016)
        if not n.is_alive(): dead.append(n)
    for n in dead:
        if n in nodes: nodes.remove(n)


# ============================================================
# BH ACCRETION
# ============================================================
def accretion_fast(world):
    bhs      = [b for b in world if isinstance(b, BlackHole)]
    to_remove = set()
    for bh in bhs:
        nbs = spatial_hash.query_radius(bh.pos, bh.radius*2)
        for o in nbs:
            if o is bh or id(o) in to_remove: continue
            if np.linalg.norm(o.pos-bh.pos) < bh.radius:
                bh.accrete(o); to_remove.add(id(o))
    world[:] = [b for b in world if id(b) not in to_remove]


# ============================================================
# WORLD CREATION
# ============================================================
# Number of dominant proto-galactic nodes (= max galaxy seeds)
N_DOMINANT_NODES = 4

def create_universe():
    w = []
    # 240 hot neutrino clouds — enough for guaranteed universe formation
    for _ in range(240):
        pos = np.random.randn(3) * 8
        c   = Cloud(pos, mass=50, radius=0.5,
                    kind="cloud", el="nu",
                    temperature=FEED_ENERGY, N=10)
        w.append(c)

    white = WhiteHole([0, 0, 0])
    w.append(white)

    nodes = []

    # ── Dominant nodes: strong, placed at r≈1, will become galaxy cores ──
    # They start close but get a strong OUTWARD kick → quickly separate
    # to different regions of space, each becoming a proto-galactic halo
    for i in range(N_DOMINANT_NODES):
        angle  = 2 * np.pi * i / N_DOMINANT_NODES
        r      = 0.8
        pos    = np.array([r*np.cos(angle), r*np.sin(angle), 0.0])
        n      = WorkNode(pos, NODE_STRENGTH_BASE * 6.0,   # 6× stronger
                          node_type="primary", origin="dominant")
        # Strong outward kick — each dominant node flies to a different quadrant
        outward = pos / (np.linalg.norm(pos) + 1e-9)
        n.drift_vel = outward * NODE_DRIFT_BY_EPOCH[0] * 15.0
        nodes.append(n)

    # ── Satellite nodes: weaker, random directions, faster decay ──────────
    # They seed local structure but don't compete with dominant nodes
    for _ in range(8):
        pos = np.random.randn(3) * 0.3
        n   = WorkNode(pos, NODE_STRENGTH_BASE * 1.5,
                       node_type="secondary", origin="satellite")
        n.drift_vel = np.random.randn(3) * NODE_DRIFT_BY_EPOCH[0] * 8.0
        nodes.append(n)

    return w, white, nodes