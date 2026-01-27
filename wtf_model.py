import numpy as np

G = 0.05
H = 0.002
C_CRIT = 0.98

# Energy thresholds (MeV)
TH_NU_E  = 0.782
TH_E_P   = 1.293
TH_PN    = 9.592
TH_STAB  = 8.7

# Framework feeding
FRAMEWORK_FEED = False
FEED_RATE = 0.1
FEED_ENERGY = 6.0

WHITE_RADIUS = 3.0

# Black hole parameters - WTF Singularity Criterion
CRITICAL_WORK_DENSITY = 120.0  # Critical work density threshold
COLLAPSE_SPEED = 0.25           # Collapse speed multiplier
BH_MERGE_DIST = 1.2
BH_EXPLODE_E = 200
BH_COLLAPSE_W = 50
WORK_GRADIENT_MIN = 2.0         # Min velocity gradient for BH spawn

# Multiverse parameters
MULTIVERSE_RATE = 0.0005
PHASE_WALL_R = 12
SPACE_DECAY_R = 18

# Thermodynamics - Bubble creation
BUBBLE_FORMATION_COST = 5000     # how much mass to burn
BUBBLE_MIN_ENERGY = 300         # minimum local energy
INTERNAL_NODE_RATE = 0.002      # was 0.02 -> much less
INTERNAL_NODE_STRENGTH = 1.0

# Internal node control
NODE_BURN_RATE = 0.0005         # probability of node burnout
NODE_MAX_AGE = 4000             # maximum node age

# New work node parameters (интеграция новых функций)
NODE_BASE_COUNT = 4              # basic node count
NODE_RADIUS = 1.5                # zone of influence
NODE_STRENGTH_BASE = 18.0        # weaker
NODE_DRIFT = 0.0015              # node drift

# Universe feedback
UNIVERSE_WORK_FEEDBACK = 0.03   # coefficient of feedback

# Bubble types with energy and node rate
BUBBLE_TYPES = {
    "H":  {"base_el": "H",  "node_bias": 1.0, "energy": 120, "node_rate": 0.002},
    "He": {"base_el": "He", "node_bias": 1.8, "energy": 200, "node_rate": 0.006},
    "exotic": {"base_el": "C",  "node_bias": 0.6, "energy": 300, "node_rate": 0.01}
}

# Framework thermodynamic reservoir
FRAMEWORK_RESERVOIR = 1e7       # "battery" of the framework
FRAMEWORK_DECAY = 0.00001       # degradation of the framework

ELEMENTS = ["nu", "e", "p", "n", "H", "He", "C", "O", "Fe", "Ni"]

# Global framework reservoir (must be accessible across updates)
_framework_reservoir = FRAMEWORK_RESERVOIR

# --- body ---
class Body:
    def __init__(self, pos, mass, radius, kind, el="nu"):
        self.pos = np.array(pos, float)
        self.vel = np.random.randn(3) * 0.01
        self.mass = mass
        self.radius = radius
        self.kind = kind
        self.el = el
        self.energy = FEED_ENERGY
        self.work = 0
        self.state = "material"
        self.coherence = np.random.rand()
        self.cluster_id = None

    def gravity(self, others):
        F = np.zeros(3)
        for o in others:
            if o is self: 
                continue
            r = o.pos - self.pos
            d = np.linalg.norm(r) + 1e-6
            F += G * self.mass * o.mass * r / (d**3)
        return F

    def update(self, others, dt, nodes=None):
        # Expansion
        self.vel += self.pos * H * dt

        # Gravity
        a = self.gravity(others) / self.mass
        self.vel += a * dt

        # Work field influence from nodes
        if nodes:
            w_field = work_field(self, nodes)
            self.vel += w_field * dt

        self.pos += self.vel * dt

        # Work accumulation
        self.work += np.linalg.norm(a) * dt

        # Phase transition
        self.coherence += 0.001 * dt
        if self.coherence > C_CRIT:
            self.state = "frozen"

        # Star evolution
        if self.kind == "star":
            self.mass += 0.02
            if self.mass > 600:
                self.supernova()

    def supernova(self):
        self.kind = "particle"
        self.mass *= 0.25
        self.radius *= 0.4
        self.el = np.random.choice(ELEMENTS[3:])
        self.energy = 6.0


# --- white hole node (inversion) ---
class WhiteHole(Body):
    def __init__(self, pos):
        super().__init__(pos, 18000, 0.7, "whitehole", "nu")

    def process(self, world):
        for b in world[:]:
            d = np.linalg.norm(b.pos - self.pos)

            # zone of repulsion
            if d < WHITE_RADIUS:
                r = b.pos - self.pos
                r /= np.linalg.norm(r) + 1e-6

                # charge
                b.energy += 5
                b.vel += r * 0.4

                # if black hole -> explosion
                if b.kind == "bh":
                    self.explode_bh(b, world)

                # if not black hole -> revolution
                else:
                    self.revolution(b)

    def explode_bh(self, bh, world):
        for i in range(40):
            p = Body(
                bh.pos + np.random.randn(3) * 0.4,
                30,
                0.1,
                "particle",
                "nu"
            )
            p.vel = np.random.randn(3) * 0.6
            p.energy = 20
            world.append(p)

        if bh in world:
            world.remove(bh)

    def revolution(self, b):
        # return to the field
        b.el = "nu"
        b.kind = "particle"
        b.energy = 15
        b.mass = 40
        b.radius = 0.1


# --- condense ---
def neutrino_condense(b):
    if b.el == "nu" and b.mass > 400:
        b.el = "H"
        b.kind = "particle"
        b.radius = 0.12


# --- assmemly of particles ---
def assemble(b):
    E = b.energy

    if b.el == "nu" and E > TH_NU_E:
        b.el = "e"

    if b.el == "e" and E > TH_E_P:
        b.el = "p"

    if b.el == "p" and E > TH_PN:
        b.el = "H"


# --- nuclear fusion ---
def fuse(b1, b2):
    if b1.el == "H": 
        b1.el = "He"
    elif b1.el == "He": 
        b1.el = "C"
    elif b1.el == "C": 
        b1.el = "O"
    elif b1.el == "O": 
        b1.el = "Fe"


# --- collision detection and fusion ---
def collide(b1, b2):
    d = np.linalg.norm(b1.pos - b2.pos)
    if d < (b1.radius + b2.radius):
        fuse(b1, b2)
        b1.mass += b2.mass
        return True
    return False


# --- fueling the framework with a reservoir ---
def framework_feed(world):
    global _framework_reservoir

    if not FRAMEWORK_FEED:
        return

    if _framework_reservoir <= 0:
        return  # death of framework

    # Decay of framework
    _framework_reservoir *= (1 - FRAMEWORK_DECAY)

    if np.random.rand() < FEED_RATE:
        p = Body(
            np.random.randn(3) * 6,
            50,
            0.12,
            "particle",
            "nu"
        )
        p.energy = FEED_ENERGY
        world.append(p)

        _framework_reservoir -= FEED_ENERGY * 10  # expense


# --- total material energy in the world ---
def total_material_energy(world):
    """Calculate total energy of all particles (excluding white hole)"""
    return sum([b.energy for b in world if b.kind != "whitehole"])


# --- supernova explosion ---
def supernova(b, world):
    if b.mass > 800:
        for i in range(20):
            p = Body(
                b.pos + np.random.randn(3) * 0.3,
                30,
                0.1,
                "particle",
                "H"
            )
            p.vel = np.random.randn(3) * 0.4
            p.energy = 15
            world.append(p)

        b.mass *= 0.3
        b.el = "Fe"


# --- black hole ---
class BlackHole(Body):
    def __init__(self, pos):
        super().__init__(pos, 50000, 1.0, "bh", "bh")
        self.inner_work = 0
        self.outer_work = 0
        self.resource = 0

    def accrete(self, b):
        self.mass += b.mass
        self.outer_work += b.work
        self.energy += b.energy

    def update_bh(self, world):
        # Accelerated collapse - faster contraction
        self.inner_work += self.mass * COLLAPSE_SPEED

        # Collapse mechanism
        if self.inner_work - self.outer_work > BH_COLLAPSE_W:
            self.resource += self.mass * 0.4
            self.mass *= 0.7  # Faster collapse

        # Explosion on high energy
        if self.energy > BH_EXPLODE_E:
            explode_bh(self, world)


def spawn_black_holes(world, white):
    """Spawn black holes only at work density nodes
    
    Singularity forms ONLY if:
    ρW > CRITICAL_WORK_DENSITY AND ∇W → maximum (gradient check)
    """
    for b in world:
        # Prohibit inside white hole zone
        if np.linalg.norm(b.pos - white.pos) < WHITE_RADIUS * 1.5:
            continue

        # Calculate work density
        rho = local_work_density(b, world)

        # Calculate work gradient (velocity as proxy for local gradient)
        grad = np.linalg.norm(b.vel)

        # Singularity criterion: high work density AND work gradient
        if rho > CRITICAL_WORK_DENSITY and grad > WORK_GRADIENT_MIN:
            # Only spawn at high-density nodes
            world.append(BlackHole(b.pos.copy()))
            return


def merge_black_holes(world):
    bhs = [b for b in world if isinstance(b, BlackHole)]
    for i in range(len(bhs)):
        for j in range(i + 1, len(bhs)):
            d = np.linalg.norm(bhs[i].pos - bhs[j].pos)
            if d < BH_MERGE_DIST:
                bhs[i].mass += bhs[j].mass
                bhs[i].energy += bhs[j].energy
                if bhs[j] in world:
                    world.remove(bhs[j])
                return


def explode_bh(bh, world):
    for i in range(60):
        p = Body(
            bh.pos + np.random.randn(3),
            30,
            0.1,
            "particle",
            "nu"
        )
        p.energy = 20
        p.vel = np.random.randn(3)
        world.append(p)

    if bh in world:
        world.remove(bh)


# Work nodes - structure generators
class WorkNode:
    def __init__(self, pos, strength):
        self.pos = np.array(pos, float)
        self.strength = strength


def work_field(b, nodes):
    """Calculate work field force from nodes"""
    F = np.zeros(3)
    for n in nodes:
        r = n.pos - b.pos
        d = np.linalg.norm(r) + 1e-6
        F += n.strength * r / (d**2)
    return F


def local_work_density(b, world):
    """Calculate local work density around body b
    
    ρW = dW/dt / V
    where dW/dt is work per unit time and V is local volume
    """
    R = 2.0  # Local radius
    local = [o for o in world
             if np.linalg.norm(o.pos - b.pos) < R]

    V = 4/3 * np.pi * R**3
    dWdt = sum([o.work for o in local])

    return dWdt / V if V > 0 else 0


# Multiverse bubbles
class UniverseBubble:
    def __init__(self, center, btype="H"):
        self.center = np.array(center)
        self.radius = 3
        self.energy = 100
        self.type = btype
        self.age = 0


def spawn_multiverse(bubbles, world):
    """Spawn multiverse bubbles with energy and mass cost checks
    
    Requirements:
    - Total material energy must exceed BUBBLE_MIN_ENERGY
    - Enough particles to burn for BUBBLE_FORMATION_COST
    - Randomly selects bubble type from BUBBLE_TYPES
    """
    if not FRAMEWORK_FEED:
        return  # without framework feeding, no or few bubbles

    E = total_material_energy(world)

    if E < BUBBLE_MIN_ENERGY:
        return

    # expend mass from particles - collect particles to burn
    burned = 0
    
    for b in world[:]:
        if b.kind == "particle":
            burned += b.mass
            world.remove(b)
            if burned >= BUBBLE_FORMATION_COST:
                break

    if burned < BUBBLE_FORMATION_COST:
        return

    # Random bubble type selection
    btype = np.random.choice(list(BUBBLE_TYPES.keys()))

    bubbles.append(
        UniverseBubble(np.random.randn(3) * 15, btype)
    )


def internal_nodes(bubbles, nodes):
    """Generate internal work nodes inside universe bubbles
    
    - Bubble type determines node generation bias
    - Nodes can burn out with NODE_BURN_RATE probability
    - Track node age up to NODE_MAX_AGE
    """
    for bub in bubbles:
        bub.age += 1

        # Get bubble type bias (higher bias = more node generation)
        bias = BUBBLE_TYPES[bub.type]["node_bias"]

        # Generation of new nodes based on bubble type
        if np.random.rand() < INTERNAL_NODE_RATE * bias:
            # Random position inside bubble
            p = bub.center + np.random.randn(3) * bub.radius
            nodes.append(
                WorkNode(p, INTERNAL_NODE_STRENGTH)
            )

    # Node burnout mechanism
    for n in nodes[:]:
        if np.random.rand() < NODE_BURN_RATE:
            nodes.remove(n)


def inside_any_bubble(pos, bubbles):
    """Check if position is inside any bubble
    
    Returns bubble if inside, None otherwise
    """
    for b in bubbles:
        if np.linalg.norm(pos - b.center) < b.radius:
            return b
    return None


def star_formation(world, bubbles):
    """Form stars from hydrogen inside bubbles
    
    H particles with mass > 300 inside bubbles become stars
    """
    for b in world:
        bub = inside_any_bubble(b.pos, bubbles)
        if not bub: 
            continue

        if b.el == "H" and b.mass > 300:
            b.kind = "star"
            b.radius = 0.4


def universe_feedback(bubbles, world):
    """Apply work-based energy feedback from universe bubbles
    
    Local work in bubble volume feeds back as energy:
    E_bubble += W_local * UNIVERSE_WORK_FEEDBACK
    """
    for bub in bubbles:
        # Collect particles inside this bubble
        local = [b for b in world
                 if np.linalg.norm(b.pos - bub.center) < bub.radius]

        # Sum local work
        W = sum([b.work for b in local])

        # Apply feedback
        bub.energy += W * UNIVERSE_WORK_FEEDBACK


def interact_universes(bubbles, world):
    for b in bubbles:
        for o in world:
            d = np.linalg.norm(o.pos - b.center)
            if d < b.radius:
                o.energy += 2
                o.vel += (o.pos - b.center) * 0.02


# Phase walls - reflection at boundary
def phase_walls(b):
    d = np.linalg.norm(b.pos)
    if d > PHASE_WALL_R:
        b.vel *= -0.6


# Space decay - removal at far distance
def space_decay(world):
    for b in world[:]:
        d = np.linalg.norm(b.pos)
        if d > SPACE_DECAY_R:
            world.remove(b)


# --- spiral ---
def make_spiral(center):
    out = []
    for a in range(3):
        for i in range(25):
            r = i * 0.25
            th = i * 0.5 + a * 2 * np.pi / 3
            p = center + np.array([r*np.cos(th), r*np.sin(th), 0])
            out.append(Body(p, 300, 0.4, "star", "H"))
    return out


# --- world creation ---
def create_universe():
    w = []

    # Neutrino field
    for i in range(200):
        w.append(
            Body(np.random.randn(3) * 8, 40, 0.08, "particle", "nu")
        )

    white = WhiteHole([0, 0, 0])
    w.append(white)

    # Initialize work nodes (future galaxy nodes)
    nodes = [
        WorkNode(np.array([-5.0, 0.0, 0.0]), 50.0),
        WorkNode(np.array([5.0, 0.0, 0.0]), 50.0),
        WorkNode(np.array([0.0, 4.0, 0.0]), 30.0),
    ]

    return w, white, nodes


def update_nodes(nodes):
    """Update work nodes - drift and degradation
    
    Nodes slowly drift through space with NODE_DRIFT velocity
    Life degrades very slowly (0..1 scale)
    Minimum life of 0.2 prevents complete death
    """
    for n in nodes:
        # Random drift movement
        n.vel = np.random.randn(3) * NODE_DRIFT
        n.pos += n.vel
        
        # Very slow degradation
        if not hasattr(n, 'life'):
            n.life = 1.0
        
        n.life -= 0.00002  # very slow decay
        n.life = max(n.life, 0.2)  # prevent complete death


def work_field_from_nodes(b, nodes):
    """Calculate enhanced work field force from all nodes
    
    Each node contributes a force based on:
    - Node life (degradation factor)
    - Node strength
    - Distance-dependent falloff (1/d^2)
    - NODE_RADIUS limits sphere of influence
    """
    F = np.zeros(3)
    for n in nodes:
        r = n.pos - b.pos
        d = np.linalg.norm(r)
        
        # Only influence within NODE_RADIUS
        if d < NODE_RADIUS:
            # Avoid singularity with epsilon
            life = n.life if hasattr(n, 'life') else 1.0
            F += life * n.strength * r / (d**2 + 1e-6)
    
    return F


def spawn_internal_nodes(bubbles, nodes):
    """Spawn internal work nodes inside universe bubbles
    
    Node generation rate depends on bubble type:
    - H: 0.002 rate (slow)
    - He: 0.006 rate (moderate)
    - exotic: 0.01 rate (fast)
    
    Each bubble spawns nodes within its radius
    """
    for u in bubbles:
        u_type = u.type if hasattr(u, 'type') else "H"
        
        cfg = BUBBLE_TYPES.get(u_type, BUBBLE_TYPES["H"])
        if np.random.rand() < cfg["node_rate"]:
            # Random position inside bubble
            pos = u.center + np.random.randn(3) * (u.radius * 0.6)
            
            # Create new node with base strength
            new_node = WorkNode(pos, NODE_STRENGTH_BASE * 0.8)
            new_node.life = 1.0  # Initialize life
            nodes.append(new_node)


def universe_evolution(bubbles, world, nodes):
    """Evolve universe bubbles through work and thermodynamics
    
    Processes for each bubble:
    1. Age increment
    2. Energy consumption from local work
    3. Thermal heat loss (leakage)
    4. Radius growth/contraction based on energy
    5. Heat death transition
    """
    for u in bubbles:
        u.age += 1
        
        # Find all particles inside this bubble
        matter = [b for b in world
                  if np.linalg.norm(b.pos - u.center) < u.radius]
        
        # Energy consumption from work done by matter
        work = sum(b.work for b in matter) * 0.002
        u.energy -= work
        
        # Thermal leakage proportional to surface area
        u.energy -= 0.02 * u.radius
        
        # Dynamic radius adjustment based on energy
        if u.energy > 80:
            u.radius += 0.001  # expansion
        else:
            u.radius *= 0.999  # contraction
        
        # Heat death check
        if u.energy < 5:
            u.dead = True


def decay_dead_universe(u, nodes):
    """Decay work nodes inside dead universe
    
    When universe bubble dies:
    - All work nodes inside gradually lose life
    - Node life decreases by 5% per update
    - Eventually nodes fade away
    """
    if not hasattr(u, 'dead') or not u.dead:
        return
    
    for n in nodes[:]:
        if np.linalg.norm(n.pos - u.center) < u.radius:
            if not hasattr(n, 'life'):
                n.life = 1.0
            n.life *= 0.95  # gradual decay
            
            # Remove completely dead nodes
            if n.life < 0.01:
                nodes.remove(n)