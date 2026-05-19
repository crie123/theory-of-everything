import streamlit as st
import json
import matplotlib.pyplot as plt
from wtf_model import (
    create_universe, framework_feed, spatial_hash, sph_density_pass,
    assemble, accretion_fast, collide_all_fast,
    merge_clouds, split_cloud, evaporate_small_clouds,
    merge_black_holes, spawn_black_holes,
    spawn_multiverse, internal_nodes, update_nodes,
    node_interactions_in_bubbles, universe_evolution, decay_dead_universe,
    cosmic_epoch_events, universe_feedback, interact_universes,
    space_decay, phase_walls, get_current_epoch, get_epoch_info,
    total_material_energy, get_framework_reservoir, local_work_density_fast,
    WhiteHole, BlackHole, Cloud,
    FRAMEWORK_RESERVOIR, CRITICAL_WORK_DENSITY, COSMIC_EPOCHS,
    COLLAPSE_LOG, log_collapse, node_gravity,
)
import numpy as np
import time
import pickle
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🌌 WTF – Universe with Work Density Singularity")

# ============== SIDEBAR CONTROLS ==============
with st.sidebar:
    st.markdown("### ⚙️ Simulation Controls")
    col1, col2 = st.columns(2)
    with col1:
        dt = st.slider("dt", 0.01, 0.3, 0.05)
    with col2:
        speed = st.slider("Speed", 1, 40, 12)

    st.markdown("---")
    st.markdown("### 🎮 Advanced Options")
    live_mode = st.toggle("🔴 Live Mode", False)
    if live_mode:
        live_fps = st.slider("Updates/sec", 1, 30, 10)

    batch_mode = st.toggle("📦 Batch Mode", False)
    if batch_mode:
        batch_size = st.slider("Batch Size", 10, 500, 100, step=10)
        auto_stop = st.toggle("Auto-stop at condition", False)
        if auto_stop:
            stop_condition = st.selectbox(
                "Stop when:",
                ["Max work density reached", "Black holes > N", "Bubbles spawned"]
            )
            stop_threshold = st.number_input("Threshold value", 0, 1000, 500)

    st.markdown("---")
    st.markdown("### 📊 Visualization")
    show_3d    = st.toggle("Show 3D View", True)
    show_tree  = st.toggle("Show State Tree", True)
    show_history = st.toggle("Show History Graphs", True)

    st.markdown("---")
    st.markdown("### 💾 Persistence")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save State"):
            snapshot = {
                "iteration": st.session_state.iteration,
                "world": st.session_state.world,
                "white": st.session_state.white,
                "nodes": st.session_state.nodes,
                "bubbles": st.session_state.bubbles,
                "history": st.session_state.work_history,
                "timestamp": datetime.now().isoformat()
            }
            filename = f"snapshot_{st.session_state.iteration}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(snapshot, f)
            st.success(f"✅ Saved to {filename}")
    with col2:
        if st.button("📂 Load State"):
            try:
                import glob
                files = glob.glob("snapshot_*.pkl")
                if files:
                    latest = max(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                    with open(latest, "rb") as f:
                        snapshot = pickle.load(f)
                    st.session_state.update(snapshot)
                    st.success(f"✅ Loaded {latest}")
            except Exception as e:
                st.error(f"Error loading: {e}")


# ============== SESSION STATE INITIALIZATION ==============
if "world" not in st.session_state:
    w, white, nodes = create_universe()
    st.session_state.world = w
    st.session_state.white = white
    st.session_state.nodes = nodes
    st.session_state.bubbles = []
    st.session_state.work_history = []
    st.session_state.bh_count = []
    st.session_state.work_density_max = []
    st.session_state.iteration = 0
    st.session_state.element_counts = {}
    st.session_state.bh_history = []
    st.session_state.performance_metrics = {}
    st.session_state.stop_flag = False
    st.session_state.reservoir_history = []
    st.session_state.node_stats_history = []
    st.session_state.epoch_history = []        # NEW: epoch per iteration
    st.session_state.galaxy_count_history = [] # NEW
    st.session_state.star_count_history   = []
    st.session_state.collapse_history     = []   # collapse events


# ============== STEP FUNCTION ==============
def step():
    w      = st.session_state.world
    white  = st.session_state.white
    bubbles = st.session_state.bubbles
    nodes  = st.session_state.nodes
    iteration = st.session_state.iteration

    start_time = time.time()

    # --- Rebuild spatial hash ONCE per step ---
    spatial_hash.build(w)

    framework_feed(w)
    white.process(w)

    # SPH density pass — sets rho, P, cs for all clouds at once
    epoch = get_current_epoch(iteration)
    sph_density_pass(w)

    # Cloud update (SPH forces + cooling)
    for b in w[:]:
        if isinstance(b, WhiteHole):
            continue
        if isinstance(b, BlackHole):
            b.update_bh(w)
        else:
            b.update(w, dt, nodes, use_spatial=True, epoch=epoch)
            phase_walls(b)

    # Assemble matter — epoch-aware chemistry
    for b in w:
        if not isinstance(b, (WhiteHole, BlackHole)):
            assemble(b, nodes, epoch=epoch)

    # Accretion
    accretion_fast(w)

    # Cloud-cloud collisions / fusions
    w[:] = collide_all_fast(w)

    # Cloud merging (overlapping diffuse clouds → one)
    merge_clouds(w)

    # Jeans fragmentation (overmassive clouds split)
    split_cloud(w, epoch=epoch)

    # Remove evaporated clouds
    evaporate_small_clouds(w)

    merge_black_holes(w)
    spawn_black_holes(w, white)

    # Multiverse
    spawn_multiverse(bubbles, w, nodes, step=iteration)
    internal_nodes(bubbles, nodes)
    update_nodes(nodes, w, epoch=epoch)
    node_interactions_in_bubbles(bubbles, nodes)
    universe_evolution(bubbles, w, nodes)

    for u in bubbles:
        decay_dead_universe(u, nodes)

    # === COSMIC EPOCH EVENTS (replaces old star_formation call) ===
    current_epoch = cosmic_epoch_events(w, bubbles, nodes, iteration)

    universe_feedback(bubbles, w)
    interact_universes(bubbles, w)
    node_gravity(w, nodes)
    space_decay(w)

    # --- Metrics ---
    total_work = sum(b.work for b in w if hasattr(b, "work"))
    st.session_state.work_history.append(total_work)

    bh_count = sum(1 for b in w if isinstance(b, BlackHole))
    st.session_state.bh_count.append(bh_count)
    st.session_state.bh_history.append(bh_count)

    # Work density — sample only up to 30 random bodies for speed
    sample = [b for b in w if hasattr(b, "work")]
    if len(sample) > 30:
        sample = list(np.random.choice(sample, 30, replace=False))
    work_densities = [local_work_density_fast(b) for b in sample]
    max_density = max(work_densities) if work_densities else 0
    st.session_state.work_density_max.append(max_density)

    elements = {}
    for b in w:
        if hasattr(b, "el"):
            elements[b.el] = elements.get(b.el, 0) + 1
    st.session_state.element_counts = elements

    node_stats = {
        "total": len(nodes),
        "primary": sum(1 for n in nodes if n.node_type == "primary"),
        "secondary": sum(1 for n in nodes if n.node_type == "secondary"),
        "tertiary": sum(1 for n in nodes if n.node_type == "tertiary"),
        "exotic": sum(1 for n in nodes if n.node_type == "exotic"),
        "avg_life": np.mean([n.life for n in nodes]) if nodes else 0,
        "avg_strength": np.mean([n.strength for n in nodes]) if nodes else 0,
    }
    st.session_state.node_stats_history.append(node_stats)

    # Epoch tracking
    st.session_state.epoch_history.append(current_epoch)

    # Record new collapse events
    known = len(st.session_state.collapse_history)
    if len(COLLAPSE_LOG) > known:
        st.session_state.collapse_history.extend(COLLAPSE_LOG[known:])

    # Galaxy & star counts
    stars = sum(1 for b in w if b.kind == "star")
    galaxies = sum(1 for b in bubbles if not b.dead and b.star_count >= 3)
    st.session_state.star_count_history.append(stars)
    st.session_state.galaxy_count_history.append(galaxies)

    elapsed = time.time() - start_time
    st.session_state.performance_metrics[iteration] = {
        "time": elapsed,
        "particles": len(w),
        "bh_count": bh_count,
        "work_density": max_density,
        "nodes": len(nodes),
        "bubbles": len(bubbles),
        "epoch": current_epoch,
    }

    st.session_state.reservoir_history.append(get_framework_reservoir())
    st.session_state.iteration += 1

    # Auto-stop
    if batch_mode and auto_stop:
        if stop_condition == "Max work density reached" and max_density >= stop_threshold:
            st.session_state.stop_flag = True
        elif stop_condition == "Black holes > N" and bh_count >= stop_threshold:
            st.session_state.stop_flag = True
        elif stop_condition == "Bubbles spawned" and len(bubbles) >= stop_threshold:
            st.session_state.stop_flag = True


# ============== HELPERS ==============
def render_tree(data, indent=0):
    output = ""
    for key, value in data.items():
        if isinstance(value, dict):
            output += "  " * indent + f"├─ {key}\n"
            output += render_tree(value, indent + 1)
        elif isinstance(value, set):
            for item in value:
                output += "  " * (indent + 1) + f"├─ {item}\n"
        elif value is not None:
            output += "  " * (indent + 1) + f"├─ {value}\n"
        else:
            output += "  " * indent + f"├─ {key}\n"
    return output


def display_epoch_banner():
    """Show current cosmic epoch as a prominent banner"""
    iteration = st.session_state.iteration
    epoch_info = get_epoch_info(iteration)
    eid = get_current_epoch(iteration)

    # Progress to next epoch
    epochs_with_start = [(k, v) for k, v in COSMIC_EPOCHS.items()
                         if k > eid and "iteration_start" in v]
    if epochs_with_start:
        next_eid, next_epoch = epochs_with_start[0]
        next_start = next_epoch["iteration_start"]
        curr_start = COSMIC_EPOCHS[eid].get("iteration_start", 0)
        progress = min((iteration - curr_start) / max(next_start - curr_start, 1), 1.0)
        progress_bar_html = f"<div style=\"background:#333;border-radius:4px;height:6px;margin-top:6px;\"><div style=\"background:{epoch_info['color']};width:{int(progress*100)}%;height:6px;border-radius:4px;\"></div></div><small style=\"color:#aaa;\">→ next: {next_epoch['emoji']} {next_epoch['name']} (iter {next_start})</small>"
    else:
        progress_bar_html = "<small style='color:#aaa;'>Final epoch reached</small>"

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {epoch_info['color']}33, transparent);
                border-left: 4px solid {epoch_info['color']};
                padding: 12px 16px; border-radius: 8px; margin-bottom: 12px;">
      <span style="font-size:1.4em;">{epoch_info['emoji']}</span>
      <strong style="font-size:1.1em; color:{epoch_info['color']};">
        Epoch {eid}: {epoch_info['name']}
      </strong>
      <p style="color:#ccc; margin:4px 0 0 0; font-size:0.9em;">{epoch_info['description']}</p>
      {progress_bar_html}
    </div>
    """, unsafe_allow_html=True)


def display_metrics():
    # Epoch banner first
    display_epoch_banner()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    bh = sum(1 for b in st.session_state.world if isinstance(b, BlackHole))
    nu = st.session_state.element_counts.get("nu", 0)
    h_atoms = st.session_state.element_counts.get("H", 0)
    bubbles_count = len(st.session_state.bubbles)
    max_work_density = max(st.session_state.work_density_max) if st.session_state.work_density_max else 0
    nodes_count = len(st.session_state.nodes)

    col1.metric("Iteration", st.session_state.iteration)
    col2.metric("Black Holes", bh)
    col3.metric("Neutrinos", nu)
    col4.metric("Hydrogen", h_atoms)
    col5.metric("Work Density", f"{max_work_density:.1f}")
    col6.metric("Work Nodes", nodes_count)

    col1b, col2b, col3b, col4b, col5b, col6b = st.columns(6)
    current_reservoir = get_framework_reservoir()
    reservoir_percent = (current_reservoir / FRAMEWORK_RESERVOIR) * 100
    total_energy = total_material_energy(st.session_state.world)
    stars_count = sum(1 for b in st.session_state.world if b.kind == "star")
    particles_count = sum(1 for b in st.session_state.world if b.kind == "particle")
    galaxies = sum(1 for b in st.session_state.bubbles if not b.dead and b.star_count >= 3)

    col1b.metric("Universe Bubbles", bubbles_count)
    col2b.metric("Stars", stars_count)
    col3b.metric("Galaxies", galaxies)
    col4b.metric("Total Matter Energy", f"{total_energy:.0f}")
    col5b.metric("Reservoir Left", f"{reservoir_percent:.1f}%")
    col6b.metric("Energy Spent", f"{(FRAMEWORK_RESERVOIR - current_reservoir)/1e6:.1f}M")


def display_collapses():
    """Collapse event log — galactic BH, direct collapses, AGN flares."""
    events = st.session_state.get("collapse_history", [])
    if not events:
        st.info("No collapse events yet. Galaxy disruptions appear here.")
        return
    from collections import Counter
    counts = Counter(e["type"] for e in events)
    c1, c2, c3 = st.columns(3)
    c1.metric("⚫ Galactic BH",    counts.get("galactic_bh", 0))
    c2.metric("💥 Direct collapse", counts.get("direct_collapse", 0))
    c3.metric("🔥 AGN flares",     counts.get("agn_flare", 0))
    st.markdown("---")
    if len(events) > 1:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(figsize=(14, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1a1a2e")
        colors  = {"galactic_bh":"#FF4500","direct_collapse":"#FFD700","agn_flare":"#00BFFF","framework_drain_bh":"#9B59B6"}
        markers = {"galactic_bh":"v","direct_collapse":"x","agn_flare":"*","framework_drain_bh":"D"}
        for e in events:
            ax.scatter(e["step"], e["peak_density"],
                       c=colors.get(e["type"],"#888"),
                       marker=markers.get(e["type"],"o"),
                       s=80+e["galaxy_size"]*10, alpha=0.85, zorder=3)
        ax.axhline(CRITICAL_WORK_DENSITY, color="red", lw=1.5, linestyle="--")
        ax.axhline(CRITICAL_WORK_DENSITY*0.6, color="orange", lw=1, linestyle=":")
        ax.set_xlabel("Iteration", color="white")
        ax.set_ylabel("Peak ρ at Collapse", color="white")
        ax.set_title("Collapse Events Timeline", color="white")
        ax.tick_params(colors="white")
        legend_els = [
            mpatches.Patch(color="#FF4500", label="Galactic BH"),
            mpatches.Patch(color="#FFD700", label="Direct collapse"),
            mpatches.Patch(color="#00BFFF", label="AGN flare"),
            mpatches.Patch(color="red",     label=f"Critical ρ={CRITICAL_WORK_DENSITY}"),
        ]
        ax.legend(handles=legend_els, loc="upper left",
                  facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.markdown("**Recent events (last 15):**")
    emoji = {"galactic_bh":"⚫","direct_collapse":"💥","agn_flare":"🔥","framework_drain_bh":"🕳️"}
    for e in reversed(events[-15:]):
        st.markdown(
            f"{emoji.get(e['type'],'❓')} **step {e['step']}** — "
            f"`{e['type']}` | galaxy={e['galaxy_size']}★ | "
            f"peak_ρ={e['peak_density']:.1f} | mass={e['mass']:.0f}"
        )


def display_graphs():
    if len(st.session_state.work_history) < 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    ax1, ax2, ax5, ax3, ax4, ax6 = axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]
    fig.suptitle("WTF Universe Dynamics", fontsize=14, fontweight='bold', color='white')

    ax1.plot(st.session_state.work_history, linewidth=2, color="cyan")
    ax1.set_ylabel("Total Work", color="cyan"); ax1.tick_params(axis='y', labelcolor='cyan')
    ax1.grid(True, alpha=0.3); ax1.set_title("Energy Accumulation", color='white')

    ax2.plot(st.session_state.bh_count, linewidth=2, color="magenta")
    ax2.set_ylabel("BH Count", color="magenta"); ax2.tick_params(axis='y', labelcolor='magenta')
    ax2.grid(True, alpha=0.3); ax2.set_title("Singularity Formation", color='white')

    ax3.plot(st.session_state.work_density_max, linewidth=2, color="lime")
    ax3.axhline(y=CRITICAL_WORK_DENSITY, color='red', linestyle='--', linewidth=2,
                label=f"Critical: {CRITICAL_WORK_DENSITY}")
    ax3.set_ylabel("Work Density (ρW)", color="lime"); ax3.tick_params(axis='y', labelcolor='lime')
    ax3.legend(loc='upper left', fontsize=8); ax3.grid(True, alpha=0.3)
    ax3.set_title("Singularity Threshold", color='white')

    # Node evolution
    if st.session_state.node_stats_history:
        nh = st.session_state.node_stats_history
        ax4.plot([s['total'] for s in nh], color="yellow", label="Total")
        ax4.plot([s['primary'] for s in nh], color="orange", alpha=0.7, label="Primary")
        ax4.plot([s['secondary'] for s in nh], color="cyan", alpha=0.7, label="Secondary")
        ax4.plot([s['exotic'] for s in nh], color="red", alpha=0.7, label="Exotic")
        ax4.set_ylabel("Node Count", color="yellow"); ax4.tick_params(axis='y', labelcolor='yellow')
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
        ax4.set_title("Work Nodes Evolution", color='white')

    # Reservoir
    if st.session_state.reservoir_history:
        rn = [r / FRAMEWORK_RESERVOIR for r in st.session_state.reservoir_history]
        ax5.plot(rn, linewidth=2, color="gold")
        ax5.fill_between(range(len(rn)), rn, alpha=0.3, color="gold")
        ax5.set_ylabel("Reservoir Level", color="gold"); ax5.tick_params(axis='y', labelcolor='gold')
        ax5.set_ylim([0, 1.1])
        ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax5.grid(True, alpha=0.3); ax5.set_title("Framework Energy Reservoir", color='white')

    # Stars & Galaxies (NEW!)
    if st.session_state.star_count_history:
        ax6.plot(st.session_state.star_count_history, color="#FFD700", linewidth=2, label="Stars")
        ax6.plot(st.session_state.galaxy_count_history, color="#7B68EE", linewidth=2, label="Galaxies")

        # Shade epoch regions
        epoch_colors = {0: "#FF4500", 1: "#FF8C00", 2: "#4B0082", 3: "#FFD700",
                        4: "#00BFFF", 5: "#7B68EE", 6: "#FF6347", 7: "#20B2AA"}
        if st.session_state.epoch_history:
            eh = st.session_state.epoch_history
            prev = eh[0]; start = 0
            for i, ep in enumerate(eh[1:], 1):
                if ep != prev:
                    ax6.axvspan(start, i, alpha=0.1, color=epoch_colors.get(prev, "#888"))
                    start = i; prev = ep
            ax6.axvspan(start, len(eh), alpha=0.1, color=epoch_colors.get(prev, "#888"))

        ax6.set_ylabel("Count", color="white"); ax6.tick_params(axis='y', labelcolor='white')
        ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)
        ax6.set_title("Stars & Galaxies Formation", color='white')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# Complete element → colour mapping (hex). Used in 3D view and legends.
EL_COLOR = {
    "nu":     "#4444FF",   # dim blue     — neutrino plasma
    "e":      "#00FFFF",   # cyan         — electrons
    "p":      "#FF8800",   # orange       — protons
    "n":      "#888888",   # GREY         — free neutrons (unstable, short-lived)
    "H":      "#FFFFFF",   # white        — hydrogen
    "He":     "#FFFF00",   # yellow       — helium
    "C":      "#FF6600",   # amber        — carbon
    "O":      "#00FF88",   # mint         — oxygen
    "Fe":     "#CC4444",   # red-brown    — iron
    "Ni":     "#AA66FF",   # violet       — nickel (endpoint)
    "bh":     "#000000",   # black        — black hole
    "white":  "#FFFFFF",   # white        — white hole
    "bubble": "#3333AA",   # dark blue    — universe bubble
    "node":   "#FFAA00",   # gold         — work node
    "star":   "#FFD700",   # gold-yellow  — star
}

def display_3d():
    st.markdown("### 🔮 3D Visualization")
    data = []
    for b in st.session_state.world:
        if isinstance(b, WhiteHole):
            data.append({"x": 0, "y": 0, "z": 0, "r": 1.0,
                         "kind": "white", "el": "white",
                         "color": EL_COLOR["white"]})
        else:
            el   = getattr(b, 'el', 'nu')
            kind = getattr(b, 'kind', 'cloud')
            # Stars get star colour regardless of element
            color = EL_COLOR.get("star" if kind == "star" else el,
                                 EL_COLOR.get(kind, "#888888"))
            data.append({
                "x": float(b.pos[0]), "y": float(b.pos[1]), "z": float(b.pos[2]),
                "r": b.radius, "kind": kind, "el": el, "color": color
            })
    for b in st.session_state.bubbles:
        data.append({
            "x": float(b.center[0]), "y": float(b.center[1]), "z": float(b.center[2]),
            "r": b.radius, "kind": "bubble", "el": "bubble",
            "color": EL_COLOR["bubble"]
        })
    for n in st.session_state.nodes:
        node_color = {"primary": "#FF6600", "exotic": "#FF00FF"}.get(
            n.node_type, EL_COLOR["node"])
        data.append({
            "x": float(n.pos[0]), "y": float(n.pos[1]), "z": float(n.pos[2]),
            "r": 0.5, "kind": "node", "el": "node", "color": node_color
        })
    html = open("components/wtf_3d.html").read()
    html = html.replace("__DATA__", json.dumps(data))
    st.components.v1.html(html, height=650)


def display_tree():
    st.markdown("### 🌳 Simulation State Tree")
    elements_dict = {f"{k}: {v}" for k, v in sorted(st.session_state.element_counts.items())}
    node_stats = {
        "Primary": sum(1 for n in st.session_state.nodes if n.node_type == "primary"),
        "Secondary": sum(1 for n in st.session_state.nodes if n.node_type == "secondary"),
        "Tertiary": sum(1 for n in st.session_state.nodes if n.node_type == "tertiary"),
        "Exotic": sum(1 for n in st.session_state.nodes if n.node_type == "exotic"),
    }
    epoch_info = get_epoch_info(st.session_state.iteration)
    tree_data = {
        "🌌 Universe": {
            f"Iteration: {st.session_state.iteration}": None,
            f"{epoch_info['emoji']} Epoch: {epoch_info['name']}": None,
            "🤍 White Hole": {
                f"Mass: {st.session_state.white.mass:.0f}": None,
                f"Energy: {st.session_state.white.energy:.1f}": None,
            },
            f"⚫ Black Holes ({sum(1 for b in st.session_state.world if isinstance(b, BlackHole))})": {
                f"Total Mass: {sum(b.mass for b in st.session_state.world if isinstance(b, BlackHole)):.0f}": None,
            },
            "⚛️ Particles": {
                f"Total: {len(st.session_state.world) - 1}": None,
                "Elements": elements_dict,
            },
            "🫧 Universe Bubbles": {
                f"Count: {len(st.session_state.bubbles)}": None,
                f"Galaxies: {sum(1 for b in st.session_state.bubbles if not b.dead and b.star_count >= 3)}": None,
            },
            "⚙️ Work Nodes": {
                f"Total: {len(st.session_state.nodes)}": None,
                "Types": node_stats,
            },
            "📊 System State": {
                f"Total Work: {sum(b.work for b in st.session_state.world if hasattr(b, 'work')):.1f}": None,
                f"Max Work Density: {max(st.session_state.work_density_max) if st.session_state.work_density_max else 0:.1f}": None,
            }
        }
    }
    st.code(render_tree(tree_data), language="text")


# ============== MODE SELECTION ==============
if live_mode:
    st.sidebar.info("🔴 Live mode active")
    metrics_placeholder = st.empty()
    graph_placeholder = st.empty()
    tree_placeholder = st.empty()
    viz_placeholder = st.empty()

    while live_mode:
        for _ in range(speed):
            step()
        with metrics_placeholder.container():
            display_metrics()
        with graph_placeholder.container():
            if show_history:
                display_graphs()
        with tree_placeholder.container():
            if show_tree:
                display_tree()
        with viz_placeholder.container():
            if show_3d:
                display_3d()
        time.sleep(1.0 / live_fps)

elif batch_mode:
    st.sidebar.info("📦 Batch mode active")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Run Batch"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i in range(batch_size):
                step()
                progress_bar.progress((i + 1) / batch_size)
                status_text.text(f"Batch: {i+1}/{batch_size} | "
                                 f"Epoch {get_current_epoch(st.session_state.iteration)}: "
                                 f"{get_epoch_info(st.session_state.iteration)['name']}")
                if st.session_state.stop_flag:
                    st.warning(f"⏹ Auto-stopped at iteration {st.session_state.iteration}")
                    st.session_state.stop_flag = False
                    break
            st.success("✅ Batch complete!")
    with col2:
        if st.button("📊 Analyze Batch"):
            if st.session_state.performance_metrics:
                iters = sorted(st.session_state.performance_metrics.keys())
                times = [st.session_state.performance_metrics[i]["time"] for i in iters]
                particles = [st.session_state.performance_metrics[i]["particles"] for i in iters]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(iters, times, marker='o', color='orange')
                ax1.set_xlabel("Iteration"); ax1.set_ylabel("Time per step (s)")
                ax1.set_title("Performance Timeline"); ax1.grid(True, alpha=0.3)
                ax2.plot(iters, particles, marker='s', color='cyan')
                ax2.set_xlabel("Iteration"); ax2.set_ylabel("Particle Count")
                ax2.set_title("System Size Evolution"); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.clear()
            st.rerun()

    st.markdown("---")
    display_metrics()
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Graphs", "💥 Collapses", "🌳 State", "🔮 3D"])
    with tab1:
        if show_history: display_graphs()
    with tab2:
        display_collapses()
    with tab3:
        if show_tree: display_tree()
    with tab4:
        if show_3d: display_3d()

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Step", key="step_btn"):
            step()
    with col2:
        if st.button("⏸ Run (10 steps)", key="run_btn"):
            progress_bar = st.progress(0)
            for i in range(10):
                step()
                progress_bar.progress((i + 1) / 10)
    with col3:
        if st.button("🔄 Reset Universe"):
            st.session_state.clear()
            st.rerun()

    st.markdown("---")
    display_metrics()
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Graphs", "💥 Collapses", "🌳 State", "🔮 3D"])
    with tab1:
        if show_history: display_graphs()
    with tab2:
        display_collapses()
    with tab3:
        if show_tree: display_tree()
    with tab4:
        if show_3d: display_3d()