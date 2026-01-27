import streamlit as st
import json
import matplotlib.pyplot as plt
from wtf_model import *
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
    
    # Advanced controls
    st.markdown("---")
    st.markdown("### 🎮 Advanced Options")
    
    live_mode = st.toggle("🔴 Live Mode", False)
    if live_mode:
        live_fps = st.slider("Updates/sec", 1, 30, 10)
    
    # Batch processing
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
    
    # Visualization options
    st.markdown("---")
    st.markdown("### 📊 Visualization")
    show_3d = st.toggle("Show 3D View", True)
    show_tree = st.toggle("Show State Tree", True)
    show_history = st.toggle("Show History Graphs", True)
    
    # Save/Load
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

# ============== STEP FUNCTION WITH METRICS ==============
def step():
    w = st.session_state.world
    white = st.session_state.white
    bubbles = st.session_state.bubbles
    nodes = st.session_state.nodes

    start_time = time.time()

    framework_feed(w)
    white.process(w)

    # Bodies update with work field
    for b in w[:]:  # Iterate over copy
        if isinstance(b, WhiteHole): 
            continue

        if isinstance(b, BlackHole):
            b.update_bh(w)
        else:
            b.update(w, dt, nodes)
            assemble(b)
            phase_walls(b)

    # Accretion
    for bh in [x for x in w if isinstance(x, BlackHole)]:
        for o in w[:]:
            if o is bh: 
                continue
            if np.linalg.norm(o.pos - bh.pos) < bh.radius:
                bh.accrete(o)
                if o in w: 
                    w.remove(o)

    # Collisions - use list copy to avoid index issues
    w_copy = w[:]  # Create copy for iteration
    rem = []
    for i in range(len(w_copy)):
        for j in range(i + 1, len(w_copy)):
            if collide(w_copy[i], w_copy[j]):
                rem.append(w_copy[j])
    for r in rem:
        if r in w: 
            w.remove(r)

    merge_black_holes(w)
    spawn_black_holes(w, white)
    
    # Multiverse bubble spawning with thermodynamic cost
    spawn_multiverse(bubbles, w)
    
    # Generate internal nodes in bubbles
    internal_nodes(bubbles, nodes)
    
    # Star formation inside bubbles
    star_formation(w, bubbles)
    
    # Universe feedback - work-based energy flow
    universe_feedback(bubbles, w)
    
    interact_universes(bubbles, w)
    space_decay(w)

    # Metrics collection
    total_work = sum([b.work for b in w if hasattr(b, "work")])
    st.session_state.work_history.append(total_work)
    
    bh_count = len([b for b in w if isinstance(b, BlackHole)])
    st.session_state.bh_count.append(bh_count)
    st.session_state.bh_history.append(bh_count)
    
    work_densities = [local_work_density(b, w) for b in w if hasattr(b, "work")]
    max_density = max(work_densities) if work_densities else 0
    st.session_state.work_density_max.append(max_density)
    
    # Element counts
    elements = {}
    for b in w:
        if hasattr(b, "el"):
            elements[b.el] = elements.get(b.el, 0) + 1
    st.session_state.element_counts = elements
    
    # Performance metrics
    elapsed = time.time() - start_time
    st.session_state.performance_metrics[st.session_state.iteration] = {
        "time": elapsed,
        "particles": len(w),
        "bh_count": bh_count,
        "work_density": max_density
    }
    
    st.session_state.iteration += 1
    
    # Check auto-stop conditions
    if batch_mode and auto_stop:
        if stop_condition == "Max work density reached" and max_density >= stop_threshold:
            st.session_state.stop_flag = True
        elif stop_condition == "Black holes > N" and bh_count >= stop_threshold:
            st.session_state.stop_flag = True
        elif stop_condition == "Bubbles spawned" and len(bubbles) >= stop_threshold:
            st.session_state.stop_flag = True

# ============== MAIN LAYOUT ==============

# Helper function to render state tree
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

# Helper to display metrics
def display_metrics():
    col1, col2, col3, col4, col5 = st.columns(5)
    
    bh = len([b for b in st.session_state.world if isinstance(b, BlackHole)])
    nu = st.session_state.element_counts.get("nu", 0)
    h_atoms = st.session_state.element_counts.get("H", 0)
    bubbles_count = len(st.session_state.bubbles)
    max_work_density = max(st.session_state.work_density_max) if st.session_state.work_density_max else 0
    
    col1.metric("Iteration", st.session_state.iteration)
    col2.metric("Black Holes", bh)
    col3.metric("Neutrinos", nu)
    col4.metric("Hydrogen", h_atoms)
    col5.metric("Work Density", f"{max_work_density:.1f}")

# Helper to display graphs
def display_graphs():
    if len(st.session_state.work_history) > 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("WTF Universe Dynamics Analysis", fontsize=14, fontweight='bold')
        
        ax1.plot(st.session_state.work_history, linewidth=2, color="cyan")
        ax1.set_ylabel("Total Work", color="cyan")
        ax1.tick_params(axis='y', labelcolor='cyan')
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Energy Accumulation")
        
        ax2.plot(st.session_state.bh_count, linewidth=2, color="magenta")
        ax2.set_ylabel("BH Count", color="magenta")
        ax2.tick_params(axis='y', labelcolor='magenta')
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Singularity Formation")
        
        ax3.plot(st.session_state.work_density_max, linewidth=2, color="lime")
        ax3.axhline(y=CRITICAL_WORK_DENSITY, color='red', linestyle='--', linewidth=2, label=f"Critical: {CRITICAL_WORK_DENSITY}")
        ax3.set_ylabel("Work Density (ρW)", color="lime")
        ax3.tick_params(axis='y', labelcolor='lime')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Singularity Threshold")
        
        elements = st.session_state.element_counts
        if elements:
            elem_names = list(elements.keys())
            elem_counts = list(elements.values())
            colors_map = {
                "nu": "#9933ff", "e": "#ff66ff", "p": "#ff3333",
                "n": "#666666", "H": "#4444ff", "He": "#44ffff",
                "C": "#00ff00", "O": "#ffaa00", "Fe": "#ff0000", "Ni": "#aa00ff"
            }
            bar_colors = [colors_map.get(e, "#888888") for e in elem_names]
            ax4.bar(elem_names, elem_counts, color=bar_colors)
            ax4.set_ylabel("Count")
            ax4.set_title("Particle Distribution")
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# Helper to display 3D visualization
def display_3d():
    st.markdown("### 🔮 3D Visualization")
    
    data = []
    for b in st.session_state.world:
        if isinstance(b, WhiteHole):
            data.append({"x": 0, "y": 0, "z": 0, "r": 1.0, "kind": "white", "el": "white"})
        else:
            data.append({
                "x": float(b.pos[0]), "y": float(b.pos[1]), "z": float(b.pos[2]),
                "r": b.radius, "kind": b.kind, "el": b.el
            })
    
    for b in st.session_state.bubbles:
        data.append({
            "x": float(b.center[0]), "y": float(b.center[1]), "z": float(b.center[2]),
            "r": b.radius, "kind": "bubble", "el": "bubble"
        })
    
    for n in st.session_state.nodes:
        data.append({
            "x": float(n.pos[0]), "y": float(n.pos[1]), "z": float(n.pos[2]),
            "r": 0.5, "kind": "node", "el": "node"
        })
    
    html = open("components/wtf_3d.html").read()
    html = html.replace("__DATA__", json.dumps(data))
    st.components.v1.html(html, height=650)

# Helper to display state tree
def display_tree():
    st.markdown("### 🌳 Simulation State Tree")
    
    elements_dict = {f"{k}: {v}" for k, v in sorted(st.session_state.element_counts.items())}
    
    tree_data = {
        "🌌 Universe": {
            f"Iteration: {st.session_state.iteration}": None,
            "🤍 White Hole": {
                f"Mass: {st.session_state.white.mass:.0f}": None,
                f"Energy: {st.session_state.white.energy:.1f}": None,
            },
            f"⚫ Black Holes ({len([b for b in st.session_state.world if isinstance(b, BlackHole)])})": {
                f"Total Mass: {sum([b.mass for b in st.session_state.world if isinstance(b, BlackHole)]):.0f}": None,
            },
            "⚛️ Particles": {
                f"Total: {len(st.session_state.world) - 1}": None,
                "Elements": elements_dict,
            },
            "🫧 Universe Bubbles": {
                f"Count: {len(st.session_state.bubbles)}": None
            },
            "📊 System State": {
                f"Total Work: {sum([b.work for b in st.session_state.world if hasattr(b, 'work')]):.1f}": None,
                f"Max Work Density: {max(st.session_state.work_density_max) if st.session_state.work_density_max else 0:.1f}": None,
            }
        }
    }
    
    st.code(render_tree(tree_data), language="text")

# ============== MODE SELECTION ==============
if live_mode:
    st.sidebar.info("🔴 Live mode active - continuous updates")
    
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
                status_text.text(f"Batch progress: {i + 1}/{batch_size}")
                
                if st.session_state.stop_flag:
                    st.warning(f"⏹ Auto-stopped at iteration {st.session_state.iteration}")
                    st.session_state.stop_flag = False
                    break
            
            st.success("✅ Batch complete!")
    
    with col2:
        if st.button("📊 Analyze Batch"):
            st.markdown("### 📈 Batch Performance Analysis")
            
            if st.session_state.performance_metrics:
                iters = sorted(st.session_state.performance_metrics.keys())
                times = [st.session_state.performance_metrics[i]["time"] for i in iters]
                particles = [st.session_state.performance_metrics[i]["particles"] for i in iters]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(iters, times, marker='o', color='orange')
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Time per step (s)")
                ax1.set_title("Performance Timeline")
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(iters, particles, marker='s', color='cyan')
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Particle Count")
                ax2.set_title("System Size Evolution")
                ax2.grid(True, alpha=0.3)
                
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
    
    if show_history:
        display_graphs()
    
    if show_tree:
        st.markdown("---")
        display_tree()
    
    if show_3d:
        st.markdown("---")
        display_3d()

else:
    # Standard mode (non-live, non-batch)
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
    
    if show_history:
        display_graphs()
    
    if show_tree:
        st.markdown("---")
        display_tree()
    
    if show_3d:
        st.markdown("---")
        display_3d()