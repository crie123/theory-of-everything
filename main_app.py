import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Framework Simulator", layout="wide")

# Degradation Simulation
def simulate_degradation(D0, lam, t_max):
    t = np.linspace(0, t_max, 1000)
    D = D0 * np.exp(-lam * t)
    fig, ax = plt.subplots()
    ax.plot(t, D)
    ax.set_xlabel("Time")
    ax.set_ylabel("Integrity D(t)")
    ax.set_title("Degradation Curve")
    ax.grid(True)
    return fig

# Work per Graviton Simulation
def simulate_graviton_work(E_total, min_g, max_g):
    N = np.logspace(np.log10(min_g), np.log10(max_g), 200)
    W = E_total / N
    fig, ax = plt.subplots()
    ax.loglog(N, W)
    ax.set_xlabel("Number of Gravitons")
    ax.set_ylabel("Work per Graviton")
    ax.set_title("Graviton Work Distribution")
    ax.grid(True)
    return fig

# Radar Map Generation
def generate_radar_map(size, peak_coords, amp, sigma):
    x = np.linspace(0, size - 1, size)
    y = np.linspace(0, size - 1, size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)
    for x0, y0 in peak_coords:
        Z += amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
    fig, ax = plt.subplots()
    cp = ax.contourf(X, Y, Z, levels=50, cmap='plasma')
    fig.colorbar(cp, ax=ax, label="Radar Intensity")
    ax.set_title("Framework Radar Map")
    return fig

def stable_isotopes_chart(Z_max=100):
    Z = np.arange(1, Z_max+1)
    N = [int(z*1.5) for z in Z]  # approximate stable isotopes
    fig, ax = plt.subplots()
    ax.plot(Z, N, 'o', label="Approx. stable isotopes")
    ax.set_xlabel("Protons (Z)")
    ax.set_ylabel("Neutrons (N)")
    ax.set_title("Stable Isotope Line")
    ax.grid(True)
    ax.legend()
    return fig

def spacetime_anomaly_map(size=100):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    anomaly = np.sin(10 * np.sqrt(X**2 + Y**2)) * np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.05)
    fig, ax = plt.subplots()
    im = ax.imshow(anomaly, cmap="coolwarm", origin="lower")
    fig.colorbar(im, ax=ax, label="Spacetime curvature")
    ax.set_title("Spacetime Anomaly Map")
    return fig

def jump_field_map(size=100, field_strength=5):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    field = np.exp(-(X**2 + Y**2) * field_strength)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, field, levels=40, cmap='viridis')
    fig.colorbar(cs, ax=ax, label="Jump Field Intensity")
    ax.set_title("Jump Field Engine Configuration")
    return fig

def local_energy_analysis(size=100):
    x = np.linspace(0, 10, size)
    y = np.sin(x) + 0.3 * np.random.randn(size)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Energy variation")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Energy Flux")
    ax.set_title("Energy Analysis in Local Space")
    ax.legend()
    return fig

# New Tabs
st.header("🧠 Extended Simulator Modules")
tabs = st.tabs(["Stable Isotopes", "Spacetime Map", "Jump Field", "Energy Profile"])

with tabs[0]:
    Zmax = st.slider("Max Proton Number (Z)", 10, 120, 100)
    st.pyplot(stable_isotopes_chart(Zmax))

with tabs[1]:
    size1 = st.slider("Map Resolution", 50, 200, 100, key="a1")
    st.pyplot(spacetime_anomaly_map(size1))

with tabs[2]:
    size2 = st.slider("Jump Field Grid", 50, 200, 100, key="a2")
    strength = st.slider("Field Strength", 1, 20, 5)
    st.pyplot(jump_field_map(size2, strength))

with tabs[3]:
    size3 = st.slider("Resolution", 50, 300, 100, key="a3")
    st.pyplot(local_energy_analysis(size3))

# Streamlit Interface
st.title("🌌 Framework Physics Simulator")
st.sidebar.header("Parameters")

with st.sidebar.expander("Degradation"):
    D0 = st.slider("Initial Integrity (D₀)", 0.1, 2.0, 1.0)
    lam = st.slider("Degradation Rate (λ)", 0.001, 1.0, 0.1)
    t_max = st.slider("Max Time", 10, 1000, 100)

with st.sidebar.expander("Graviton Work"):
    E_total = st.number_input("Total Energy (J)", 0.01, 1000.0, 1.0)
    min_g = st.number_input("Min Gravitons", 1, 100000, 10)
    max_g = st.number_input("Max Gravitons", 10, 1000000, 10000)

with st.sidebar.expander("Radar Map"):
    size = st.slider("Grid Size", 10, 200, 100)
    amp = st.slider("Peak Amplitude", 0.1, 10.0, 1.0)
    sigma = st.slider("Spread σ", 1, 50, 10)
    coords = st.text_input("Peak Coords [(x1,y1),(x2,y2)]", "(25,25),(75,70)")

# Run simulations
tab1, tab2, tab3 = st.tabs(["Degradation", "Graviton Work", "Radar Map"])

with tab1:
    fig = simulate_degradation(D0, lam, t_max)
    st.pyplot(fig)

with tab2:
    fig2 = simulate_graviton_work(E_total, min_g, max_g)
    st.pyplot(fig2)

with tab3:
    try:
        peak_coords = eval(coords)
        fig3 = generate_radar_map(size, peak_coords, amp, sigma)
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"Invalid coords format: {e}")
