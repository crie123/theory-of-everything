import numpy as np
import matplotlib.pyplot as plt

def simulate_memory_degradation():
    time = np.linspace(0, 100, 200)
    normal_decay = np.exp(-time / 60)
    accelerated_decay = np.exp(-time / 20)
    plt.plot(time, normal_decay, label='Linear degradation')
    plt.plot(time, accelerated_decay, label='WTF accelerated')
    plt.title("Memory degradation under field stress")
    plt.xlabel("Time")
    plt.ylabel("State integrity")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/Figure_1.png")
    plt.close()

def simulate_instant_collapse():
    time = np.linspace(0, 10, 100)
    collapse = np.piecewise(time, [time < 5, time >= 5], [1, 0])
    plt.plot(time, collapse, color='red')
    plt.title("Instantaneous decay after threshold")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.grid(True)
    plt.savefig("figures/Figure_2.png")
    plt.close()

def simulate_tunneling():
    width = np.linspace(1, 10, 100)
    prob_qm = np.exp(-width)
    prob_wtf = np.exp(-width / 0.7)
    plt.plot(width, prob_qm, label='Quantum Mechanics')
    plt.plot(width, prob_wtf, label='WTF Model', linestyle='--')
    plt.axvline(4, color='gray', linestyle=':', label='~4nm threshold')
    plt.title("Tunneling probability vs barrier width")
    plt.xlabel("Barrier width (nm)")
    plt.ylabel("Tunneling probability")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/tunneling_vs_width_4nm.png")
    plt.close()

def simulate_gravi_map():
    X, Y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    field = np.exp(-(X**2 + Y**2) / 6)
    plt.imshow(field, extent=(-5, 5, -5, 5), cmap='viridis')
    plt.colorbar(label='Field strength')
    plt.title("Simulated field map (warp curvature)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("figures/gravi_map_final.png")
    plt.close()

if __name__ == "__main__":
    simulate_memory_degradation()
    simulate_instant_collapse()
    simulate_tunneling()
    simulate_gravi_map()
