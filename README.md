# WTF Engine - Work Tensor Framework

## 🌌 Description

**WTF (Work Tensor Framework)** is a physics engine that implements an alternative theory of gravity and quantum mechanics based on the concept of "work" as a fundamental parameter of the Universe.

The engine models the evolution of particles, stars, black holes, and white holes in a dynamic system governed by the distribution of work in space.

---

## 🔧 Core Components

### Engine Core (`wtf_model.py`)
- **Body** — base class for all material objects (particles, stars)
- **BlackHole** — black hole class with collapse and explosion mechanics
- **WhiteHole** — inversion nodes that eject matter and destroy black holes

### Physical Processes
- **Gravity** — classical interaction between bodies
- **Work Accumulation** — tracking the energetic state of each particle
- **Nuclear Fusion** — transformation of elements (ν → e → p → H → He → C → O → Fe)
- **Phase Transitions** — transition of matter to frozen state at high coherence
- **Black Hole Formation** — criterion based on work density and velocity gradients

### Interfaces
- **main_app.py** — Streamlit application for simulation visualization
- **wtf_app.py** — additional application with extended features

---

## 📊 Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `G` | 0.05 | Gravitational constant |
| `H` | 0.002 | Universe expansion coefficient |
| `CRITICAL_WORK_DENSITY` | 120.0 | Threshold work density for singularity formation |
| `WHITE_RADIUS` | 3.0 | Radius of white hole influence zone |
| `BUBBLE_FORMATION_COST` | 5000 | Energy for creating new space region |

---

## 🚀 Fundamental Concepts

### Work
Instead of traditional energy, the system tracks **work** as the accumulated effect of interactions. Each body has:
- `energy` — current energetic charge
- `work` — accumulated work from gravitational accelerations
- `coherence` — level of phase coherence

### Singularity
Black holes form not simply from mass, but from **local concentrations of work**:
```
Condition: ρ_work > CRITICAL_WORK_DENSITY AND ∇ρ_work > WORK_GRADIENT_MIN
```

### Inversion (White Holes)
White holes serve as **space cleaners**:
- Explode black holes in the vicinity
- Return matter to elementary state (neutrinos)
- Eject energy into space

---

## 📁 Project Structure

```
.
├── wtf_model.py              # Engine core
├── main_app.py               # Streamlit application
├── wtf_app.py                # Alternative application
├── exp.py                    # Experimental code
├── components/
│   └── wtf_3d.html          # 3D visualization
├── desc/
│   ├── gravity.pdf           # Gravity theory
│   ├── field.pdf             # Field theory
│   └── equations.pdf         # Mathematical apparatus
└── desc/wtf/
    └── wtf_theory_arxiv.tex  # Full theory in TeX
```

---

## 🎯 Running

### Main Simulation
```bash
python exp.py
```

### Interactive Application (Streamlit)
```bash
streamlit run main_app.py
```

---

## 🧪 Engine Features

- ✅ N-body gravitational simulation
- ✅ Dynamic formation and destruction of black holes
- ✅ Nuclear fusion and element transmutation
- ✅ Matter phase transitions
- ✅ White holes as inversion nodes
- ✅ Universe thermodynamic reservoir
- ✅ Real-time visualization

---

## 📚 Sources

Full theory is described in:
- `desc/wtf/wtf_theory_arxiv.tex` — main document
- `desc/gravity.pdf` — gravitational theory extension
- `desc/field.pdf` — work field theory

---

## ⚖️ License

See `LICENSE.txt`
