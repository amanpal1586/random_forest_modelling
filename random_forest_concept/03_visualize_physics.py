"""
03_visualize_physics.py
=======================
Understand the underlying physics/chemistry relationship ONLY.
No Random Forest (no machine learning) is used in this file.

Simplified educational model - NOT a validated chemical reactor model.

    k(T)        = A * exp(-Ea / (R*T))
    k_effective = k(T) * C
    Y(t)        = 100 * (1 - exp(-k_effective * t))

Run:  python 03_visualize_physics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SHOW_PLOTS = True

# Same constants as in the other two files
A_PRE = 2.2e4            # [1/min]
EA = 40_000.0            # [J/mol]
R_GAS = 8.314            # [J/(mol K)]

PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def arrhenius_k(T):
    return A_PRE * np.exp(-EA / (R_GAS * T))


def physics_yield(T, t, catalyst=1.0):
    return 100.0 * (1.0 - np.exp(-arrhenius_k(T) * catalyst * t))


# ---- Plot 1: k vs T --------------------------------------------------------
T = np.linspace(300, 450, 300)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(T, arrhenius_k(T), color="tab:red", lw=2)
ax[0].set_xlabel("Temperature (K)")
ax[0].set_ylabel("Rate constant k (1/min)")
ax[0].set_title("Linear scale")
ax[0].grid(alpha=0.3)
ax[1].semilogy(T, arrhenius_k(T), color="tab:red", lw=2)
ax[1].set_xlabel("Temperature (K)")
ax[1].set_ylabel("Rate constant k (1/min, log scale)")
ax[1].set_title("Log scale")
ax[1].grid(alpha=0.3, which="both")
fig.suptitle("Plot 1: Arrhenius rate constant vs temperature\n"
             "k grows exponentially: a higher T means a larger fraction of molecules "
             "have energy above Ea")
plt.tight_layout()
plt.savefig(PLOT_DIR / "physics1_k_vs_T.png", dpi=150)

# ---- Plot 2: Yield vs time at several temperatures --------------------------
t = np.linspace(0, 60, 300)
plt.figure(figsize=(7.5, 5))
for T_fix in [300, 330, 360, 390, 420]:
    plt.plot(t, physics_yield(T_fix, t), lw=2, label=f"{T_fix} K")
plt.xlabel("Reaction time (min)")
plt.ylabel("Physics yield (%)")
plt.title("Plot 2: Yield vs reaction time (catalyst factor = 1)\n"
          "Higher temperature -> faster approach to 100 %")
plt.legend(title="Temperature")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "physics2_yield_vs_time.png", dpi=150)

# ---- Plot 3: 3D surface -----------------------------------------------------
T_grid, t_grid = np.meshgrid(np.linspace(300, 450, 60), np.linspace(1, 60, 60))
Z = physics_yield(T_grid, t_grid, catalyst=1.0)
fig = plt.figure(figsize=(8, 6))
ax3 = fig.add_subplot(111, projection="3d")
surf = ax3.plot_surface(T_grid, t_grid, Z, cmap="viridis", edgecolor="none", alpha=0.95)
ax3.set_xlabel("Temperature (K)")
ax3.set_ylabel("Reaction time (min)")
ax3.set_zlabel("Physics yield (%)")
ax3.set_title("Plot 3: Process response surface (catalyst factor = 1)")
fig.colorbar(surf, shrink=0.6, label="Yield (%)")
ax3.view_init(elev=28, azim=-125)
plt.tight_layout()
plt.savefig(PLOT_DIR / "physics3_surface.png", dpi=150)

# ---- Plot 4: Catalyst effect -----------------------------------------------
C = np.linspace(0.5, 2.0, 200)
plt.figure(figsize=(7.5, 5))
for T_fix in [330, 360, 390, 420]:
    plt.plot(C, physics_yield(T_fix, 30, C), lw=2, label=f"{T_fix} K")
plt.xlabel("Catalyst / activity factor")
plt.ylabel("Physics yield (%)")
plt.title("Plot 4: Catalyst effect (reaction time = 30 min)\n"
          "C multiplies k, so it acts like a 'temperature boost'")
plt.legend(title="Temperature")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "physics4_catalyst_effect.png", dpi=150)

print("Saved 4 physics plots to plots/")
print(f"""
PHYSICS ANALYSIS
================
k at 300 K : {arrhenius_k(300):.5f} 1/min
k at 450 K : {arrhenius_k(450):.5f} 1/min  (about {arrhenius_k(450)/arrhenius_k(300):.0f}x larger)

- The rate constant rises exponentially with temperature (Arrhenius).
- For a first-order reaction, yield rises quickly at first and then
  levels off towards 100 % as the reactant is used up.
- The catalyst factor scales k, so doubling C is equivalent to doubling
  the reaction time at the same temperature.
- This is a simplified educational model, not a validated reactor model.
""")

if SHOW_PLOTS:
    plt.show()
