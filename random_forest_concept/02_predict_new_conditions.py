"""
02_predict_new_conditions.py
============================
Predict the yield for ONE new operating condition and compare:

    Physics model  (the equation that generated the data)
    Random Forest  (a model that only saw the noisy data)

Simplified educational model - NOT a validated chemical reactor model.

Run 01_train_and_analyze.py first (it creates data/chemical_process.csv).
Then edit the three numbers below and run:  python 02_predict_new_conditions.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# >>> EDIT THESE (allowed ranges: 300-450 K, 1-60 min, 0.5-2.0) <<<
# --------------------------------------------------------------------------
temperature = 380        # K
reaction_time = 30       # minutes
catalyst_factor = 1.5    # activity factor

SHOW_PLOTS = True

# Same constants as in 01_train_and_analyze.py (must match!)
A_PRE = 2.2e4            # [1/min]
EA = 40_000.0            # [J/mol]
R_GAS = 8.314            # [J/(mol K)]

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "chemical_process.csv"
PLOT_DIR = BASE_DIR / "plots"


def physics_yield(temperature_K, time_min, catalyst):
    """Y_physics = 100 * (1 - exp(-k_eff * t)),  k_eff = A*exp(-Ea/(R*T)) * C"""
    k = A_PRE * np.exp(-EA / (R_GAS * temperature_K))
    return 100.0 * (1.0 - np.exp(-k * catalyst * time_min))


# --- Load data and train the same Random Forest ---------------------------
if not DATA_PATH.exists():
    raise SystemExit("data/chemical_process.csv not found - run 01_train_and_analyze.py first.")

df = pd.read_csv(DATA_PATH)
features = ["temperature_K", "reaction_time_min", "catalyst_factor"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["actual_yield"], test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predict the new condition --------------------------------------------
new_point = pd.DataFrame(
    [[temperature, reaction_time, catalyst_factor]], columns=features
)
y_physics = float(physics_yield(temperature, reaction_time, catalyst_factor))
y_rf = float(model.predict(new_point)[0])
error = y_rf - y_physics          # Error = Y_RF - Y_physics

print(f"""
--------------------------------
NEW CHEMICAL PROCESS
--------------------------------

Temperature       : {temperature} K
Reaction Time     : {reaction_time} min
Catalyst Factor   : {catalyst_factor}

Physics Yield     : {y_physics:.2f} %
Random Forest     : {y_rf:.2f} %

Difference        : {error:+.2f} %   (RF - physics)
--------------------------------
""")

# --- Bar chart -------------------------------------------------------------
PLOT_DIR.mkdir(exist_ok=True)
plt.figure(figsize=(6, 5))
bars = plt.bar(["Physics Model", "Random Forest"], [y_physics, y_rf],
               color=["tab:blue", "tab:green"])
for b, v in zip(bars, [y_physics, y_rf]):
    plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f} %", ha="center")
plt.ylim(0, 110)
plt.ylabel("Yield (%)")
plt.title(f"New condition: T={temperature} K, t={reaction_time} min, C={catalyst_factor}\n"
          f"Error (RF - physics) = {error:+.2f} %")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "new_condition_comparison.png", dpi=150)
print("saved plots/new_condition_comparison.png")

if SHOW_PLOTS:
    plt.show()
