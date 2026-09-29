"""
01_train_and_analyze.py
=======================
Random Forest learning a SIMPLIFIED, EDUCATIONAL chemical process model.

!! IMPORTANT !!
This is a simplified educational model, NOT a validated chemical reactor
model. The constants below are chosen only to give nicely varying yields.

The big picture
---------------
    Physics model (Arrhenius + first-order kinetics)
          |
          v
    Generates synthetic process behaviour
          |
          v
    Actual / synthetic observations  (physics + Gaussian noise)
          |
          v
    Random Forest learns from the observations
          |
          v
    Predicted yield
          |
          v
    Compare prediction with physics / actual yield

The Random Forest does NOT know the Arrhenius equation. It only sees
    (Temperature, Reaction Time, Catalyst Factor)  ->  Yield
and learns the nonlinear mapping purely from data.

The physics
-----------
    k(T)        = A * exp(-Ea / (R*T))          (Arrhenius)
    k_effective = k(T) * C                      (C = catalyst/activity factor)
    X           = 1 - exp(-k_effective * t)     (first-order conversion)
    Y_physics   = 100 * X                       (percent yield)
    Y_actual    = Y_physics + N(0, sigma^2)     (measurement noise), clipped to 0..100

How a Random Forest works (elementary level)
--------------------------------------------
A single decision tree learns yes/no rules, e.g.

            Is temperature < 350 K?
                 /          \\
               Yes           No
                |             |
          lower yield     higher yield

Another tree, trained on a slightly different random sample of the data,
learns different rules. A Random Forest builds many trees and averages
their predictions:

    Y_hat = (1/N) * sum_{i=1..N} T_i(X)

where T_i(X) is the prediction of tree i.

Run:  python 01_train_and_analyze.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
SHOW_PLOTS = True          # set False to only save the PNG files
N_SAMPLES = 800            # 500-1000 is fine
NOISE_STD = 2.0            # measurement noise, in yield percentage points
SEED = 42

# Simplified kinetic constants (educational values, not real chemistry)
A_PRE = 2.2e4              # pre-exponential factor  [1/min]
EA = 40_000.0              # activation energy       [J/mol]
R_GAS = 8.314              # gas constant            [J/(mol K)]

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "chemical_process.csv"
PLOT_DIR = BASE_DIR / "plots"


def physics_yield(temperature_K, time_min, catalyst):
    """Y_physics = 100 * (1 - exp(-k_eff * t)),  k_eff = A*exp(-Ea/(R*T)) * C"""
    k = A_PRE * np.exp(-EA / (R_GAS * temperature_K))
    k_eff = k * catalyst
    return 100.0 * (1.0 - np.exp(-k_eff * time_min))


def finish(fig_name):
    """Save the current figure (and optionally show it later)."""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / fig_name, dpi=150)
    print(f"  saved plots/{fig_name}")


# ==========================================================================
# Step 1 - Generate data from the physics model + noise
# ==========================================================================
print("STEP 1: generating synthetic data ...")
rng = np.random.default_rng(SEED)

temperature = rng.uniform(300, 450, N_SAMPLES)       # K
reaction_time = rng.uniform(1, 60, N_SAMPLES)        # minutes
catalyst = rng.uniform(0.5, 2.0, N_SAMPLES)          # activity factor

y_physics = physics_yield(temperature, reaction_time, catalyst)
noise = rng.normal(0.0, NOISE_STD, N_SAMPLES)
y_actual = np.clip(y_physics + noise, 0.0, 100.0)

df = pd.DataFrame(
    {
        "temperature_K": temperature,
        "reaction_time_min": reaction_time,
        "catalyst_factor": catalyst,
        "physics_yield": y_physics,
        "actual_yield": y_actual,
    }
)
DATA_PATH.parent.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
df.to_csv(DATA_PATH, index=False)
print(f"  saved {DATA_PATH.relative_to(BASE_DIR)}  ({len(df)} rows)")

# ==========================================================================
# Step 2 - Train the Random Forest
# ==========================================================================
print("\nSTEP 2: training Random Forest ...")
features = ["temperature_K", "reaction_time_min", "catalyst_factor"]
X = df[features]
y = df["actual_yield"]

# Keep the physics yield aligned with the same split (used in plots 4 and 5)
X_train, X_test, y_train, y_test, phys_train, phys_test = train_test_split(
    X, y, df["physics_yield"], test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"  trained on {len(X_train)} samples, testing on {len(X_test)} samples")

# ==========================================================================
# Step 3 - Predictions and metrics
# ==========================================================================
print("\nSTEP 3: evaluating on unseen test data ...")
y_pred = model.predict(X_test)
residual = y_test.values - y_pred                    # actual - predicted

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n  Test-set metrics")
print("  ----------------")
print(f"  MAE  : {mae:.3f} %  (average absolute error)")
print(f"  RMSE : {rmse:.3f} %  (penalises large errors more)")
print(f"  R2   : {r2:.4f}    (1.0 = perfect)")
print(f"  (noise put into the data: sigma = {NOISE_STD} %. RMSE above this means the")
print("   forest also has some approximation error from learning with only 640 points)")

# ==========================================================================
# Step 4 - Plots
# ==========================================================================
print("\nSTEP 4: making plots ...")
metrics_text = f"R² = {r2:.4f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"

# ---- Plot 1: Actual vs Predicted -----------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
plt.plot([0, 100], [0, 100], "r--", label="Perfect prediction (y = x)")
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, va="top",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
plt.xlabel("Actual yield (%)")
plt.ylabel("Predicted yield (%)")
plt.title("Plot 1: Actual vs Predicted Yield\n(closer to the red line = better model)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
finish("plot1_actual_vs_predicted.png")

# ---- Plot 2: Prediction error --------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, residual, alpha=0.6, edgecolor="k", linewidth=0.3)
plt.axhline(0, color="r", linestyle="--", label="Zero error")
plt.xlabel("Actual yield (%)")
plt.ylabel("Prediction error = actual - predicted (%)")
plt.title("Plot 2: Prediction Error (Residuals)\n"
          "Random scatter around zero = little systematic error")
plt.legend()
plt.grid(alpha=0.3)
finish("plot2_prediction_error.png")

# ---- Plot 3: Actual vs Predicted for 100 test samples --------------------
n_line = min(100, len(y_test))
plt.figure(figsize=(11, 4.5))
plt.plot(range(n_line), y_test.values[:n_line], "o-", label="Actual yield", markersize=4)
plt.plot(range(n_line), y_pred[:n_line], "s--", label="Random Forest prediction", markersize=4)
plt.xlabel("Test sample number")
plt.ylabel("Yield (%)")
plt.title(f"Plot 3: Actual vs Predicted for {n_line} Test Samples")
plt.legend()
plt.grid(alpha=0.3)
finish("plot3_sample_comparison.png")

# ---- Plot 4: Physics vs Actual -------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(phys_test, y_test, alpha=0.6, edgecolor="k", linewidth=0.3)
plt.plot([0, 100], [0, 100], "r--", label="y = x")
plt.xlabel("Physics-based yield (%)")
plt.ylabel("Actual (noisy) yield (%)")
plt.title("Plot 4: Physics vs Actual\n(spread around the line = measurement noise)")
plt.legend()
plt.grid(alpha=0.3)
finish("plot4_physics_vs_actual.png")

# ---- Plot 5: Physics vs Random Forest ------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(phys_test, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3, color="tab:green")
plt.plot([0, 100], [0, 100], "r--", label="y = x")
plt.xlabel("Physics-based yield (%)")
plt.ylabel("Random Forest prediction (%)")
plt.title("Plot 5: Physics vs Random Forest\n(did the model learn the physics-generated relationship?)")
plt.legend()
plt.grid(alpha=0.3)
finish("plot5_physics_vs_rf.png")

# ---- Plot 6: Temperature vs Yield (colour = time) ------------------------
plt.figure(figsize=(7.5, 5))
sc = plt.scatter(df["temperature_K"], df["actual_yield"], c=df["reaction_time_min"],
                 cmap="viridis", alpha=0.7, s=18)
plt.colorbar(sc, label="Reaction time (min)")
plt.xlabel("Temperature (K)")
plt.ylabel("Actual yield (%)")
plt.title("Plot 6: Temperature vs Yield (colour = reaction time)")
plt.grid(alpha=0.3)
finish("plot6_temperature_vs_yield.png")

# ---- Plot 7: Reaction time vs Yield (colour = temperature) ---------------
plt.figure(figsize=(7.5, 5))
sc = plt.scatter(df["reaction_time_min"], df["actual_yield"], c=df["temperature_K"],
                 cmap="plasma", alpha=0.7, s=18)
plt.colorbar(sc, label="Temperature (K)")
plt.xlabel("Reaction time (min)")
plt.ylabel("Actual yield (%)")
plt.title("Plot 7: Reaction Time vs Yield (colour = temperature)")
plt.grid(alpha=0.3)
finish("plot7_time_vs_yield.png")

# ---- Plot 8: Feature importance ------------------------------------------
labels = ["Temperature", "Reaction Time", "Catalyst Factor"]
importances = model.feature_importances_
order = np.argsort(importances)
plt.figure(figsize=(7, 4))
plt.barh(np.array(labels)[order], importances[order], color="tab:orange")
for i, v in enumerate(importances[order]):
    plt.text(v + 0.005, i, f"{v:.3f}", va="center")
plt.xlabel("Feature importance (sums to 1)")
plt.title("Plot 8: Random Forest Feature Importance\n"
          "How useful each input was to THIS trained model - not causal importance")
plt.xlim(0, max(importances) * 1.15)
finish("plot8_feature_importance.png")

# ==========================================================================
# Analysis
# ==========================================================================
top = labels[int(np.argmax(importances))]
print(f"""
MODEL ANALYSIS
==============

Random Forest Performance (test set)
R²   : {r2:.4f}
RMSE : {rmse:.3f} %
MAE  : {mae:.3f} %

The Random Forest learned an approximation of the nonlinear
relationship between temperature, reaction time and catalyst
activity, using only the noisy observations.

The actual-vs-predicted plot shows how closely the model
predictions follow the generated process data.

The residual plot shows whether the prediction errors are
centred around zero (mean residual = {residual.mean():.3f} %).

Feature importance indicates which process variables were most
useful to the model. Here: {top} ranked highest. This describes
the trained model, not a causal statement about the chemistry.

The physics-based model provides the underlying relationship
used to generate the synthetic data, while the Random Forest
acts as a data-driven approximation of this relationship.
It does NOT discover or prove the physical laws.
""")

if SHOW_PLOTS:
    plt.show()
