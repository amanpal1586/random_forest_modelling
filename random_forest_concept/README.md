# Random Forest + Simplified Chemical Process (Educational)

> **Simplified educational model - NOT a validated chemical reactor model.**

## Run order
```bash
pip install numpy pandas matplotlib scikit-learn
python 01_train_and_analyze.py       # creates data/, plots/, trains RF, 8 plots + analysis
python 02_predict_new_conditions.py  # edit T, t, C at top of file
python 03_visualize_physics.py       # physics only, no ML
```

## Data flow
```
Physics model (Arrhenius + first-order kinetics)
      -> synthetic process behaviour
      -> actual/noisy observations (physics + Gaussian noise)
      -> Random Forest learns from observations
      -> predicted yield
      -> compare with physics / actual yield
```

## Physics
- k(T) = A exp(-Ea / RT)
- k_eff = k(T) * C
- Y = 100 (1 - exp(-k_eff * t))
- Y_actual = Y + N(0, sigma^2), clipped to [0, 100]

The Random Forest **does not know** the Arrhenius equation. It only sees
(temperature, time, catalyst) -> yield and learns the mapping from data.
It approximates the relationship; it does not discover or prove physical laws.

## Random Forest in one paragraph
One decision tree learns rules like "is T < 350 K?" -> lower/higher yield.
Each tree in the forest sees a different random sample of the data and learns
different rules. For regression the forest averages them: Y_hat = (1/N) sum T_i(X).
