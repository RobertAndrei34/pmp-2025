import pandas as pd
import numpy as np
from em_mixture import em_mixture_regression
from waic import compute_waic

# ---------- Load data ----------
data = pd.read_csv("date_colesterol.csv")

x = data.iloc[:, 0].values   # hours of exercise
y = data.iloc[:, 1].values   # cholesterol

results = {}

# ---------- Fit models ----------
for K in [3, 4, 5]:
    print(f"\nFitting model with K = {K}")
    model = em_mixture_regression(x, y, K)
    waic = compute_waic(x, y, model)

    results[K] = (model, waic)

    print("Weights:", np.round(model["weights"], 3))
    print("Sigmas :", np.round(model["sigmas"], 2))
    print("WAIC   :", round(waic, 2))

    for k, beta in enumerate(model["betas"]):
        print(f"Subgroup {k+1}:")
        print(f"  a = {beta[0]:.2f}, b = {beta[1]:.2f}, gamma = {beta[2]:.3f}")

# ---------- Model selection ----------
best_K = min(results, key=lambda k: results[k][1])
print("\n===================================")
print(f"Best model according to WAIC: K = {best_K}")
print("===================================")
