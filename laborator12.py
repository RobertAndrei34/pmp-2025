import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

df = pd.read_csv("date_promovare_examen.csv")

X = df[["Ore_Studiu", "Ore_Somn"]].to_numpy()
y = df["Promovare"].to_numpy()

values, counts = np.unique(y, return_counts=True)
balance = dict(zip(values, counts))
print(balance)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
Xn = (X - X_mean) / X_std

with pm.Model() as model:
    w = pm.Normal("w", mu=0, sigma=5, shape=2)
    b = pm.Normal("b", mu=0, sigma=5)
    logits = pm.math.dot(Xn, w) + b
    p = pm.Deterministic("p", pm.math.sigmoid(logits))
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
    trace = pm.sample(500, tune=500, chains=1, cores=1, target_accept=0.9, progressbar=True)


summary = az.summary(trace, var_names=["w", "b"])
print(summary)

w_mean = trace.posterior["w"].mean(dim=("chain", "draw")).values
b_mean = trace.posterior["b"].mean(dim=("chain", "draw")).values

x1 = np.linspace(Xn[:, 0].min(), Xn[:, 0].max(), 100)
x2_boundary = -(w_mean[0] * x1 + b_mean) / w_mean[1]
x2_boundary_real = x2_boundary * X_std[1] + X_mean[1]
x1_real = x1 * X_std[0] + X_mean[0]

print(np.mean(x1_real), np.mean(x2_boundary_real))

importance = np.abs(w_mean)
print(importance)
print("Ore_Studiu" if importance[0] > importance[1] else "Ore_Somn")
