import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import pymc as pm
import arviz as az

# 1) Load + EDA
df = pd.read_csv("bike_daily.csv")
df["season"] = df["season"].astype(str).str.strip().str.lower()

print(df.head())
print(df.shape)
print(df["season"].value_counts())

plt.figure()
plt.scatter(df["temp_c"], df["rentals"], s=10)
plt.xlabel("temp_c"); plt.ylabel("rentals"); plt.title("temp_c vs rentals")
plt.show()

plt.figure()
plt.scatter(df["humidity"], df["rentals"], s=10)
plt.xlabel("humidity"); plt.ylabel("rentals"); plt.title("humidity vs rentals")
plt.show()

plt.figure()
plt.scatter(df["wind_kph"], df["rentals"], s=10)
plt.xlabel("wind_kph"); plt.ylabel("rentals"); plt.title("wind_kph vs rentals")
plt.show()

# 2) Standardize continuous X + y
scX = StandardScaler()
X_cont = scX.fit_transform(df[["temp_c", "humidity", "wind_kph"]])
temp_z, hum_z, wind_z = X_cont.T

scY = StandardScaler()
y_z = scY.fit_transform(df[["rentals"]]).ravel()

temp2 = temp_z**2  # polynomial term

season_dum = pd.get_dummies(df["season"], drop_first=True)

X_lin = np.column_stack([temp_z, hum_z, wind_z, df["is_holiday"].values, season_dum.values])
names_lin = ["temp_z", "hum_z", "wind_z", "is_holiday"] + [f"season_{c}" for c in season_dum.columns]

X_poly = np.column_stack([temp_z, temp2, hum_z, wind_z, df["is_holiday"].values, season_dum.values])
names_poly = ["temp_z", "temp2", "hum_z", "wind_z", "is_holiday"] + [f"season_{c}" for c in season_dum.columns]

# 2b) Bayesian linear regression
with pm.Model() as m_lin:
    alpha = pm.Normal("alpha", 0, 1)
    beta  = pm.Normal("beta", 0, 1, shape=X_lin.shape[1])
    sigma = pm.HalfNormal("sigma", 1)

    mu = alpha + pm.math.dot(X_lin, beta)
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_z)

    idata_lin = pm.sample(2000, tune=2000, chains=2, target_accept=0.9, random_seed=123)
    idata_lin = pm.sample_posterior_predictive(idata_lin, extend_inferencedata=True, random_seed=123)

print(az.summary(idata_lin, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95))

# 2c) Bayesian polynomial regression
with pm.Model() as m_poly:
    alpha = pm.Normal("alpha", 0, 1)
    beta  = pm.Normal("beta", 0, 1, shape=X_poly.shape[1])
    sigma = pm.HalfNormal("sigma", 1)

    mu = alpha + pm.math.dot(X_poly, beta)
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_z)

    idata_poly = pm.sample(2000, tune=2000, chains=2, target_accept=0.9, random_seed=123)
    idata_poly = pm.sample_posterior_predictive(idata_poly, extend_inferencedata=True, random_seed=123)

print(az.summary(idata_poly, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95))

# 4a) Compare via WAIC or LOO
cmp = az.compare({"lin": idata_lin, "poly": idata_poly}, ic="loo", method="BB-pseudo-BMA", scale="deviance")
print(cmp)
az.plot_compare(cmp)
plt.show()

# 4b) PPC plot: predicted mean + HDI vs temp_c
temp_grid = np.linspace(temp_z.min(), temp_z.max(), 100)
hum0 = 0.0
wind0 = 0.0
holiday0 = 0.0
season0 = np.zeros(season_dum.shape[1])  # baseline season

Xg_lin = np.column_stack([temp_grid, np.full_like(temp_grid, hum0), np.full_like(temp_grid, wind0),
                          np.full_like(temp_grid, holiday0), np.tile(season0, (len(temp_grid), 1))])

posterior = idata_lin.posterior
alpha_s = posterior["alpha"].stack(s=("chain","draw")).values
beta_s  = posterior["beta"].stack(s=("chain","draw")).values
mu_s = alpha_s[:, None] + (beta_s.T @ Xg_lin.T)
hdi = az.hdi(mu_s, hdi_prob=0.95)

mu_mean = mu_s.mean(axis=0)

plt.figure()
plt.plot(temp_grid, mu_mean)
plt.fill_between(temp_grid, hdi[:,0], hdi[:,1], alpha=0.3)
plt.xlabel("temp_z (standardized)")
plt.ylabel("E[rentals_z] (posterior)")
plt.title("Posterior mean + 95% HDI vs temp")
plt.show()

# 5) Binary target: high demand
Q = np.percentile(df["rentals"], 75)
is_high = (df["rentals"] >= Q).astype(int).values
print("Q(75%) =", Q, "| rate_high =", is_high.mean())

# 6) Bayesian logistic regression
with pm.Model() as m_logit:
    alpha = pm.Normal("alpha", 0, 1)
    beta  = pm.Normal("beta", 0, 1, shape=X_poly.shape[1])  # include temp2
    logits = alpha + pm.math.dot(X_poly, beta)
    p = pm.Deterministic("p", pm.math.sigmoid(logits))
    y = pm.Bernoulli("y", p=p, observed=is_high)

    idata_logit = pm.sample(2000, tune=2000, chains=2, target_accept=0.9, random_seed=123)

print(az.summary(idata_logit, var_names=["alpha","beta"], hdi_prob=0.95))

# 7) Which variable influences outcome most?
summ = az.summary(idata_logit, var_names=["beta"], hdi_prob=0.95)
summ["name"] = names_poly
summ = summ.set_index("name")
print(summ[["mean","hdi_2.5%","hdi_97.5%"]].sort_values("mean", key=np.abs, ascending=False))
