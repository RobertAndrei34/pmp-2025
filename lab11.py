import numpy as np
import pandas as pd
import pymc as pm
import arviz as az




CSV_PATH = "Prices.csv"


def zscore(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std(ddof=0)
    return (x - m) / s, m, s


def main():

    df = pd.read_csv(CSV_PATH)

    y = df["Price"].astype(float).to_numpy()
    x1 = df["Speed"].astype(float).to_numpy()       
    hd = df["HardDrive"].astype(float).to_numpy()   
    x2 = np.log(hd)                                

    prem = (
        df["Premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    )

    if prem.isna().any():
        bad = df.loc[prem.isna(), "Premium"].unique()
        raise ValueError(f"Unexpected values in Premium column: {bad}")

    prem = prem.to_numpy(dtype=int)

    x1z, x1_mean, x1_sd = zscore(x1)
    x2z, x2_mean, x2_sd = zscore(x2)

    y_mean = y.mean()
    yc = y - y_mean

    coords = {"obs_id": np.arange(len(y))}

    # (a)

    with pm.Model(coords=coords) as model:
        x1_data = pm.Data("x1z", x1z, dims="obs_id")
        x2_data = pm.Data("x2z", x2z, dims="obs_id")

        alpha = pm.Normal("alpha", mu=0.0, sigma=1000.0)
        beta1 = pm.Normal("beta1", mu=0.0, sigma=1000.0)
        beta2 = pm.Normal("beta2", mu=0.0, sigma=1000.0)
        sigma = pm.HalfNormal("sigma", sigma=1000.0)

        mu = alpha + beta1 * x1_data + beta2 * x2_data
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=yc, dims="obs_id")

        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,  
            target_accept=0.9,
            random_seed=42,
            progressbar=True
        )

    # (b)
    hdi_95 = az.hdi(idata, var_names=["beta1", "beta2"], hdi_prob=0.95)
    print("\n(b) 95% HDI for beta1 and beta2 (standardized predictors, centered y):")
    print(hdi_95)

    # (c) 
    b1_low, b1_high = hdi_95["beta1"].to_numpy()
    b2_low, b2_high = hdi_95["beta2"].to_numpy()

    def useful(low, high):
        return not (low <= 0.0 <= high)

    print("\n(c) Usefulness (95% HDI excludes 0?):")
    print(f"  beta1 (Speed): {'YES' if useful(b1_low, b1_high) else 'NO'}  | HDI: [{b1_low:.4f}, {b1_high:.4f}]")
    print(f"  beta2 (ln(HardDrive)): {'YES' if useful(b2_low, b2_high) else 'NO'}  | HDI: [{b2_low:.4f}, {b2_high:.4f}]")

    posterior = idata.posterior
    alpha_s = posterior["alpha"].values.reshape(-1)
    beta1_s = posterior["beta1"].values.reshape(-1)
    beta2_s = posterior["beta2"].values.reshape(-1)
    sigma_s = posterior["sigma"].values.reshape(-1)

    # (d) 
    x1_new = 33.0
    hd_new = 540.0
    x2_new = np.log(hd_new)

    x1_new_z = (x1_new - x1_mean) / x1_sd
    x2_new_z = (x2_new - x2_mean) / x2_sd

    mu_new_centered = alpha_s + beta1_s * x1_new_z + beta2_s * x2_new_z
    mu_new = mu_new_centered + y_mean

    mu_hdi_90 = az.hdi(mu_new, hdi_prob=0.90)
    print("\n(d) 90% HDI for expected price mu at Speed=33 MHz, HardDrive=540 MB:")
    print(f"  mean(mu) ≈ {mu_new.mean():.2f}")
    print(f"  90% HDI  ≈ [{mu_hdi_90[0]:.2f}, {mu_hdi_90[1]:.2f}]")

    # (e) 
    rng = np.random.default_rng(123)
    y_new_centered = rng.normal(loc=mu_new_centered, scale=sigma_s)
    y_new = y_new_centered + y_mean

    y_hdi_90 = az.hdi(y_new, hdi_prob=0.90)
    print("\n(e) 90% posterior predictive interval for price at Speed=33 MHz, HardDrive=540 MB:")
    print(f"  mean(y_new) ≈ {y_new.mean():.2f}")
    print(f"  90% PI     ≈ [{y_hdi_90[0]:.2f}, {y_hdi_90[1]:.2f}]")

    # Bonus
    premz, prem_mean, prem_sd = zscore(prem)

    with pm.Model(coords=coords) as model_premium:
        x1_data = pm.Data("x1z", x1z, dims="obs_id")
        x2_data = pm.Data("x2z", x2z, dims="obs_id")
        p_data  = pm.Data("premz", premz, dims="obs_id")

        alpha = pm.Normal("alpha", mu=0.0, sigma=1000.0)
        beta1 = pm.Normal("beta1", mu=0.0, sigma=1000.0)
        beta2 = pm.Normal("beta2", mu=0.0, sigma=1000.0)
        gamma = pm.Normal("gamma_premium", mu=0.0, sigma=1000.0)
        sigma = pm.HalfNormal("sigma", sigma=1000.0)

        mu = alpha + beta1 * x1_data + beta2 * x2_data + gamma * p_data
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=yc, dims="obs_id")

        idata_p = pm.sample(
            draws=3000,
            tune=2000,
            chains=4,
            target_accept=0.9,
            random_seed=42
        )

    gamma_hdi_95 = az.hdi(idata_p, var_names=["gamma_premium"], hdi_prob=0.95)
    gl, gh = gamma_hdi_95["gamma_premium"].to_numpy()

    print("\n(Bonus) 95% HDI for premium effect (gamma_premium):")
    print(gamma_hdi_95)
    print(f"  Premium affects price? {'YES' if useful(gl, gh) else 'NO'} (95% HDI excludes 0)")

    print("\nPosterior summary (base model):")
    print(az.summary(idata, var_names=["alpha", "beta1", "beta2", "sigma"], hdi_prob=0.95))

    print("\nPosterior summary (premium model):")
    print(az.summary(idata_p, var_names=["alpha", "beta1", "beta2", "gamma_premium", "sigma"], hdi_prob=0.95))


if __name__ == "__main__":
    main()
