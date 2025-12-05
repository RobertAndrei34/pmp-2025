import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def main():
    publicity = np.array([
        1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
        6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0
    ])

    sales = np.array([
        5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
        15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0
    ])

    with pm.Model() as model:

        intercept = pm.Normal("intercept", mu=0, sigma=10)
        slope = pm.Normal("slope", mu=0, sigma=10)

        sigma = pm.HalfNormal("sigma", sigma=5)

        mu = intercept + slope * publicity

        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=sales)
        trace = pm.sample(
            draws=2000,
            tune=1000,
            target_accept=0.95,
            random_seed=42
        )

    #a)
    print("\nPosterior summary (intercept, slope, sigma):\n")
    summary = az.summary(trace, var_names=["intercept", "slope", "sigma"])
    print(summary)

    #b)
    print("\n95% HDI for intercept, slope, sigma:\n")
    hdis = az.hdi(trace, hdi_prob=0.95)
    print(hdis[["intercept", "slope", "sigma"]])

    #c)
    new_publicity = np.array([12.0, 13.0, 14.0])
    intercept_samples = trace.posterior["intercept"].values.reshape(-1)
    slope_samples = trace.posterior["slope"].values.reshape(-1)
    pred_matrix = intercept_samples[:, None] + slope_samples[:, None] * new_publicity[None, :]
    pred_mean = pred_matrix.mean(axis=0)
    pred_hdi = az.hdi(pred_matrix, hdi_prob=0.95)

    print("\nPredicted sales for new publicity levels (regression line only):\n")
    for p, m, h in zip(new_publicity, pred_mean, pred_hdi):
        print(
            f"Publicity = {p:4.1f} (thousand $)  ->  "
            f"Predicted sales ≈ {m:5.2f} (thousand $), "
            f"95% HDI ≈ [{h[0]:5.2f}, {h[1]:5.2f}]"
        )
    publicity_grid = np.linspace(publicity.min(), publicity.max(), 100)
    grid_matrix = intercept_samples[:, None] + slope_samples[:, None] * publicity_grid[None, :]
    grid_mean = grid_matrix.mean(axis=0)
    grid_hdi = az.hdi(grid_matrix, hdi_prob=0.95)

    plt.figure(figsize=(8, 5))
    plt.scatter(publicity, sales, label="Observed data")
    plt.plot(publicity_grid, grid_mean, label="Posterior mean line")
    plt.fill_between(
        publicity_grid,
        grid_hdi[:, 0],
        grid_hdi[:, 1],
        alpha=0.3,
        label="95% HDI band"
    )
    plt.xlabel("Publicity (thousands of dollars)")
    plt.ylabel("Sales (thousands of dollars)")
    plt.title("Bayesian Linear Regression: Sales vs Publicity")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
