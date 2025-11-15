import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def run_bayesian_model():
    data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

        trace = pm.sample(1000, tune=500, chains=2, return_inferencedata=True)

    print(az.summary(trace, hdi_prob=0.95))
    az.plot_posterior(trace)
    plt.show()

if __name__ == "__main__":
    run_bayesian_model()
