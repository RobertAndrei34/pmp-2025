import numpy as np
from scipy.stats import norm


def compute_waic(x, y, model, n_samples=500):
    """
    Approximate WAIC via parametric bootstrap
    """
    weights = model["weights"]
    betas = model["betas"]
    sigmas = model["sigmas"]
    X = model["X"]
    K = len(weights)
    n = len(y)

    log_lik_samples = np.zeros((n_samples, n))

    for s in range(n_samples):
        k = np.random.choice(K, p=weights)
        beta = betas[k]
        sigma = sigmas[k]
        mu = X @ beta
        log_lik_samples[s] = norm.logpdf(y, mu, sigma)

    lppd = np.sum(np.log(np.mean(np.exp(log_lik_samples), axis=0)))
    p_waic = np.sum(np.var(log_lik_samples, axis=0))

    return -2 * (lppd - p_waic)
