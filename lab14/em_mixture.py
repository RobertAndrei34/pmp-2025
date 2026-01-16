import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def em_mixture_regression(x, y, K, max_iter=200, tol=1e-5, seed=42):
    """
    EM algorithm for mixture of quadratic regressions
    """
    np.random.seed(seed)
    n = len(x)

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X = poly.fit_transform(x.reshape(-1, 1))

    # Initialization
    weights = np.ones(K) / K
    sigmas = np.random.uniform(8, 15, K)
    betas = np.random.randn(K, X.shape[1])

    log_likelihood_old = -np.inf

    for _ in range(max_iter):
        # ---------- E-step ----------
        resp = np.zeros((n, K))
        for k in range(K):
            mu = X @ betas[k]
            resp[:, k] = weights[k] * norm.pdf(y, mu, sigmas[k])

        resp /= resp.sum(axis=1, keepdims=True)

        # ---------- M-step ----------
        Nk = resp.sum(axis=0)
        weights = Nk / n

        for k in range(K):
            W = np.diag(resp[:, k])
            XtW = X.T @ W
            betas[k] = np.linalg.solve(XtW @ X, XtW @ y)

            mu = X @ betas[k]
            sigmas[k] = np.sqrt(np.sum(resp[:, k] * (y - mu) ** 2) / Nk[k])

        # ---------- log-likelihood ----------
        ll = 0.0
        for k in range(K):
            mu = X @ betas[k]
            ll += weights[k] * norm.pdf(y, mu, sigmas[k])
        log_likelihood = np.sum(np.log(ll))

        if np.abs(log_likelihood - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood

    return {
        "weights": weights,
        "betas": betas,
        "sigmas": sigmas,
        "log_likelihood": log_likelihood,
        "X": X
    }
