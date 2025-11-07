# Bayesian Poisson rate (lambda) with Gamma prior — compute posterior, 94% HDI, and mode.
# We'll keep it parameterized, but default to a weak prior Gamma(a0=1, b0=1).
# Data: over n hours, total calls S (sum of Poisson counts).

import numpy as np
import math
import matplotlib.pyplot as plt

# === Inputs ===
n = 10               # number of hours observed
S = 180              # total calls observed
a0 = 1.0             # prior alpha
b0 = 1.0             # prior beta (rate form)

# === Posterior parameters (conjugacy) ===
alpha_post = a0 + S
beta_post  = b0 + n

# === Posterior summaries ===
mean_post = alpha_post / beta_post
mode_post = (alpha_post - 1) / beta_post if alpha_post > 1 else 0.0
var_post  = alpha_post / (beta_post**2)
sd_post   = math.sqrt(var_post)

# === Gamma(pdf) utilities in rate parameterization ===
def gamma_logpdf(x, alpha, beta):
    # log of: beta^alpha / Gamma(alpha) * x^(alpha-1) * exp(-beta*x)
    if x <= 0:
        return -np.inf
    return alpha * math.log(beta) - math.lgamma(alpha) + (alpha - 1) * math.log(x) - beta * x

def gamma_pdf(x, alpha, beta):
    return math.exp(gamma_logpdf(x, alpha, beta))

# === Compute 94% HDI via density-threshold method on a grid ===
prob_mass = 0.94

# choose a reasonable x-range: [0, mean + 8*sd], but ensure upper > mean
x_max = max(mean_post + 8*sd_post, mean_post * 2, 30.0)
x_grid = np.linspace(1e-6, x_max, 20000)  # fine grid for accuracy
pdf_vals = np.array([gamma_pdf(x, alpha_post, beta_post) for x in x_grid])

# Normalize numerically (for accurate probability mass computations)
area = np.trapz(pdf_vals, x_grid)
pdf_norm = pdf_vals / area

# Binary search density threshold 't' so that area where pdf>=t equals prob_mass
lo, hi = 0.0, pdf_norm.max()
for _ in range(60):
    mid = 0.5 * (lo + hi)
    mask = pdf_norm >= mid
    mass_mid = np.trapz(pdf_norm[mask], x_grid[mask])
    if mass_mid >= prob_mass:
        lo = mid  # need a higher threshold to squeeze mass down to target
    else:
        hi = mid
t = lo
mask = pdf_norm >= t

# Extract HDI bounds as min/max x where mask is True (unimodal => single interval)
# To be robust to discretization, find first/last True indices
idx = np.where(mask)[0]
hdi_low = x_grid[idx[0]]
hdi_high = x_grid[idx[-1]]

# === Print numeric results ===
print("Posterior: Gamma(alpha, beta) with rate parameterization")
print(f"alpha = {alpha_post:.6g}, beta = {beta_post:.6g}")
print(f"Mean  = {mean_post:.6g} calls/hour")
print(f"Mode  = {mode_post:.6g} calls/hour  (MAP)")
print(f"SD    = {sd_post:.6g} calls/hour")
print(f"94% HDI ≈ [{hdi_low:.6g}, {hdi_high:.6g}] calls/hour")

# === Plot posterior with HDI and mode ===
plt.figure(figsize=(7,4.2))
plt.plot(x_grid, pdf_norm)
plt.axvline(hdi_low, linestyle="--")
plt.axvline(hdi_high, linestyle="--")
plt.axvline(mode_post, linestyle=":")
plt.title("Posterior density of λ (Gamma) with 94% HDI and mode")
plt.xlabel("λ (calls/hour)")
plt.ylabel("Posterior density")
plt.tight_layout()
plt.show()
