# Ex. 1

Considerăm un magazin vizitat de `n` clienți într-o zi.

- `Y` = numărul de clienți care cumpără un anumit produs  
- `θ` = probabilitatea (cunoscută) ca un client să cumpere  
- `Y | n, θ ∼ Binomial(n, θ)`  
- `n ∼ Poisson(10)`

Analizăm cazurile:
- `Y ∈ {0, 5, 10}`
- `θ ∈ {0.2, 0.5}`

## (a)

Estimăm distribuția a posteriori `p(n | Y, θ)` pentru toate combinațiile `(Y, θ)`.

```python
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]
posterior_results = {}

for Y in Y_values:
    for theta in theta_values:
        with pm.Model() as model:
            # prior pentru numarul de clienti
            n = pm.Poisson("n", mu=10)

            # verosimilitatea binomiala
            Y_obs = pm.Binomial("Y_obs", n=n, p=theta, observed=Y)

            # esantionare din posterior
            idata = pm.sample(2000, return_inferencedata=True, cores=1)

        posterior_results[(Y, theta)] = idata

# vizualizare posteriors p(n | Y, θ)
fig = plt.figure(figsize=(12, 10))
idx = 1
for Y in Y_values:
    for theta in theta_values:
        plt.subplot(3, 2, idx)
        az.plot_posterior(
            posterior_results[(Y, theta)],
            var_names=["n"],
            ax=plt.gca()
        )
        plt.title(f"Posterior p(n | Y={Y}, θ={theta})")
        idx += 1

plt.tight_layout()
plt.show()
```
## (b)

Efectul lui Y
	•	Y mare ⇒ posteriorul se muta spre n mare.
	•	Y mic ⇒ posteriorul favorizeaza n mic.

Explicatia: pentru a produce multe cumparari trebuie sa fi avut multi clienti.

⸻

Efectul lui θ
	•	θ mare ⇒ e usor sa obtii Y ⇒ n poate fi mic.
	•	θ mic ⇒ e greu sa obtii Y ⇒ n trebuie sa fie mare.
