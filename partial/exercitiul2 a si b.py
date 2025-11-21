import numpy as np
from hmmlearn import hmm

states = ["W", "R", "S"]
observations = ["L", "M", "H"]

startprob = np.array([0.4, 0.3, 0.3])

transmat = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.5]
])

emissionprob = np.array([
    [0.1, 0.1, 0.7],
    [0.05, 0.25, 0.7],
    [0.8, 0.15, 0.05]
])

model = hmm.CategoricalHMM(n_components=3)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

pi = np.array([0.4, 0.3, 0.3])

A = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.5]
])

B = np.array([
    [0.1, 0.1, 0.7],
    [0.05, 0.25, 0.7],
    [0.8, 0.15, 0.05]
])

O = [1, 2, 0]

alpha = pi * B[:, O[0]]

for t in range(1, len(O)):
    alpha = (alpha @ A) * B[:, O[t]]

probability = np.sum(alpha)

print("P([M,H,L]) =", probability)

