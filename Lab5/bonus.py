import numpy as np

states = ["D","M","E"]
obs_map = {"FB":0, "B":1, "S":2, "NS":3}

pi = np.array([1/3, 1/3, 1/3], dtype=float)
A  = np.array([
    [0.0 , 0.50, 0.50],
    [0.50, 0.25, 0.25],
    [0.50, 0.25, 0.25]
], dtype=float)
B  = np.array([
    [0.10, 0.20, 0.40, 0.30],
    [0.15, 0.25, 0.50, 0.10],
    [0.20, 0.30, 0.40, 0.10]
], dtype=float)

obs_labels = ["FB","FB","S","B","B","S","B","B","NS","B","B"]
O = np.array([obs_map[x] for x in obs_labels], dtype=int)

T, N = len(O), len(states)

def safe_log(p):
    out = np.full_like(p, -np.inf, dtype=float)
    mask = p > 0.0
    out[mask] = np.log(p[mask])
    return out

log_pi = safe_log(pi)
log_A  = safe_log(A)
log_B  = safe_log(B)

delta = np.full((T, N), -np.inf)
psi   = np.zeros((T, N), dtype=int)

delta[0] = log_pi + log_B[:, O[0]]
for t in range(1, T):
    for j in range(N):
        prev = delta[t-1] + log_A[:, j]   # all -inf-safe
        psi[t, j] = int(np.argmax(prev))
        delta[t, j] = prev[psi[t, j]] + log_B[j, O[t]]

path = np.zeros(T, dtype=int)
path[-1] = int(np.argmax(delta[-1]))
for t in range(T-2, -1, -1):
    path[t] = psi[t+1, path[t+1]]

path_states = [states[i] for i in path]
viterbi_joint_prob = float(np.exp(np.max(delta[-1])))  # P(best path AND obs)

print("Observations:", obs_labels)
print("Viterbi states:", path_states)
print("Joint P(best path, obs) =", viterbi_joint_prob)
