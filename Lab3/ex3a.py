import random

def play_once():
    s = random.randint(0,1)
    n = random.randint(1,6)
    second = 1-s
    p = 4/7 if second==1 else 0.5
    m = sum(random.random()<p for _ in range(2*n))
    return s if n>=m else 1-s

def simulate(N=100000):
    w=[0,0]
    for _ in range(N): w[play_once()]+=1
    return w[0]/N, w[1]/N

p0,p1 = simulate()
print('P0 win≈',p0,' P1 win≈',p1)
