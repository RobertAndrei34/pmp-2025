import random

p_red = 0.5*(3/10) + (1/6)*(4/10) + (1/3)*(3/10)
print('Exact P(red)=', p_red)

def sim(N=100000):
    r=0
    for _ in range(N):
        d = random.randint(1,6)
        R,B,K = 3,4,2
        if d in (2,3,5): K+=1
        elif d==6: R+=1
        else: B+=1
        r += (random.randrange(R+B+K) < R)
    return r/N

print('MC P(red)≈', sim())
