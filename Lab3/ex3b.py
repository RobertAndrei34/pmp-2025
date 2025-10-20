try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:
    from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import math

bn = BN([('S','C'), ('S','N'), ('C','M'), ('N','M'), ('S','W'), ('N','W'), ('M','W')])

cpd_S = TabularCPD('S',2,[[0.5],[0.5]])
cpd_N = TabularCPD('N',6,[[1/6,1/6]]*6, evidence=['S'], evidence_card=[2])
cpd_C = TabularCPD('C',2,[[0.0,1.0],[1.0,0.0]], evidence=['S'], evidence_card=[2])

rows=[]
for c in (0,1):
    p = 0.5 if c==0 else 4/7
    for n in range(1,7):
        dist = [math.comb(2*n,k)*(p**k)*((1-p)**(2*n-k)) for k in range(13)]
        dist += [0.0]*(13-len(dist))
        rows.append(dist)
cpd_M = TabularCPD('M',13,list(map(list,zip(*rows))), evidence=['C','N'], evidence_card=[2,6])

cols=[]
for s in (0,1):
    for n in range(1,7):
        for m in range(13):
            w = s if n>=m else 1-s
            cols.append([1.0,0.0] if w==0 else [0.0,1.0])
cpd_W = TabularCPD('W',2,list(map(list,zip(*cols))), evidence=['S','N','M'], evidence_card=[2,6,13])

bn.add_cpds(cpd_S,cpd_N,cpd_C,cpd_M,cpd_W)
infer = VariableElimination(bn)

pW = infer.query(['W']).values
post = infer.query(['S'], evidence={'M':1}).values
print('P(W=P0)=',float(pW[0]),' P(W=P1)=',float(pW[1]))
print('P(S=P0|M=1)=',float(post[0]),' P(S=P1|M=1)=',float(post[1]))
