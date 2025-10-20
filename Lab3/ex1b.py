try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:
    from pgmpy.models import BayesianNetwork as BN

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

bn = BN([('S','O'), ('S','L'), ('S','M'), ('L','M')])

cpd_S = TabularCPD('S', 2, [[0.6], [0.4]])
cpd_O = TabularCPD('O', 2, [[0.9, 0.3],
                            [0.1, 0.7]], evidence=['S'], evidence_card=[2])
cpd_L = TabularCPD('L', 2, [[0.7, 0.2],
                            [0.3, 0.8]], evidence=['S'], evidence_card=[2])
cpd_M = TabularCPD('M', 2, [[0.8, 0.4, 0.5, 0.1],
                            [0.2, 0.6, 0.5, 0.9]], evidence=['S','L'], evidence_card=[2,2])

bn.add_cpds(cpd_S, cpd_O, cpd_L, cpd_M)
bn.check_model()

infer = VariableElimination(bn)

def post_spam(o, l, m):
    q = infer.query(['S'], evidence={'O': o, 'L': l, 'M': m})
    return float(q.values[1])

for O in (0,1):
    for L in (0,1):
        for M in (0,1):
            p = post_spam(O,L,M)
            cls = 'SPAM' if p >= 0.5 else 'NOT SPAM'
            print(f"O={O} L={L} M={M} -> P(spam)={p:.4f} => {cls}")
