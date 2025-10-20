try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:
    from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import TabularCPD

bn = BN([('S','O'), ('S','L'), ('S','M'), ('L','M')])

cpd_S = TabularCPD('S', 2, [[0.6],[0.4]])
cpd_O = TabularCPD('O', 2, [[0.9,0.3],[0.1,0.7]], evidence=['S'], evidence_card=[2])
cpd_L = TabularCPD('L', 2, [[0.7,0.2],[0.3,0.8]], evidence=['S'], evidence_card=[2])
cpd_M = TabularCPD('M', 2, [[0.8,0.4,0.5,0.1],[0.2,0.6,0.5,0.9]], evidence=['S','L'], evidence_card=[2,2])
bn.add_cpds(cpd_S, cpd_O, cpd_L, cpd_M)

def indep(X, Y, given=None):
    return Y not in bn.active_trail_nodes(X, observed=set(given or []))

print("indep(O,L|S) =", indep('O','L',given=['S']))
print("indep(O,M|S) =", indep('O','M',given=['S']))
print("indep(M,L|S) =", indep('M','L',given=['S']))
print("indep(O,L)   =", indep('O','L'))
