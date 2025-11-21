from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

model = DiscreteBayesianNetwork([
    ('O', 'H'),
    ('O', 'W'),
    ('H', 'R'),
    ('W', 'R'),
    ('H', 'E'),
    ('R', 'C')
])

cpd_O = TabularCPD(
    variable='O',
    variable_card=2,
    values=[[0.3], [0.7]],
    state_names={'O': ['cold', 'mild']}
)

cpd_H = TabularCPD(
    variable='H',
    variable_card=2,
    values=[[0.9, 0.2], [0.1, 0.8]],
    evidence=['O'], evidence_card=[2],
    state_names={'H': ['yes', 'no'], 'O': ['cold', 'mild']}
)

cpd_W = TabularCPD(
    variable='W',
    variable_card=2,
    values=[[0.1, 0.6], [0.9, 0.4]],
    evidence=['O'], evidence_card=[2],
    state_names={'W': ['yes', 'no'], 'O': ['cold', 'mild']}
)

cpd_R = TabularCPD(
    variable='R',
    variable_card=2,
    values=[[0.6, 0.9, 0.3, 0.5],
            [0.4, 0.1, 0.7, 0.5]],
    evidence=['H', 'W'], evidence_card=[2, 2],
    state_names={'R': ['warm', 'cool'], 'H': ['yes', 'no'], 'W': ['yes', 'no']}
)

cpd_E = TabularCPD(
    variable='E',
    variable_card=2,
    values=[[0.8, 0.2],
            [0.2, 0.8]],
    evidence=['H'], evidence_card=[2],
    state_names={'E': ['high', 'low'], 'H': ['yes', 'no']}
)

cpd_C = TabularCPD(
    variable='C',
    variable_card=2,
    values=[[0.85, 0.40],
            [0.15, 0.60]],
    evidence=['R'], evidence_card=[2],
    state_names={'C': ['comfortable', 'uncomfortable'], 'R': ['warm', 'cool']}
)


model.add_cpds(cpd_O, cpd_H, cpd_W, cpd_R, cpd_E, cpd_C)
print("Model valid:", model.check_model())

#grafic
G = nx.DiGraph()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=7)

nx.draw(
    G, pos, with_labels=True, arrows=True,
    node_size=2500, node_color="lightblue",
    font_size=12, font_weight="bold",
    arrowstyle='->', arrowsize=20
)

plt.title("grafic")
plt.show()

infer = VariableElimination(model)

#P(H=yes|C=comfortable)
posterior_H = infer.query(['H'], evidence={'C': 'comfortable'})
print(posterior_H)

#P(E=high|C=comfortable)
posterior_E = infer.query(['E'], evidence={'C': 'comfortable'})
print(posterior_E)

#MAP(H, W | C = comfortable)
map_HW = infer.map_query(['H', 'W'], evidence={'C': 'comfortable'})
print(map_HW)