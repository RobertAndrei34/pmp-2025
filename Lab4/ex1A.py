import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import MarkovNetwork

model = MarkovNetwork()

model.add_nodes_from(['A1', 'A2', 'A3', 'A4', 'A5'])

model.add_edges_from([('A1', 'A2'), ('A1', 'A3'), ('A2', 'A5'), ('A2', 'A4'), ('A4', 'A5'), ('A3', 'A4')])

G = nx.Graph()

G.add_edges_from(model.edges())

nx.draw(G, with_labels=True, node_size=5000, node_color="skyblue", font_size=12)
plt.title("Markov Network Graph")
plt.show()

cliques = list(nx.find_cliques(G))
print("Cliques in the model:", cliques)
