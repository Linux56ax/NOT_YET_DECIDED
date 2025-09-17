# ['A1',1,2,3,4,'B1']
# ['B2',5,6,7,8,'A2']
# ['B1',4]
# ['A2',8]
# [1,'A1']
# [5,'A2']
# [3,8]
# [4,7]
# [6,1]
# [5,2]


import networkx as nx
from pyvis.network import Network

# --- Build your directed graph ---
G = nx.DiGraph()

# Example railway-style connections
G.add_edge("A1", "J1", direction="up")
G.add_edge("J1", "B1", direction="side")
G.add_edge("J1", "J2", direction="main")
G.add_edge("B2", "J2", direction="entry")
G.add_edge("J2", "A2", direction="down")

# --- Create interactive network ---
net = Network(directed=True, height="750px", width="100%", bgcolor="#222222", font_color="white")

# Load NetworkX graph
net.from_nx(G)

# Save and open in browser
net.show("railway_network.html",notebook=False)

