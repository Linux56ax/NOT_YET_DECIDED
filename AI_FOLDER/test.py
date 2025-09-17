import networkx as nx

graph = nx.DiGraph()

graph.add_node("A", attr1="value1")
graph.add_node("B", attr1="value2")
graph.add_edge("A", "B", weight=5)

