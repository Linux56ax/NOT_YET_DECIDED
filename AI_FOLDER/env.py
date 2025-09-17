import networkx as nx

railway = nx.DiGraph()

class Station:
    def __init__(self, name, num_platforms, graph, platform_types=None):
        """
        name: Station name (string)
        num_platforms: number of platforms (int)
        graph: networkx graph where platforms will be added
        platform_types: optional list of platform types, 
                        defaults to "both" for all
        """
        self.graph = graph
        self.name = name
        self.platforms = []

        if platform_types is None:
            platform_types = ["both"] * num_platforms
  
        for i in range(num_platforms):
            platform_id = f"{name}_P{i+1}"
            graph.add_node(
                platform_id,
                platform_no=i+1,
                platform_type=platform_types[i],
                is_available=True,
                station=name
            )
            self.platforms.append(platform_id)

    def __repr__(self):
        return f"Station({self.name}, Platforms: {list(zip(self.platforms, [self.graph.nodes[p]['platform_type'] for p in self.platforms]))})"


railway.add_node("A1", type="platform", platform_no=1, direction="up", is_available=True, station="A")
railway.add_node("A2", type="platform", platform_no=2, direction="down", is_available=True, station="A")

railway.add_node("B1", type="platform", platform_no=1, direction="up", is_available=True, station="B")
railway.add_node("B2", type="platform", platform_no=2, direction="down", is_available=True, station="B")

railway.add_node("iupAB0",type="intersection")
railway.add_edge("A1", "iupAB0")
railway.add_edge("iupAB0", "B1")

railway.add_node("idownBA0",type="intersection")

railway.add_edge("iupAB0", "idownBA0")
railway.add_edge("B2", "idownBA0")
railway.add_edge("idownBA0", "A2")



import matplotlib.pyplot as plt

# Add this at the end of your file
plt.figure(figsize=(12, 8))

# Create position layout for better visualization
pos = nx.spring_layout(railway, seed=42)

# Draw different node types with different colors
platform_nodes = [node for node, data in railway.nodes(data=True) if data.get('type') == 'platform']
intersection_nodes = [node for node, data in railway.nodes(data=True) if data.get('type') == 'intersection']

# Draw nodes
nx.draw_networkx_nodes(railway, pos, nodelist=platform_nodes, node_color='lightblue', node_size=1500, alpha=0.8)
nx.draw_networkx_nodes(railway, pos, nodelist=intersection_nodes, node_color='orange', node_size=800, alpha=0.8)

# Draw edges
nx.draw_networkx_edges(railway, pos, edge_color='gray', arrows=True, arrowsize=20, arrowstyle='->')

# Draw labels
nx.draw_networkx_labels(railway, pos, font_size=8)

plt.title("Railway Network Graph")
plt.axis('off')
plt.show()