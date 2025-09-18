import networkx as nx
import matplotlib.pyplot as plt

from pyvis.network import Network as pn

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

railway.add_node("C1", type="platform", platform_no=1, direction="up", is_available=True, station="C")
railway.add_node("C2", type="platform", platform_no=2, direction="down", is_available=True, station="C")

railway.add_node("D1", type="platform", platform_no=1, direction="up", is_available=True, station="D")

railway.add_node("carshed", type="carshad", train_count=12)
railway.add_edge("carshed", "ibothBD0")
railway.add_edge("ibothBD0", "carshed")

railway.add_node("iupAB1",type="intersection")
railway.add_node("iupAB2",type="intersection")
railway.add_node("iupAB3",type="intersection")

railway.add_node("idownBA0",type="intersection")
railway.add_node("idownBA1",type="intersection")
railway.add_node("idownBA2",type="intersection")
railway.add_node("idownBA3",type="intersection")


railway.add_node("ibothBD0",type="intersection")

railway.add_edge("A1", "iupAB3")
railway.add_edge("iupAB3", "A1")
railway.add_edge("iupAB3", "iupAB2")
railway.add_edge("iupAB2", "iupAB1") 
railway.add_edge("iupAB1", "B1")


railway.add_node("iupBC0",type="intersection")
railway.add_node("idownCB0",type="intersection")

railway.add_node("iupBC1",type="intersection")
railway.add_node("idownCB1",type="intersection")

railway.add_edge("B1","iupBC0")
railway.add_edge("iupBC0","iupBC1")
railway.add_edge("iupBC1","C1")
railway.add_edge("C1","iupBC1")





# railway.add_edge("iupAB0", "idownBA0")
railway.add_edge("idownBA1", "iupAB1")

railway.add_edge("B2", "idownBA0")
railway.add_edge("idownBA0", "idownBA1")
railway.add_edge("idownBA1", "idownBA3")
railway.add_edge("idownBA3", "idownBA2")
railway.add_edge("idownBA2", "A2")
railway.add_edge("idownBA2","iupAB2")
railway.add_edge("A2","idownBA2")
railway.add_edge("idownBA3", "iupAB3")



railway.add_edge("C2", "idownCB0")
railway.add_edge("idownCB0", "C2")
railway.add_edge("idownCB0", "idownCB1")
railway.add_edge("idownCB1", "B2")

railway.add_edge("iupBC0", "idownCB0")
railway.add_edge("iupBC1", "idownCB1")

railway.add_edge("D1", "ibothBD0")
railway.add_edge("ibothBD0", "D1")
railway.add_edge("ibothBD0", "idownBA1")
railway.add_edge( "idownBA0","ibothBD0")
net = pn(directed=True, height="900px", width="100%", bgcolor="#222222", font_color="white", )

net.barnes_hut()
net.toggle_physics(False)
net.set_edge_smooth('straight')

# Load NetworkX graph
net.from_nx(railway)

# Save and open in browser
net.show("railway_network.html",notebook=False)








