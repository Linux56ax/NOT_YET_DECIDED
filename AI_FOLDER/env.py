import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network as pn

railway = nx.DiGraph()

railway.add_node("A1", type="platform", platform_no=1, direction="both", is_available=True, station="A")
railway.add_node("A2", type="platform", platform_no=2, direction="both", is_available=True, station="A")

railway.add_node("B1", type="platform", platform_no=1, direction="up", is_available=True, station="B")
railway.add_node("B2", type="platform", platform_no=2, direction="down", is_available=True, station="B")

railway.add_node("C1", type="platform", platform_no=1, direction="both", is_available=True, station="C")
railway.add_node("C2", type="platform", platform_no=2, direction="both", is_available=True, station="C")

railway.add_node("D1", type="platform", platform_no=1, direction="both", is_available=True, station="D")

railway.add_node("carshed", type="carshed", train_count=12)

railway.add_node("iupAB1", type="intersection")
railway.add_node("iupAB2", type="intersection")
railway.add_node("iupAB3", type="intersection")

railway.add_node("idownBA0", type="intersection")
railway.add_node("idownBA1", type="intersection")
railway.add_node("idownBA2", type="intersection")
railway.add_node("idownBA3", type="intersection")

railway.add_node("ibothBD0", type="intersection")

railway.add_node("iupBC0", type="intersection")
railway.add_node("idownCB0", type="intersection")
railway.add_node("iupBC1", type="intersection")
railway.add_node("idownCB1", type="intersection")

railway.add_edge("carshed", "ibothBD0",length=300)
railway.add_edge("ibothBD0", "carshed",length=300)


railway.add_edge("A1", "iupAB3",length=50)
railway.add_edge("iupAB3", "A1",length=50)


railway.add_edge("iupAB3", "iupAB2",length=50)
railway.add_edge("iupAB2", "iupAB1",length=20000)
railway.add_edge("iupAB1", "B1",length=50)


railway.add_edge("B1", "iupBC0",length=50)
railway.add_edge("iupBC0", "iupBC1",length=50)
railway.add_edge("iupBC1", "C1",length=50)
railway.add_edge("C1", "iupBC1",length=50)


railway.add_edge("C2", "idownCB0",length=50)
railway.add_edge("idownCB0", "C2",length=50)  
railway.add_edge("idownCB0", "idownCB1",length=50)
railway.add_edge("idownCB1", "B2",length=50)

railway.add_edge("B2", "idownBA0",length=50)
railway.add_edge("idownBA0", "idownBA1",length=50)
railway.add_edge("idownBA1", "idownBA3",length=20000)
railway.add_edge("idownBA3", "idownBA2")
railway.add_edge("idownBA2", "A2",length=50)
railway.add_edge("A2", "idownBA2",length=50)  


railway.add_edge("idownBA1", "iupAB1",length=25)
railway.add_edge("idownBA2", "iupAB2",length=25)
railway.add_edge("idownBA3", "iupAB3",length=25)
railway.add_edge("iupBC0", "idownCB0",length=25)
railway.add_edge("iupBC1", "idownCB1",length=25)


railway.add_edge("D1", "ibothBD0",length=50)
railway.add_edge("ibothBD0", "D1",length=50)
railway.add_edge("ibothBD0", "idownBA1",length=10000)
railway.add_edge("idownBA0", "ibothBD0",length=10000)

def validate_network():
    """Validate the railway network structure"""
    print(f"Nodes: {railway.number_of_nodes()}")
    print(f"Edges: {railway.number_of_edges()}")
    
    
    isolated = list(nx.isolates(railway))
    if isolated:
        print(f"Warning: Isolated nodes found: {isolated}")
    
    
    if not nx.is_weakly_connected(railway):
        print("Warning: Network is not weakly connected")
        components = list(nx.weakly_connected_components(railway))
        print(f"Connected components: {len(components)}")
    
    return len(isolated) == 0 and nx.is_weakly_connected(railway)

if __name__ == "__main__":
    
    is_valid = validate_network()
    print(f"Network validation: {'PASSED' if is_valid else 'FAILED'}")
    
    net = pn(directed=True, height="900px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut()
    net.toggle_physics(False)
    net.set_edge_smooth('straight')

    net.from_nx(railway)
    net.show("railway_network.html", notebook=False)








