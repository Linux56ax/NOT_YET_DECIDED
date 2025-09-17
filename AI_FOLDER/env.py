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


railway.add_node("iupAB0",type="intersection")
railway.add_node("iupAB1",type="intersection")
railway.add_node("iupAB2",type="intersection")
railway.add_node("iupAB3",type="intersection")

railway.add_node("idownBA0",type="intersection")
railway.add_node("idownBA1",type="intersection")
railway.add_node("idownBA2",type="intersection")
railway.add_node("idownBA3",type="intersection")

railway.add_edge("A1", "iupAB3")
railway.add_edge("iupAB3", "A1")
railway.add_edge("iupAB3", "iupAB2")
railway.add_edge("iupAB2", "iupAB0") 
railway.add_edge("iupAB0", "iupAB1") 
railway.add_edge("iupAB1", "B1")
railway.add_edge("B1","iupAB1" )


railway.add_edge("iupAB0", "idownBA0")
railway.add_edge("iupAB1", "idownBA1")

railway.add_edge("B2", "idownBA0")
railway.add_edge("idownBA0", "B2")
railway.add_edge("idownBA0", "idownBA1")
railway.add_edge("idownBA1", "idownBA3")
railway.add_edge("idownBA3", "idownBA2")
railway.add_edge("idownBA2", "A2")
railway.add_edge("idownBA2","iupAB2")
railway.add_edge("A2","idownBA2")
railway.add_edge("idownBA3", "iupAB3")


# --- Create interactive network ---
net = pn(directed=True, height="750px", width="100%", bgcolor="#222222", font_color="white", )

net.barnes_hut()  # physics layout (default)
# OR disable physics for fixed layout
net.toggle_physics(False)
net.set_edge_smooth('straight')

# Load NetworkX graph
net.from_nx(railway)

# Save and open in browser
net.show("railway_network.html",notebook=False)


def visualize_railway_interactive_matplotlib(g):
    """
    Drag nodes inside a matplotlib window and print final coordinates on close.
    Pure Python (no extra deps).
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.spring_layout(g, seed=8)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Drag nodes. Close window to print final positions.")
    ax.axis("off")

    def draw():
        ax.clear()
        ax.axis("off")
        platform_nodes = [n for n, d in g.nodes(data=True) if d.get("type") == "platform"]
        intersection_nodes = [n for n, d in g.nodes(data=True) if d.get("type") == "intersection"]
        nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowstyle="->", connectionstyle="arc3,rad=0.05")
        nx.draw_networkx_nodes(g, pos, nodelist=platform_nodes, node_color="#4F81BD", node_size=900, edgecolors="black")
        nx.draw_networkx_nodes(g, pos, nodelist=intersection_nodes, node_color="#F79646", node_size=700, edgecolors="black")
        nx.draw_networkx_labels(g, pos, font_color="black")
        fig.canvas.draw_idle()


    draw()

    class Dragger:
        def __init__(self):
            self.dragging = None
            self.press_offset = (0, 0)

        def nearest(self, event):
            if event.xdata is None or event.ydata is None:
                return None
            # Simple distance search
            mind, node = 0.05, None
            for n, (x, y) in pos.items():
                d = ((x - event.xdata)**2 + (y - event.ydata)**2) ** 0.5
                if d < mind:
                    mind, node = d, n
            return node

        def on_press(self, event):
            if event.button != 1:
                return
            n = self.nearest(event)
            if n:
                self.dragging = n
                px, py = pos[n]
                self.press_offset = (px - event.xdata, py - event.ydata)

        def on_release(self, event):
            if event.button != 1:
                return
            self.dragging = None

        def on_move(self, event):
            if self.dragging and event.xdata is not None and event.ydata is not None:
                dx, dy = self.press_offset
                pos[self.dragging] = (event.xdata + dx, event.ydata + dy)
                draw()

    d = Dragger()
    cidp = fig.canvas.mpl_connect("button_press_event", d.on_press)
    cidr = fig.canvas.mpl_connect("button_release_event", d.on_release)
    cidm = fig.canvas.mpl_connect("motion_notify_event", d.on_move)

    def on_close(evt):
        print("Final positions dict:\n", pos)

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()
    return pos  # also returned for programmatic use


if __name__ == "__main__":
    # visualize_railway_interactive_html(railway)
    visualize_railway_interactive_matplotlib(railway)







