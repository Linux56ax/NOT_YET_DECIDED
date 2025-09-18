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

