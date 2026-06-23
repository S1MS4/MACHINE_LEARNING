import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

layers = [
    {"n": 6, "label": "Input Layer",   "color": "#C4DADD00", "edge": "#4DD4C9", "text_color": "#0D9488"},
    {"n": 3, "label": "Hidden Layer 1","color": "#C3B5FD00", "edge": "#7C3AED", "text_color": "#5B21B6"},
    {"n": 4, "label": "Hidden Layer 2","color": "#C3B5FD00", "edge": "#7C3AED", "text_color": "#5B21B6"},
    {"n": 2, "label": "Output Layer",  "color": "#FED7AA", "edge": "#EA580C", "text_color": "#C2410C"},
]

node_labels = [
    ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"],
    ["H1_1", "H1_2", "H1_3"],
    ["H2_1", "H2_2", "H2_3", "H2_4"],
    ["No Disease", "Has Disease"],
]
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.axis("off")

x_positions = [0.1, 0.37, 0.63, 0.9]
node_radius = 0.033
node_positions = []

for li, layer in enumerate(layers):
    n = layer["n"]
    x = x_positions[li]
    ys = [(i + 1) / (n + 1) for i in range(n)]
    node_positions.append(list(zip([x] * n, ys)))

for li in range(len(layers) - 1):
    for (x1, y1) in node_positions[li]:
        for (x2, y2) in node_positions[li + 1]:
            ax.annotate("", xy=(x2 - node_radius, y2),
                        xytext=(x1 + node_radius, y1),
                        arrowprops=dict(arrowstyle="-|>",
                                        color="#9CA3AF",
                                        lw=0.8,
                                        mutation_scale=8))

for li, layer in enumerate(layers):
    for ni, (x, y) in enumerate(node_positions[li]):
        circle = plt.Circle((x, y), node_radius,
                             color=layer["color"],
                             ec=layer["edge"],
                             linewidth=2,
                             zorder=5)
        ax.add_patch(circle)

        # node label inside circle (short)
        short = node_labels[li][ni].replace(" ", "\n") if li in [0, 3] else node_labels[li][ni]
        fontsize = 6.5 if li in [0, 3] else 7.5
        ax.text(x, y, short, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=layer["edge"], zorder=6)

for li, layer in enumerate(layers):
    x = x_positions[li]
    box = mpatches.FancyBboxPatch((x - 0.09, 0.92), 0.18, 0.055,
                                   boxstyle="round,pad=0.01",
                                   linewidth=1.5,
                                   edgecolor=layer["edge"],
                                   facecolor=layer["color"],
                                   zorder=4)
    ax.add_patch(box)
    n_label = f"{layer['label']}\n({layers[li]['n']} neurons)"
    ax.text(x, 0.948, n_label, ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color=layer["text_color"], zorder=5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("../Sources/network_architecture.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()