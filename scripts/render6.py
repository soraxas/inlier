#!/usr/bin/env python3
"""Render a PolyFit-style .vg point cloud as 6 orthographic faces.

Points are coloured by their vertex-group's `group_color` (ungrouped = grey),
so the caller encodes semantics (storey / exterior-wall) via colour.

Usage: render6.py <input.vg> <output.png> [title]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

vg, png = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else vg

toks = open(vg).read().split()
i = 0
def nx():
    global i
    v = toks[i]; i += 1; return v

assert nx() == "num_points:"; N = int(nx())
P = np.array([[float(nx()), float(nx()), float(nx())] for _ in range(N)])
assert nx() == "num_colors:"
for _ in range(int(nx())): nx(); nx(); nx()
assert nx() == "num_normals:"
for _ in range(int(nx())): nx(); nx(); nx()
assert nx() == "num_groups:"; G = int(nx())

col = np.tile([0.45, 0.45, 0.45], (N, 1))  # ungrouped = grey
for _ in range(G):
    nx(); nx()                    # group_type: t
    nx(); npar = int(nx()); nx()  # num_group_parameters: n + label
    params = [float(nx()) for _ in range(npar)]
    nx(); nx()                    # group_label: L
    nx(); r, g, b = float(nx()), float(nx()), float(nx())  # group_color: r g b
    nx(); K = int(nx())           # group_num_point: K
    idx = np.array([int(nx()) for _ in range(K)], dtype=np.int64)
    nx(); nx()                    # num_children: 0
    if K:
        col[idx] = [r, g, b]
print(f"{N} pts, {G} groups")

lo, hi = P.min(0), P.max(0)
ctr = (lo + hi) / 2
r = (hi - lo).max() / 2 + 1e-6
rng = np.random.default_rng(0)
s = rng.choice(N, min(N, 130000), replace=False)

views = [("TOP (floor plan)", 90, -90), ("BOTTOM", -90, -90),
         ("FRONT", 0, -90), ("BACK", 0, 90), ("RIGHT", 0, 0), ("LEFT", 0, 180)]
fig = plt.figure(figsize=(18, 11), facecolor="#111")
for k, (name, el, az) in enumerate(views):
    ax = fig.add_subplot(2, 3, k + 1, projection="3d", facecolor="#111")
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.scatter(P[s, 0], P[s, 1], P[s, 2], c=col[s], s=1.5, marker=".", linewidths=0)
    for a, c in zip("xyz", ctr):
        getattr(ax, f"set_{a}lim")(c - r, c + r)
    ax.view_init(elev=el, azim=az)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(name, color="w", fontsize=12)
fig.suptitle(title, color="w", fontsize=15)
fig.tight_layout()
fig.savefig(png, dpi=110, bbox_inches="tight", facecolor="#111")
print("wrote", png)
