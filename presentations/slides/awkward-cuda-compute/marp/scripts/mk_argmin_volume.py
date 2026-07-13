import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "awkward_reduce_argmin.cu")).read()
kernels = [p for p in re.split(r'(?=__global__ void)', src) if p.strip().startswith("__global__")][:3]

OUT = os.path.join(HERE, "..", "figs")
W = 5.4
LH = 0.066                                     # data units per code line
PAD = 0.30
def card_h(k): return 2 * PAD + len(k.rstrip().splitlines()) * LH
H = max(card_h(k) for k in kernels)            # all cards the same size

# back -> front; the biggest kernel (b) is drawn LAST = in front.
# back cards cascade up-and-left so each one's code stays visible.
order = [2, 0, 1]                               # kernel indices, last = front (biggest)
tlx, tly = 3.4, 8.6                             # front card top-left
ddx, ddy = -1.6, 0.24                           # per step further back (up-left)
fig, ax = plt.subplots(figsize=(12.0, 9.9)); ax.set_xlim(0, 12); ax.set_ylim(0, 9.9); ax.axis("off")
n = len(order)
for depth, idx in enumerate(order):
    back_rank = n - 1 - depth                   # front=0, back=n-1
    h = H
    x = tlx + back_rank * ddx
    top = tly + back_rank * ddy
    y = top - h
    front = depth == n - 1
    z = depth * 3
    ax.add_patch(FancyBboxPatch((x + 0.13, y - 0.13), W, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc="#0000001a", ec="none", zorder=z))                    # shadow
    ax.add_patch(FancyBboxPatch((x, y), W, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc="white", ec=("#123f4d" if front else "#aab6bb"),
                                lw=(1.8 if front else 1.2), zorder=z + 1))
    ax.text(x + 0.16, top - 0.18, "\n".join(kernels[idx].rstrip().splitlines()), ha="left", va="top",
            family="monospace", fontsize=3.6, color="#5f7078",
            alpha=(0.85 if front else 0.55), linespacing=1.02, zorder=z + 2)
ax.text(5.0, 0.55, "ak.argmin (CUDA C++ implementation): 3 kernels, ~150 lines",
        ha="center", va="center", fontsize=22, fontweight="bold", color="#123f4d", zorder=100)
fig.savefig(os.path.join(OUT, "argmin_kernels_volume.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("wrote argmin_kernels_volume.png (layered, content-sized)")
