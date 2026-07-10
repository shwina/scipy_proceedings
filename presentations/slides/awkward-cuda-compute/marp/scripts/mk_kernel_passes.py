"""Generic 'separate kernels round-trip through global memory' diagram
for the Kernel fusion intro slide (memory traffic + launch overhead)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight"})
OUT = "/home/coder/scipy_proceedings/presentations/slides/awkward-cuda-compute/marp/figs"

INK="#17303a"; GREY="#8a969b"; LGREY="#eef1f2"; RED="#b23b2e"; LORANGE="#fbe7d6"; DGREY="#5c6a70"

fig, ax = plt.subplots(figsize=(11, 4.9))
ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")

ax.text(7, 8.55, "Separate kernels each round trip through global memory",
        ha="center", fontsize=18.0, fontweight="bold", color=INK)

# ---- three generic kernel boxes ----
boxes = [(0.6, "Kernel 1"), (5.15, "Kernel 2"), (9.7, "Kernel 3")]
BW, BH, BY = 3.7, 1.7, 5.4
for x, label in boxes:
    ax.add_patch(FancyBboxPatch((x, BY), BW, BH, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=LGREY, ec=GREY, lw=2))
    ax.text(x + BW/2, BY + BH/2, label, ha="center", va="center",
            fontsize=18.0, fontweight="bold", color=INK)

# launch + sync between kernels
for xsep in (4.55, 9.1):
    ax.plot([xsep, xsep], [BY-0.35, BY+BH+0.35], ls=(0, (2, 2)), color=GREY, lw=1.4)
    ax.text(xsep, BY+BH+0.72, "launch\n+ sync", ha="center", va="center",
            fontsize=12.0, color=GREY, linespacing=1.15)

# ---- global memory bar ----
MY, MH = 1.0, 1.35
ax.add_patch(FancyBboxPatch((0.6, MY), 12.8, MH, boxstyle="round,pad=0.02,rounding_size=0.10",
                            fc=LORANGE, ec=RED, lw=2))
ax.text(7, MY + MH/2, "GPU global memory", ha="center", va="center",
        fontsize=16.8, fontweight="bold", color=RED)

# ---- write / read round-trip arrows for every kernel ----
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=15, lw=2.2, color=RED, shrinkA=0, shrinkB=0))
for i, (x, _) in enumerate(boxes):
    cx = x + BW/2
    arrow(cx-0.55, BY, cx-0.55, MY+MH)   # write down
    arrow(cx+0.55, MY+MH, cx+0.55, BY)   # read up
ax.text(boxes[0][0]+BW/2-1.35, (BY+MY+MH)/2, "write", ha="right", va="center",
        fontsize=13.2, color=RED, fontweight="bold")
ax.text(boxes[0][0]+BW/2+1.35, (BY+MY+MH)/2, "read", ha="left", va="center",
        fontsize=13.2, color=RED, fontweight="bold")

fig.savefig(f"{OUT}/kernel_passes.png"); plt.close(fig)
print("wrote figs/kernel_passes.png")
