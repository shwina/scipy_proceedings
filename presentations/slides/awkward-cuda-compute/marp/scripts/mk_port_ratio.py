"""Port-ratio figure for the 'present & future' slide.
Recomputed from scikit-hep/awkward issue #3793: 103 of 120 kernel-migration
sub-tasks complete = 85.8%. High-priority tier (reductions & sort) is 100%."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight"})
OUT = "/home/coder/scipy_proceedings/presentations/slides/awkward-cuda-compute/marp/figs"

INK="#17303a"; GREEN="#4c8c00"; LGREEN="#e4f2d0"; GREY="#c3ccd0"; DGREY="#7c8a90"; RED="#b23b2e"
DONE, TOTAL = 103, 120
REMAIN = TOTAL - DONE
HEADLINE = "85%"   # 103/120 = 85.8%, reported as 85% to match the deck

fig, ax = plt.subplots(figsize=(9.2, 3.5))
ax.set_xlim(0, TOTAL); ax.set_ylim(0, 10); ax.axis("off")

# big headline
ax.text(0, 9.4, HEADLINE, fontsize=55.2, fontweight="bold", color=GREEN, va="top")
ax.text(43, 8.55, "of Awkward's GPU kernels\nnow run as pure Python", fontsize=16.2,
        color=INK, va="top", linespacing=1.25)

# the bar: 0..DONE python (green), DONE..TOTAL still C++ (grey)
y, h = 3.0, 1.8
ax.add_patch(FancyBboxPatch((0, y), DONE, h, boxstyle="round,pad=0,rounding_size=0.25",
                            fc=GREEN, ec="white", lw=1.5, zorder=3))
ax.add_patch(FancyBboxPatch((DONE, y), REMAIN, h, boxstyle="round,pad=0,rounding_size=0.25",
                            fc=GREY, ec="white", lw=1.5, zorder=3))
ax.text(DONE/2, y+h/2, f"{DONE}  cuda.compute (Python)", ha="center", va="center",
        color="white", fontsize=15.6, fontweight="bold", zorder=4)
ax.text(DONE + REMAIN/2, y+h/2 + 2.15, f"{REMAIN} CUDA C++", ha="center", va="bottom",
        color=DGREY, fontsize=13.2, fontweight="bold")
ax.annotate("", xy=(DONE + REMAIN/2, y+h+0.05), xytext=(DONE + REMAIN/2, y+h+1.9),
            arrowprops=dict(arrowstyle="-", color=DGREY, lw=1))

# caption
ax.text(0, 1.35, "103 of 120 kernel migrations complete  ·  reductions, scans and sorts: 100%",
        fontsize=12.6, color=DGREY)
ax.text(0, 0.5, "scikit-hep/awkward  issue #3793", fontsize=11.4, color=GREY, style="italic")

fig.savefig(f"{OUT}/port_ratio.png"); plt.close(fig)
print(f"wrote port_ratio.png  ({DONE}/{TOTAL} = {DONE/TOTAL*100:.1f}%, shown as {HEADLINE})")
