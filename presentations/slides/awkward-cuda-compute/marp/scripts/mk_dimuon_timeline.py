"""Render the ACTUAL nsys GPU timeline for the dimuon mass calculation.
Reads dimuon_timeline.json (real cuda_gpu_trace: per-op start/duration, from
nsys profiling of mass_awkward.py and mass_cuda_compute.py) and draws a Gantt
with Compute / Memory lanes for before vs after, on a shared time axis."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight"})

HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from deck_config import data as _data
data = json.load(open(_data("dimuon_timeline.json")))

INK="#17303a"; RED="#b23b2e"; LRED="#e08b80"; GREEN="#4c8c00"; LGREEN="#8bbf4c"; GREY="#9aa8ad"
US = 1e3  # ns -> us divisor is 1e3; we plot in ms so divide ns by 1e6
def ms(ns): return ns / 1e6

fig, ax = plt.subplots(figsize=(11, 3.4))
xmax = ms(max(d["span_ns"] for d in data.values())) * 1.02

# lanes (y positions), top to bottom
lanes = [
    ("before", "kernel", 3, "Compute",  RED,    "Awkward array ops"),
    ("before", "mem",    2, "Memory",   LRED,   ""),
    ("after",  "kernel", 1, "Compute",  GREEN,  "cuda.compute binary_transform"),
    ("after",  "mem",    0, "Memory",   LGREEN, ""),
]
H = 0.62
for which, kind, y, lane_lbl, color, group in lanes:
    ops = data[which]["ops"]
    for o in ops:
        if (kind == "kernel") != o["kernel"]:
            continue
        w = max(ms(o["dur"]), xmax * 0.0012)   # floor width so single ops stay visible
        ax.add_patch(Rectangle((ms(o["start"]), y - H/2), w, H,
                                facecolor=color, edgecolor="none", alpha=0.85, zorder=3))
    ax.text(-xmax*0.008, y, lane_lbl, ha="right", va="center", fontsize=12.0, color=INK)

# group labels + span/speedup annotations
sp = data["before"]["span_ns"] / data["after"]["span_ns"]
b_span, a_span = ms(data["before"]["span_ns"]), ms(data["after"]["span_ns"])
ax.text(-xmax*0.135, 2.5, "before", ha="center", va="center", fontsize=14.4, fontweight="bold", color=RED, rotation=90)
ax.text(-xmax*0.135, 0.5, "after",  ha="center", va="center", fontsize=14.4, fontweight="bold", color=GREEN, rotation=90)
ax.text(b_span, 3.0, f"  {data['before']['nkern']} kernels + {data['before']['nmem']} memory ops · {b_span:.0f} ms",
        ha="left", va="center", fontsize=12.0, color=RED, fontweight="bold")
ax.text(a_span, 1.0, f"  1 fused kernel + {data['after']['nmem']} memory ops · {a_span:.0f} ms",
        ha="left", va="center", fontsize=12.0, color=GREEN, fontweight="bold")
# span end markers
ax.axvline(b_span, color=RED, ls=":", lw=1, zorder=1)
ax.axvline(a_span, color=GREEN, ls=":", lw=1, zorder=1)

ax.set_xlim(-xmax*0.16, xmax)
ax.set_ylim(-0.7, 3.9)
ax.set_yticks([])
ax.set_xlabel("GPU time (ms)  ·  actual nsys trace", fontsize=12.6, color=INK)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="x", labelsize=9)
ax.set_title(f"Dimuon invariant mass on the GPU  ·  {sp:.1f}x faster, one kernel instead of {data['before']['nkern']}",
             fontsize=15.0, fontweight="bold", color=INK, loc="left", pad=10)

os.makedirs(os.path.join(HERE, "..", "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "..", "figs", "dimuon_timeline.png"))
plt.close(fig)
print("wrote figs/dimuon_timeline.png  (speedup %.1fx)" % sp)
