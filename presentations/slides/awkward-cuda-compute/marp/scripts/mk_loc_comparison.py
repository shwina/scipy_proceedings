#!/usr/bin/env python3
r"""loc_comparison.png - GPU kernel code that must be maintained, by language.

Counts are DERIVED FROM SOURCE, not hand-entered, so the figure and the paper's
numbers can always be re-checked. Point it at two installed awkward trees:

    python3 scripts/mk_loc_comparison.py \
        --before /path/to/site-packages/awkward \        # awkward 2.8.11
        --after  /path/to/site-packages/awkward          # awkward 2.10.0

Counting method (stated so it is reproducible and arguable):

  CUDA C++ (maintained)
      Lines of `_connect/cuda/cuda_kernels/awkward_*.cu` for kernels that are
      STILL DISPATCHED as CUDA C++. A kernel counts as migrated when
      `_backends/cupy.py` maps its name to a `cuda_compute.*` implementation.
      In 2.8.11 nothing is migrated, so this is every kernel file.
      `cuda_common.cu` (625 lines of shared hand-written CUDA scaffolding,
      identical in both versions) is included in both, since it is upkeep too.

  Python (maintained)
      Lines of `_connect/cuda/*.py`, EXCLUDING the generated
      `_kernel_signatures.py` (machine-produced, not maintained by hand).

Superseded `.cu` files that remain in the tree but are no longer dispatched are
reported separately: they are dead weight pending removal, not upkeep.
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figs", "loc_comparison.png")

# Palette: validated with the dataviz validator (light surface #fcfcfb) --
# lightness band, chroma floor, CVD separation, normal-vision floor and contrast
# all PASS. Green matches the deck's "new/Python" hue; blue matches the second
# series in cpu_speedup_4056.png.
CPP = "#4a90d9"
PY = "#4c8c00"
INK = "#17303a"
MUTED = "#58717a"
SURFACE = "#ffffff"


def _lines(path):
    with open(path, errors="ignore") as fh:
        return sum(1 for _ in fh)


def measure(root):
    """(cpp_maintained, python_maintained, superseded_cpp, n_migrated, n_cpp)."""
    kernels = {os.path.basename(f)[:-3]: _lines(f)
               for f in glob.glob(os.path.join(
                   root, "_connect", "cuda", "cuda_kernels", "awkward_*.cu"))}

    backend = os.path.join(root, "_backends", "cupy.py")
    migrated = set()
    if os.path.exists(backend):
        with open(backend, errors="ignore") as fh:
            migrated = set(re.findall(
                r'"(awkward_[A-Za-z0-9_]+)"\s*:\s*cuda_compute\.', fh.read()))

    still = {k: v for k, v in kernels.items() if k not in migrated}
    superseded = {k: v for k, v in kernels.items() if k in migrated}

    # shared hand-written CUDA scaffolding: upkeep in both versions
    common = os.path.join(root, "_connect", "cuda", "cuda_kernels", "cuda_common.cu")
    common_lines = _lines(common) if os.path.exists(common) else 0

    py = 0
    for f in glob.glob(os.path.join(root, "_connect", "cuda", "*.py")):
        if os.path.basename(f) == "_kernel_signatures.py":
            continue          # generated, not hand-maintained
        py += _lines(f)

    return (sum(still.values()) + common_lines, py, sum(superseded.values()),
            len(migrated), len(still))


def bar(ax, x, cpp, py, top, width=0.52):
    """One stacked bar: C++ on the bottom, Python on top, 2px surface gap.

    Every segment is direct-labelled: inside when it is tall enough to hold the
    text, otherwise on a leader outside the bar (small segments must still be
    readable -- the contrast WARN relief is visible labels).
    """
    gap = top * 0.005                        # ~2px of surface between fills
    for y0, h, color, name in ((0.0, cpp, CPP, "cpp"), (cpp + gap, py, PY, "py")):
        ax.add_patch(plt.Rectangle((x - width / 2, y0), width, h,
                                   fc=color, ec="none", zorder=3))
        if h > top * 0.08:
            ax.text(x, y0 + h / 2, f"{int(h):,}", ha="center", va="center",
                    color="white", fontsize=13.5, fontweight="bold", zorder=4)
        else:                                # too thin for an inside label
            ax.annotate(f"{int(h):,}", xy=(x + width / 2, y0 + h / 2),
                        xytext=(x + width / 2 + 0.16, y0 + h / 2 + top * 0.045),
                        fontsize=11.5, fontweight="bold", color=INK,
                        va="center", ha="left", zorder=4,
                        arrowprops=dict(arrowstyle="-", color="#b8c2c6", lw=1.1))
    ax.text(x, cpp + py + gap + top * 0.025, f"{cpp + py:,}", ha="center",
            va="bottom", color=INK, fontsize=14, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="awkward package dir (pre cuda.compute)")
    ap.add_argument("--after", required=True, help="awkward package dir (latest)")
    ap.add_argument("--before-label", default="Awkward 2.8\n(before cuda.compute)")
    ap.add_argument("--after-label", default="Awkward 2.10\n(cuda.compute)")
    a = ap.parse_args()

    b_cpp, b_py, b_dead, b_mig, b_still = measure(a.before)
    n_cpp, n_py, n_dead, n_mig, n_still = measure(a.after)

    print(f"before: C++ {b_cpp:,}  Python {b_py:,}  "
          f"(migrated {b_mig}, still C++ {b_still}, superseded .cu {b_dead:,} lines)")
    print(f"after : C++ {n_cpp:,}  Python {n_py:,}  "
          f"(migrated {n_mig}, still C++ {n_still}, superseded .cu {n_dead:,} lines)")
    drop = (b_cpp - n_cpp) / b_cpp * 100
    print(f"C++ to maintain: -{drop:.0f}%")

    top = max(b_cpp + b_py, n_cpp + n_py)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bar(ax, 0, b_cpp, b_py, top)
    bar(ax, 1, n_cpp, n_py, top)

    ax.set_xlim(-0.62, 1.72)
    ax.set_ylim(0, top * 1.20)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([a.before_label, a.after_label], fontsize=11.5, color=INK)
    ax.set_ylabel("Lines of GPU kernel code (maintained)", fontsize=11.5, color=INK)
    ax.set_title("GPU kernel code: less C++ to maintain, mostly Python",
                 fontsize=13.5, fontweight="bold", color=INK, pad=14)

    # recessive axes / grid
    ax.grid(axis="y", color="#e6e6e2", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#d6dade")
    ax.tick_params(axis="both", colors=MUTED, length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=CPP, ec="none"),
               plt.Rectangle((0, 0), 1, 1, fc=PY, ec="none")]
    leg = ax.legend(handles, ["CUDA C++", "Python"], title="language",
                    loc="upper right", frameon=False, fontsize=11.5)
    leg.get_title().set_fontsize(11)
    leg.get_title().set_color(MUTED)
    for t in leg.get_texts():
        t.set_color(INK)

    ax.annotate("", xy=(0.735, n_cpp), xytext=(0.265, b_cpp),
                arrowprops=dict(arrowstyle="->", color="#b8c2c6", lw=1.8,
                                connectionstyle="arc3,rad=-0.22"), zorder=2)
    ax.text(0.48, top * 0.36, f"C++ to maintain\n−{drop:.0f}%", fontsize=11,
            fontweight="bold", color=MUTED, ha="center", va="center")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
