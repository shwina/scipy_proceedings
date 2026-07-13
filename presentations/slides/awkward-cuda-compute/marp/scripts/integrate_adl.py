#!/usr/bin/env python3
"""Pull the ADL benchmark result into the deck.

The ADL queries are a separate benchmark (repo `columnar_gpu_bench`, which
compares awkward3/cuda.compute vs the awkward 2.8.11 RawKernel baseline). This
script takes that repo's output and produces everything the deck needs:

  1. copies its 3-panel chart          -> figs/benchmark_cudf_rerun.png
  2. crops the headline panel          -> figs/adl_speedup_panel.png
  3. reads its logs and fills the ADL  -> results/<MACHINE>/numbers.json
     numbers (max compute-stage speedup, light-query range)

Prereq: run the ADL suite first so the logs + chart exist. From the
columnar_gpu_bench repo (see its README):
    cd columnar_gpu
    for s in 100k 1M 10M; do bash bench_all.sh ../data/pq_subset_$s.parquet logs/cmp_$s.txt; done
    python make_speedup_chart.py            # writes plots/benchmark_cudf_rerun.png

Then, from marp/:
    DECK_MACHINE=B200 COLUMNAR_GPU=/path/to/columnar_gpu_bench/columnar_gpu \
        python3 scripts/integrate_adl.py
"""
import json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deck_config import MACHINE, RESULTS

MARP = os.path.dirname(HERE)
CG = os.environ.get("COLUMNAR_GPU",
                    os.path.join(os.path.expanduser("~"), "columnar_gpu_bench", "columnar_gpu"))
LOGS = os.path.join(CG, "logs")
CHART = os.path.join(CG, "plots", "benchmark_cudf_rerun.png")
SCALES = ["100k", "1M", "10M"]
FUSED_Q = {3, 4, 7}


def _adl_numbers():
    """Replicate make_speedup_chart's compute-stage speedup = baseline/ours."""
    def load(s):
        r = {}
        for ln in open(os.path.join(LOGS, f"cmp_{s}.txt")):
            ln = ln.strip()
            if ln.startswith("{"):
                d = json.loads(ln); q = d["q"]
                q = int(q) if isinstance(q, str) and q.isdigit() else q   # bench_driver emits str
                r.setdefault(q, {})[d["label"]] = d
        return r
    data = {s: load(s) for s in SCALES}
    fused = {}
    for ln in open(os.path.join(LOGS, "fused.txt")):
        ln = ln.strip()
        if ln.startswith("{"):
            d = json.loads(ln); fused[(d["scale"], int(d["q"][:-1]))] = d
    ours = lambda s, q: fused[(s, q)]["comp"] if q in FUSED_Q else data[s][q]["awkward3"]["comp"]
    allsp, light = [], []
    for s in SCALES:
        for q in (3, 4, 5, 6, 7):
            b = data[s][q].get("baseline"); ov = ours(s, q)
            if b and b.get("ok") and ov > 0:
                sp = b["comp"] / ov; allsp.append(sp)
                if q in (3, 4, 7): light.append(sp)
    return {"max_compute_speedup_x": f"{max(allsp):.0f}",
            "light_low_x": f"{min(light):.0f}", "light_high_x": f"{max(light):.0f}"}


def _crop_panel(src, dst, y0, y1):
    from PIL import Image
    im = Image.open(src); W, _ = im.size
    im.crop((0, y0, W, y1)).save(dst)


def main():
    if not os.path.exists(CHART):
        sys.exit(f"missing {CHART}\nRun the ADL suite + make_speedup_chart.py first "
                 f"(set COLUMNAR_GPU to the columnar_gpu dir). See this script's docstring.")
    figs = os.path.join(MARP, "figs")
    shutil.copy(CHART, os.path.join(figs, "benchmark_cudf_rerun.png"))
    _crop_panel(CHART, os.path.join(figs, "adl_speedup_panel.png"), 82, 972)     # panel A: compute-stage
    _crop_panel(CHART, os.path.join(figs, "adl_e2e_panel.png"), 1028, 1878)       # panel B: end-to-end
    print("copied figs/benchmark_cudf_rerun.png, adl_speedup_panel.png, adl_e2e_panel.png")

    adl = _adl_numbers()
    npath = os.path.join(RESULTS, "numbers.json")
    nums = json.load(open(npath)) if os.path.exists(npath) else {}
    nums.setdefault("adl", {}).update(adl)
    json.dump(nums, open(npath, "w"), indent=2)
    print(f"updated {npath}  adl = {adl}")


if __name__ == "__main__":
    main()
