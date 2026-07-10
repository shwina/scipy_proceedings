#!/usr/bin/env python3
"""Run every machine-dependent benchmark and populate results/<MACHINE>/.

Usage (on the target machine, e.g. a B200):

    export DECK_MACHINE=B200
    export PY=/path/to/python            # a python with cuda.compute + awkward + torch
    export AWKWARD_SRC=/path/to/awkward   # checkout for the raw argmin kernels
    export CUDA_VISIBLE_DEVICES=0
    python3 scripts/collect_results.py

It writes results/$DECK_MACHINE/{numbers.json, dimuon_timeline.json,
bench_ak_argmin.json}. The ADL figure (benchmark_cudf_rerun.png) comes from a
separate harness -- see results/B200/README.md. Anything a benchmark cannot
produce is left as the string "TODO" in numbers.json so it is obvious.
"""
import json, os, re, subprocess, sys, shutil

HERE   = os.path.dirname(os.path.abspath(__file__))
MARP   = os.path.dirname(HERE)
MACHINE = os.environ.get("DECK_MACHINE") or sys.exit("set DECK_MACHINE (e.g. B200)")
OUT    = os.path.join(MARP, "results", MACHINE)
os.makedirs(OUT, exist_ok=True)
PY     = os.environ.get("PY", sys.executable)
DIM    = os.path.join(HERE, "dimuon")

def run(cmd, env=None, cwd=None, timeout=1200):
    e = dict(os.environ); e.update(env or {})
    print("  $", " ".join(cmd)); sys.stdout.flush()
    r = subprocess.run(cmd, env=e, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr

def ms2(us):  # microseconds string -> milliseconds, 2 sig figs
    try: return f"{float(us)/1000:.2g}"
    except Exception: return "TODO"

# ---------------- fusion abs-sum (cupy / torch.compile / cuda.compute) --------
def collect_fusion():
    print("[fusion] cuda.compute + cupy ...")
    cc = run([PY, os.path.join(HERE, "fusion", "bench_fusion_timing.py"), "cc"])
    print("[fusion] torch ...")
    tt = run([PY, os.path.join(HERE, "fusion", "bench_fusion_timing.py"), "torch"])
    def us(txt, label):
        m = re.search(re.escape(label) + r"\s+([\d.]+)\s*us", txt)
        return m.group(1) if m else None
    cupy = us(cc, "cupy eager abs (3 kern)")
    cccl = us(cc, "cuda.compute abs (2 kern)")
    tc   = us(tt, "torch.compile abs")
    return {
        "eager_cupy":    {"kernels": 3, "ms": ms2(cupy) if cupy else "TODO"},
        "torch_compile": {"kernels": 2, "ms": ms2(tc)   if tc   else "TODO"},
        "cuda_compute":  {"kernels": 2, "ms": ms2(cccl) if cccl else "TODO"},
    }

# ---------------- JIT vs ahead-of-time ---------------------------------------
def collect_aot():
    print("[aot] measure_aot.py ...")
    t = run([PY, os.path.join(HERE, "measure_aot.py")])
    g = lambda p: (re.search(p, t).group(1) if re.search(p, t) else "TODO")
    return {
        "jit_first_ms":   g(r"1st call\s*\(JIT compiles op\)\s*:\s*([\d.]+)\s*ms"),
        "jit_cached_ms":  g(r"2nd call\s*\(cached, process\)\s*:\s*([\d.]+)\s*ms"),
        "blob_kb":        g(r"serialized blob\s*:\s*([\d.]+)\s*KB"),
        "deser_first_ms": g(r"1st call\s*\(loaded, no JIT\)\s*:\s*([\d.]+)\s*ms"),
        "deser_cached_ms":g(r"2nd call\s*\(cached, in process\)\s*:\s*([\d.]+)\s*ms"),
    }

# ---------------- ak.argmin (writes its own json) ----------------------------
def collect_argmin():
    print("[argmin] bench_ak_argmin.py ...")   # finds cuda_common.cu in the installed awkward
    run([PY, os.path.join(HERE, "bench_ak_argmin.py")])
    src = os.path.join(HERE, "bench_ak_argmin.json")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, "bench_ak_argmin.json"))
        d = json.load(open(src))
        return {"before_ms": f"{d['before_ms']:.3f}", "after_ms": f"{d['after_ms']:.3f}"}
    return {"before_ms": "TODO", "after_ms": "TODO"}

# ---------------- dimuon timeline (nsys) -------------------------------------
def _profile(script, out):
    run(["nsys", "profile", "-t", "cuda,nvtx", "--capture-range=nvtx",
         "--nvtx-capture=mass_calculation", "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
         "-o", out, "--force-overwrite", "true", PY, script],
        env={"PYTHONPATH": DIM}, cwd=DIM)

def _trace(rep):
    out = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_trace", "--format", "csv",
                          "--force-export", "true", rep], capture_output=True, text=True).stdout.splitlines()
    import csv
    h = [i for i, l in enumerate(out) if l.startswith("Start (ns)")][0]
    ops = []
    for r in csv.DictReader(out[h:]):
        st = float(r["Start (ns)"].replace(",", "")); du = float(r["Duration (ns)"].replace(",", ""))
        ops.append({"start": st, "dur": du, "kernel": bool(r.get("GrdX")), "name": r["Name"]})
    t0 = min(o["start"] for o in ops)
    for o in ops: o["start"] -= t0
    return ops, max(o["start"] + o["dur"] for o in ops), sum(o["dur"] for o in ops)

def collect_dimuon():
    print("[dimuon] profiling before + after (nsys) ...")
    _profile(os.path.join(DIM, "mass_awkward.py"),      os.path.join(DIM, "mass_awkward"))
    _profile(os.path.join(DIM, "mass_cuda_compute.py"), os.path.join(DIM, "mass_cuda_compute"))
    data = {}
    for rep, lab in [("mass_awkward", "before"), ("mass_cuda_compute", "after")]:
        ops, span, busy = _trace(os.path.join(DIM, rep + ".nsys-rep"))
        nk = sum(o["kernel"] for o in ops)
        data[lab] = {"ops": ops, "span_ns": span, "busy_ns": busy, "nkern": nk, "nmem": len(ops) - nk}
    json.dump(data, open(os.path.join(OUT, "dimuon_timeline.json"), "w"))
    return {"speedup_x": f"{data['before']['span_ns'] / data['after']['span_ns']:.1f}"}

# ---------------- assemble ---------------------------------------------------
def safe(fn, fallback):
    try:
        return fn()
    except Exception as e:
        print(f"  !! {fn.__name__} failed ({type(e).__name__}: {e}); leaving TODO")
        return fallback

def main():
    print(f"Collecting results for MACHINE={MACHINE} -> {OUT}\n  PY={PY}\n")
    dev = safe(lambda: run([PY, "-c", "import cupy as cp;p=cp.cuda.runtime.getDeviceProperties(0);"
               "print(p['name'].decode())"]).strip().splitlines()[-1], "TODO")
    numbers = {
        "device": dev,
        "fusion_abssum": safe(collect_fusion, {"eager_cupy": {"kernels": 3, "ms": "TODO"},
                              "torch_compile": {"kernels": 2, "ms": "TODO"},
                              "cuda_compute": {"kernels": 2, "ms": "TODO"}}),
        "aot": safe(collect_aot, {k: "TODO" for k in
                    ("jit_first_ms", "jit_cached_ms", "blob_kb", "deser_first_ms", "deser_cached_ms")}),
        "argmin": safe(collect_argmin, {"before_ms": "TODO", "after_ms": "TODO"}),
        "dimuon": safe(collect_dimuon, {"speedup_x": "TODO"}),
        "adl": {"max_compute_speedup_x": "TODO", "light_low_x": "TODO", "light_high_x": "TODO",
                "_note": "run the external ADL harness; drop benchmark_cudf_rerun.png into figs/ and fill these"},
    }
    open(os.path.join(OUT, "numbers.json"), "w").write(json.dumps(numbers, indent=2))
    open(os.path.join(OUT, "device.txt"), "w").write(f"{dev}\nmachine label: {MACHINE}\n")
    print(f"\nwrote {OUT}/numbers.json")
    todo = [k for k, v in numbers.items() if "TODO" in json.dumps(v)]
    if todo: print("STILL TODO (fill by hand / external harness):", todo)

if __name__ == "__main__":
    main()
