#!/usr/bin/env python3
"""
Benchmark ak.argmin over RAGGED data, two ways:

  BEFORE : Awkward's hand-written CUDA kernel (raw CuPy RawModule; the
           three-pass awkward_reduce_argmin_a/b/c, compiled exactly as
           Awkward compiled it).
  AFTER  : Awkward's cuda.compute backend (unary_transform over a
           CountingIterator of segment ids -- a verbatim copy of
           awkward._connect.cuda._compute.awkward_reduce_argmin).

Both compute the per-sublist argmin (global index) over the same jagged
array, and the script checks that they agree.

Run in a working cuda.compute env:
    python bench_ak_argmin.py

Writes scripts/bench_ak_argmin.json for the slide figure.
"""
import os, json, math, time
import numpy as np
import cupy as cp

# ---- config ----
NUM_SEG   = 5_000_000          # number of sublists (e.g. "events")
MAX_LEN   = 16                 # sublist lengths drawn from [0, MAX_LEN)
REPS      = 30
BLOCK     = 256
HERE      = os.path.dirname(os.path.abspath(__file__))
ARGMIN_CU = os.path.join(HERE, "awkward_reduce_argmin.cu")


def _find_cuda_common():
    """cuda_common.cu ships inside the installed awkward package; a source
    checkout (AWKWARD_SRC) is only a fallback."""
    import awkward
    inst = os.path.join(os.path.dirname(awkward.__file__),
                        "_connect", "cuda", "cuda_kernels", "cuda_common.cu")
    if os.path.exists(inst):
        return inst
    src = os.environ.get("AWKWARD_SRC")
    if src:
        p = os.path.join(src, "src/awkward/_connect/cuda/cuda_kernels/cuda_common.cu")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("cuda_common.cu not found in installed awkward or AWKWARD_SRC")


COMMON_CU = _find_cuda_common()

ERROR_BITS = 8
NO_ERROR   = int(np.iinfo(np.uint64).max)


def build_ragged():
    rng = np.random.default_rng(0)
    lengths = rng.integers(0, MAX_LEN, size=NUM_SEG).astype(np.int64)
    offsets_h = np.empty(NUM_SEG + 1, dtype=np.int64)
    offsets_h[0] = 0
    np.cumsum(lengths, out=offsets_h[1:])
    nelem = int(offsets_h[-1])
    content = cp.asarray(rng.standard_normal(nelem), dtype=cp.float64)
    offsets = cp.asarray(offsets_h)
    parents = cp.asarray(np.repeat(np.arange(NUM_SEG, dtype=np.int64), lengths))
    return content, offsets, parents, nelem, lengths


def timed(fn, reps=REPS, warmup=5):
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    ts = []
    for _ in range(reps):
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


# ---------- BEFORE: raw hand-written kernel ----------
def make_raw_argmin():
    src = (f"#define ERROR_BITS {ERROR_BITS}\n#define NO_ERROR {NO_ERROR}\n"
           + open(COMMON_CU).read() + "\n" + open(ARGMIN_CU).read())
    names = [f"awkward_reduce_argmin_{s}<int64_t, double, int64_t>" for s in ("a", "b", "c")]
    mod = cp.RawModule(code=src, options=("--std=c++11", "--diag-suppress=186"),
                       name_expressions=names)
    return [mod.get_function(nm) for nm in names]


def run_before(kernels, content, parents, nelem, num_seg, toptr, atomic_toptr, temp, err):
    grid = (math.ceil(nelem / BLOCK),)
    block = (BLOCK,)
    args = (toptr, content, parents, np.int64(nelem), np.int64(num_seg),
            atomic_toptr, temp, np.uint64(0), err)
    for k in kernels:
        k(grid, block, args)


# ---------- AFTER: cuda.compute backend (verbatim from awkward) ----------
def run_after_factory(content, offsets, num_seg, result):
    from cuda.compute import CountingIterator, unary_transform
    start_o, end_o = offsets[:-1], offsets[1:]          # make_segment_views

    def segment_reduce_argmin(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = content[start_idx:end_idx]
        if start_idx == end_idx:
            return -1
        return np.argmin(segment) + start_idx           # global index

    seg_ids = CountingIterator(np.int64(0))

    def run():
        unary_transform(d_in=seg_ids, d_out=result, op=segment_reduce_argmin,
                        num_items=num_seg)
    return run


def main():
    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"].decode()
    content, offsets, parents, nelem, lengths = build_ragged()
    empty = int((lengths == 0).sum())
    print(f"ragged: {NUM_SEG:,} sublists, {nelem:,} elements "
          f"(mean len {nelem/NUM_SEG:.1f}, {empty:,} empty)  ·  {name}")

    # BEFORE
    kernels = make_raw_argmin()
    toptr = cp.full(NUM_SEG, -1, dtype=cp.int64)
    atomic_toptr = cp.empty(NUM_SEG, dtype=cp.uint64)
    temp = cp.empty(nelem, dtype=cp.int64)
    err = cp.full(1, NO_ERROR, dtype=cp.uint64)
    def before():
        run_before(kernels, content, parents, nelem, NUM_SEG, toptr, atomic_toptr, temp, err)
    before(); cp.cuda.Stream.null.synchronize()
    res_before = toptr.copy()
    t_before = timed(before)

    # AFTER
    result = cp.full(NUM_SEG, -1, dtype=cp.int64)
    after = run_after_factory(content, offsets, NUM_SEG, result)
    after(); cp.cuda.Stream.null.synchronize()          # warm up (JIT once)
    res_after = result.copy()
    t_after = timed(after)

    agree = bool(cp.array_equal(res_before, res_after))

    out = {"device": name, "num_segments": NUM_SEG, "n_elements": nelem,
           "reps": REPS, "before_ms": t_before, "after_ms": t_after, "agree": agree}
    json.dump(out, open(os.path.join(HERE, "bench_ak_argmin.json"), "w"), indent=2)

    print("=" * 62)
    print(f"  ak.argmin over ragged data   ·   {name}")
    print(f"  {NUM_SEG:,} sublists / {nelem:,} elements   (median of {REPS}, warm)")
    print(f"  before/after agree: {agree}")
    print("-" * 62)
    print(f"  BEFORE  hand-written CUDA kernel      : {t_before:11.3f} ms")
    print(f"  AFTER   cuda.compute backend          : {t_after:11.3f} ms")
    print("-" * 62)
    print(f"  speedup (before / after)              : {t_before / t_after:8.2f}x")
    print("=" * 62)


if __name__ == "__main__":
    main()
