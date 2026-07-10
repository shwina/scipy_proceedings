import sys, math, time, numpy as np, statistics as st
# Working set MUST exceed the GPU L2 (96 MB on AD102) or you measure cache, not DRAM.
# 100M float64 = 800 MB per array -> safely DRAM-bound.
N = 100_000_000

def bench(fn, sync, warmup=20, reps=50):
    for _ in range(warmup): fn()          # JIT / autotune, discarded
    sync()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t) * 1e6)   # us, walltime
    return st.median(ts)

if sys.argv[1] == "cc":
    import cupy as cp
    from cuda.compute import (TransformIterator, ZipIterator, reduce_into,
                              OpKind, Determinism)
    xc = (cp.arange(N, dtype=cp.float64) % 7) - 3.0
    yc = (cp.arange(N, dtype=cp.float64) % 5) - 2.0
    out = cp.empty(1, dtype=cp.float64); h0 = np.zeros(1); G = Determinism.NOT_GUARANTEED
    # iterators built ONCE, reused every call
    absx  = TransformIterator(xc, lambda v: abs(v))
    expxy = TransformIterator(ZipIterator(xc, yc), lambda p: math.exp(p[0] * p[1]))
    def red(it, det=None):
        kw = {} if det is None else {"determinism": det}
        reduce_into(d_in=it, d_out=out, num_items=N, op=OpKind.PLUS, h_init=h0, **kw)
    sync = cp.cuda.Stream.null.synchronize
    rows = [("cupy eager abs (3 kern)",   lambda: cp.abs(xc).sum()),   # unfused baseline
            ("cuda.compute abs (2 kern)", lambda: red(absx)),
            ("cuda.compute abs (1 kern)", lambda: red(absx, G)),
            ("cuda.compute exp (1 kern)", lambda: red(expxy, G))]
else:
    import torch
    xt = (torch.arange(N, device="cuda", dtype=torch.float64) % 7) - 3.0
    yt = (torch.arange(N, device="cuda", dtype=torch.float64) % 5) - 2.0
    @torch.compile
    def abs_c(x): return torch.abs(x).sum()
    @torch.compile
    def exy_c(x, y): return torch.exp(x * y).sum()
    sync = torch.cuda.synchronize
    rows = [("torch eager abs",    lambda: torch.abs(xt).sum()),
            ("torch.compile abs",  lambda: abs_c(xt)),
            ("torch eager exp",    lambda: torch.exp(xt * yt).sum()),
            ("torch.compile exp",  lambda: exy_c(xt, yt))]

print(f"N={N:,}  ({N*8/1e6:.0f} MB/array, DRAM-bound)   walltime, warmup 20, median of 50")
for label, fn in rows:
    print(f"  {label:28s} {bench(fn, sync):8.1f} us")
