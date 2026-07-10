# Populating results/B200/  (the numbers the paper references)

The deck and paper read every machine-dependent number and data-driven figure
from `results/<MACHINE>/`. This folder is the B200 set. Run the steps below on a
B200 machine, then copy this whole `results/B200/` folder back.

When it is populated, flip the deck over to it by editing one line in
`scripts/deck_config.py`:

    DEFAULT_MACHINE = "B200"          # was "RTX_A6000"

(or build with `DECK_MACHINE=B200 python3 scripts/gen_slides.py`), then rebuild:

    python3 scripts/mk_dimuon_timeline.py
    python3 scripts/mk_bench_ak_argmin.py
    python3 scripts/gen_slides.py
    CHROME_PATH=<chrome> marp --no-stdin --pdf --allow-local-files --browser chrome slides.md -o slides.pdf

## One command does most of it

From the `marp/` directory on the B200:

    ./reproduce.sh

That's it. `reproduce.sh` installs `uv` (if missing), builds a self-contained
venv with everything needed (awkward and its CUDA kernels live in the env - no
source checkout), runs the fusion, JIT/AoT, ak.argmin, and dimuon benchmarks,
and writes `results/B200/`. Override defaults via env vars (see the top of
`reproduce.sh`): `CUDA_VISIBLE_DEVICES`, `CUPY_PKG` (CUDA 12 vs 13),
`TORCH_SPEC` (the default cu128 wheel supports Blackwell), `VENV`.

Under the hood it runs `scripts/collect_results.py`, which writes:

    results/B200/numbers.json          <- scalar numbers used in slide notes
    results/B200/dimuon_timeline.json  <- per-op nsys trace for the timeline figure
    results/B200/bench_ak_argmin.json  <- before/after for the argmin figure
    results/B200/device.txt

Anything it cannot produce is written as `"TODO"` so it is obvious.

## The ADL benchmark (separate repo: columnar_gpu_bench)

The ADL query speedups come from the `columnar_gpu_bench` repo
(`github.com/shwina/columnar-gpu-cuda-compute`), which compares
awkward3/cuda.compute vs the awkward 2.8.11 RawKernel baseline. It has its own
two-venv + cudf + real-data setup; do NOT fold it into reproduce.sh. On B200:

1. **Run the suite** (follow `columnar_gpu_bench/README.md` "Reproduce" §1-4:
   build `.venv-awkward3` + `.venv-baseline`, apply the combinations JIT patch,
   install cudf 26.6, fetch the three parquet subsets). Then:
   ```bash
   cd columnar_gpu_bench/columnar_gpu
   for s in 100k 1M 10M; do bash bench_all.sh ../data/pq_subset_$s.parquet logs/cmp_$s.txt; done
   # plus the fused Q3/Q4/Q7 rewrites -> logs/fused.txt (see perf/ / NOTES.md)
   python make_speedup_chart.py         # writes plots/benchmark_cudf_rerun.png
   ```

2. **Pull it into the deck** (from `marp/`), which copies the chart, crops the
   headline panel, and fills the `adl.*` numbers in `results/<MACHINE>/numbers.json`:
   ```bash
   DECK_MACHINE=B200 COLUMNAR_GPU=/path/to/columnar_gpu_bench/columnar_gpu \
       python3 scripts/integrate_adl.py
   ```

That is the only benchmark outside `reproduce.sh`, because it needs the ADL
data + the released baseline env.

## Reference: the RTX 6000 Ada numbers.json (format + current values)

See `results/RTX_A6000/numbers.json`. The B200 file must have the same shape:

    fusion_abssum.{eager_cupy,torch_compile,cuda_compute}.{kernels, ms}
    aot.{jit_first_ms, jit_cached_ms, blob_kb, deser_first_ms, deser_cached_ms}
    dimuon.speedup_x
    adl.{max_compute_speedup_x, light_low_x, light_high_x}
    argmin.{before_ms, after_ms}

The lazy-execution slide's "~90x" is the upstream PR's own published figure, not
a measurement of ours, so it is not part of numbers.json and does not change per
machine (leave it as-is unless you re-benchmark lazy fusion on B200).
