# REPRO.md — reproducing this deck's results (agent guide)

This is the presenter section of the SciPy 2026 talk **"GPU-Accelerated Awkward
Arrays with CUDA Python"** (Ashwin Srinath's half; Ianna Osborne's half is
interleaved — see [Co-authored slides](#co-authored-slides-critical)). The deck
is generated from scripts and quotes measured GPU numbers. This file tells an
agent everything needed to reproduce those numbers and rebuild the deck.

You are almost always in the deck dir:
`.../scipy_proceedings/presentations/slides/awkward-cuda-compute/marp/` (call it `MARP`).

---

## 0. TL;DR — the whole reproduction

On a GPU box (defaults target a **B200**; the dev box was an RTX 6000 Ada):

```bash
# A) deck micro-benchmarks (fusion, ak.argmin, dimuon, JIT/AoT)
cd $MARP
DECK_MACHINE=B200 CUDA_VISIBLE_DEVICES=<gpu> ./reproduce.sh
# -> results/B200/{numbers.json, dimuon_timeline.json, bench_ak_argmin.json}

# B) ADL query benchmark (separate repo)
cd ~/columnar_gpu_bench
CUDA_VISIBLE_DEVICES=<gpu> CCCL_SRC=/path/to/cccl/python/cuda_cccl ./reproduce_adl.sh
# -> columnar_gpu/plots/benchmark_cudf_rerun.png + logs/*
cd $MARP
DECK_MACHINE=B200 COLUMNAR_GPU=~/columnar_gpu_bench/columnar_gpu python3 scripts/integrate_adl.py
# -> copies the chart into figs/, fills results/B200/numbers.json:adl.*

# C) point the deck at B200 and rebuild
sed -i 's/DEFAULT_MACHINE = "RTX_A6000"/DEFAULT_MACHINE = "B200"/' scripts/deck_config.py
python3 scripts/mk_dimuon_timeline.py && python3 scripts/mk_bench_ak_argmin.py
python3 scripts/gen_slides.py
CHROME_PATH=<chrome> marp --no-stdin --pdf --allow-local-files --browser chrome slides.md -o slides.pdf
```

The paper references **only B200**. `RTX_A6000/` is the dev archive.

---

## 1. How the deck is built (data flow)

```
results/<MACHINE>/numbers.json   ─┐
results/<MACHINE>/*.json          ├─> scripts/gen_slides.py ──> slides.md ──(marp)──> slides.pdf
figs/*.png (from scripts/mk_*.py) ─┘         ▲
                                             │ splices ONLY Ashwin's section
                                    scripts/deck_config.py  (DEFAULT_MACHINE / $DECK_MACHINE)
```

- **`scripts/deck_config.py`** — single switch: `DEFAULT_MACHINE` (or env `DECK_MACHINE`)
  selects `results/<MACHINE>/`. `numbers()` loads the scalars; `data(name)` gives a path.
  A missing `numbers.json` raises a clear error (never builds with wrong data).
- **`scripts/gen_slides.py`** — builds Ashwin's slides. Reads machine numbers from
  `deck_config.numbers()`. See [Co-authored slides](#co-authored-slides-critical).
- **`scripts/mk_*.py`** — one figure each; `mk_dimuon_timeline.py` and
  `mk_bench_ak_argmin.py` read `results/<MACHINE>/*.json` via `deck_config`.
- **`results/<MACHINE>/`** — the per-machine truth: `numbers.json`,
  `dimuon_timeline.json`, `bench_ak_argmin.json`, `device.txt`.

### numbers.json schema
```json
{
  "device": "…",
  "fusion_abssum": { "eager_cupy":{"kernels":3,"ms":"2.9"},
                     "torch_compile":{"kernels":2,"ms":"0.93"},
                     "cuda_compute":{"kernels":2,"ms":"0.93"} },
  "aot":    { "jit_first_ms":"…","jit_cached_ms":"…","blob_kb":"…",
              "deser_first_ms":"…","deser_cached_ms":"…" },
  "dimuon": { "speedup_x":"2.9" },
  "adl":    { "max_compute_speedup_x":"1509","light_low_x":"4","light_high_x":"44" },
  "argmin": { "before_ms":"1.424","after_ms":"0.659" }
}
```
All values are strings (rendered verbatim). `collect_results.py` writes everything
except `adl.*` (that comes from `integrate_adl.py`); anything it can't measure it
writes as `"TODO"`.

---

## 2. Part A — deck micro-benchmarks (`reproduce.sh`)

`reproduce.sh` builds a self-contained **uv** venv (awkward lives in the env) and runs
`scripts/collect_results.py`, which drives:

| benchmark | script | writes |
|---|---|---|
| fusion abs-sum (cupy / torch.compile / cuda.compute) | `scripts/fusion/bench_fusion_timing.py {cc,torch}` | `fusion_abssum` |
| JIT vs ahead-of-time | `scripts/measure_aot.py` | `aot` |
| ragged `ak.argmin` before/after | `scripts/bench_ak_argmin.py` | `bench_ak_argmin.json` + `argmin` |
| dimuon mass 88 kernels → 1 (nsys) | `scripts/dimuon/mass_{awkward,cuda_compute}.py` | `dimuon_timeline.json` + `dimuon` |

Env vars: `DECK_MACHINE` (folder), `CUDA_VISIBLE_DEVICES` (GPU; default 0, caller picks),
`CUPY_PKG` (`cupy-cuda13x==14.1.1` for CUDA 13, use `cupy-cuda12x` on CUDA 12),
`TORCH_SPEC` (default the **cu128** wheel — Blackwell needs it), `VENV`.

Notes an agent must know:
- **`nsys`** (Nsight Systems) must be on PATH for the dimuon timeline; else that number is `TODO`.
- **torch and cuda.compute must run in separate processes** (they fight over the CUDA
  context; a shared context fails the reduce build with CUDA error 999). `collect_results`
  already runs `bench_fusion_timing.py cc` and `torch` separately.
- The **fusion working set must exceed the GPU L2** or you measure cache, not DRAM
  (`bench_fusion_timing.py` uses 100M float64 = 800 MB for this reason).
- `bench_ak_argmin.py` finds awkward's `cuda_common.cu` **inside the installed package**
  (no source checkout); `awkward_reduce_argmin.cu` (the pre-migration kernel) is in `scripts/`.

---

## 3. Part B — ADL query benchmark (separate repo `columnar_gpu_bench`)

Repo: `~/columnar_gpu_bench` (`github.com/shwina/columnar-gpu-cuda-compute`). It runs the
ADL `q1`–`q8` physics queries on **two Awkward backends** and produces the deck's 3-panel
chart:

- `.venv-awkward3` — awkward built from **main** (awkward3/cuda.compute is merged) + editable
  local `cuda.compute` (`CCCL_SRC`, default `~/cccl/python/cuda_cccl`; falls back to pip
  `cuda-cccl`). This is "our work".
- `.venv-baseline` — released **awkward==2.8.11**, the last version *before* cuda.compute
  (hand-written CuPy RawKernels). This is the baseline.

`./reproduce_adl.sh` (from the repo root) does the whole thing:
1. installs uv; ensures the 3 parquet subsets in `data/` (downloads if the UCSD host is up).
2. **deletes both venvs** and rebuilds from scratch.
3. **cudf is force-installed LAST** in both venvs (the key hack — see below).
4. runs `columnar_gpu/bench_all.sh` × {100k,1M,10M} → `logs/cmp_*.txt`; runs the fused
   Q3/Q4/Q7 rewrites → `logs/fused.txt`; runs `make_speedup_chart.py` → `plots/benchmark_cudf_rerun.png`.

Env vars: `CUDA_VISIBLE_DEVICES` (GPU; PCI order, default 0), `CCCL_SRC`,
`SCALES` ("100k 1M 10M"), `SKIP_BUILD=1`, `SKIP_BENCH=1`.

### The cudf conflict (why cudf is installed last)
`cudf-cu13`'s metadata pins an old `numba-cuda`, which fights cuda.compute's
`cuda-core 1.0.x`. The conflict is **metadata-only** — libcudf is C++ and never touches
numba-cuda at read time. So: build the awkward/cuda.compute stack first, then force cudf on
top with `uv pip install --override cudf_inject_overrides.txt --extra-index-url
https://pypi.nvidia.com --index-strategy unsafe-best-match "cudf-cu13==26.6.*"`. The override
file pins the working versions cudf would otherwise downgrade. **Do not** let a solver install
cudf alongside the rest — it must come last with the override.

### Pull the ADL result into the deck
```bash
cd $MARP
DECK_MACHINE=<machine> COLUMNAR_GPU=~/columnar_gpu_bench/columnar_gpu python3 scripts/integrate_adl.py
```
Copies `benchmark_cudf_rerun.png` → `figs/`, crops the top ("compute-stage speedup") panel →
`figs/adl_speedup_panel.png`, and computes `adl.*` from the logs
(`speedup = baseline.comp / ours.comp`; `ours` = the fused rewrite for Q3/Q4/Q7).

### ADL gotchas
- **`bench_driver.py` emits `"q"` as a string** ("3", or "3c" for fused); `make_speedup_chart.py`
  and `integrate_adl.py` coerce numeric-string→int on load. Keep that coercion.
- **ADL data provenance**: CMS 2012 `Run2012B_SingleMu` open data. The UCSD download URL "is
  not guaranteed to persist" — have `data/pq_subset_{100k,1M,10M}.parquet` ready if it's down.
- Baseline **Q3 (1M) and Q8 crash** — genuine old-backend bugs, expected; the harness isolates
  each query in its own process so a crash doesn't poison the others.
- Single run per cell — treat as indicative. Speedups reproduce within ~single-run variance
  (dev box: 1509× / 4–44× vs an earlier 1485× / 4–40×).

---

## 4. Part C — switch machine & rebuild

`scripts/deck_config.py` → set `DEFAULT_MACHINE = "B200"` (or export `DECK_MACHINE=B200`).
Then regenerate the data-driven figures and the deck:
```bash
python3 scripts/mk_dimuon_timeline.py     # reads results/<MACHINE>/dimuon_timeline.json
python3 scripts/mk_bench_ak_argmin.py     # reads results/<MACHINE>/bench_ak_argmin.json
python3 scripts/gen_slides.py             # reads results/<MACHINE>/numbers.json; SPLICES Ashwin's section
CHROME_PATH=<chrome> marp --no-stdin --pdf --allow-local-files --browser chrome slides.md -o slides.pdf
```
`CHROME_PATH` on the dev box was `/home/coder/.local/chrome/chrome-nosandbox.sh`.

---

## Co-authored slides (CRITICAL)

`slides.md` contains **both** authors' slides: shared frontmatter/title → **Ianna's intro**
(Scientific Data… → From Python to CUDA) → **Ashwin's section** (What is cuda.compute? →
present and future) → **Ianna's outro** (Results → Thank You + appendix).

`gen_slides.py` **does not rewrite the whole file**. It splices only the region between
`<!-- ASHWIN:BEGIN -->` and `<!-- ASHWIN:END -->`, and **refuses to write** if those markers
are missing. Therefore:
- **Never** hand-edit `slides.md` for Ashwin's content — edit `gen_slides.py` and re-run.
- **Never** "regenerate a fresh slides.md" — that would delete Ianna's slides. Treat Ianna's
  part (everything outside the markers) as read-only.
- After building, sanity-check both authors survive: the deck is ~45 pages and still contains
  "Scientific Data" (intro) and "Thank You" (outro).

---

## Editorial rules (match the existing deck)

- **No em/en dashes anywhere** in Ashwin's slides (use commas/colons/"handwritten"). Ianna's
  slides have some — leave them. Check: `pdftotext slides.pdf - | grep -c "—"` should only
  count Ianna's.
- The lazy-execution slide's "~90x" is the **upstream PR's** published number, not our
  measurement — it is intentionally not in `numbers.json` and does not change per machine.
- The `RTX_A6000` folder label is Ashwin's choice; the actual device is the **RTX 6000 Ada**
  (AD102, CC 8.9), *not* the Ampere "RTX A6000". Recorded in `device.txt`.

---

## Environment reference (dev box that produced RTX_A6000/)

- GPU: NVIDIA RTX 6000 Ada Generation (49 GB), CUDA 13.3. With `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  it was index 1 (index 0 was a 4 GB T400 — do not use). On a single-GPU B200, index 0.
- Deck venv key pins: `awkward==2.10.0`, `cuda-cccl==1.0.1`, `cupy-cuda13x==14.1.1`,
  `numba-cuda==0.30.4`, `uproot==5.7.5`, `nvtx==0.2.15`, torch cu128.
- ADL awkward3 env (from `reference/freeze-awkward3.txt`): editable awkward (main) + editable
  local cuda.compute + `cudf-cu13==26.6.0`, `cupy-cuda13x==14.1.1`, `numba-cuda==0.30.2`,
  `cuda-core==1.0.1`, `numpy==2.2.6`, `pyarrow==24.0.0`, coffea @ git `jitters`.

---

## Verification checklist

- [ ] `results/<MACHINE>/numbers.json` has no `"TODO"` (except intentionally-absent lazy).
- [ ] `dimuon` speedup and the timeline figure agree (both from `dimuon_timeline.json`).
- [ ] ADL: `figs/benchmark_cudf_rerun.png` refreshed; slide shows the new `up to Nx`.
- [ ] Deck builds; ~45 pages; Ianna's "Scientific Data" and "Thank You" present.
- [ ] No em/en dashes in Ashwin's section.
- [ ] These are the reproduction drivers (git-add them so they persist):
      `reproduce.sh`, `scripts/{deck_config,collect_results,integrate_adl}.py`, `results/`,
      and `~/columnar_gpu_bench/reproduce_adl.sh`.
