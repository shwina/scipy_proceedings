#!/usr/bin/env bash
#
# reproduce.sh - one command to produce the deck's benchmark results on a GPU box.
#
# Builds a self-contained uv venv (awkward lives in the env), runs every
# machine-dependent benchmark, and writes results/<MACHINE>/. Copy that folder
# back and point the deck at it (see results/B200/README.md).
#
#   ./reproduce.sh
#
# Override anything via env vars:
#   DECK_MACHINE=B200          which results/<name>/ folder to write (default: B200)
#   CUDA_VISIBLE_DEVICES=0     which GPU
#   VENV=~/.venv-awkward-bench where to build the venv
#   CUPY_PKG=cupy-cuda13x==14.1.1     matches the CUDA 13 toolkit; use cupy-cuda12x on CUDA 12
#   TORCH_SPEC="torch --index-url https://download.pytorch.org/whl/cu128"
#                             cu128 supports Blackwell (B200/sm_100); torch is only used
#                             for the torch.compile comparison and may be skipped if it fails
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # the marp/ dir
cd "$HERE"

MACHINE="${DECK_MACHINE:-B200}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VENV="${VENV:-$HOME/.venv-awkward-bench}"
CUPY_PKG="${CUPY_PKG:-cupy-cuda13x==14.1.1}"
TORCH_SPEC="${TORCH_SPEC:-torch --index-url https://download.pytorch.org/whl/cu128}"

echo "== reproduce.sh =="
echo "   machine label : $MACHINE"
echo "   GPU           : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   venv          : $VENV"
echo

# ---- 1. uv (install if missing) --------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "-- installing uv --"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv still not on PATH"; exit 1; }

# ---- 2. nsys check (needed for the dimuon timeline) ------------------------
if ! command -v nsys >/dev/null 2>&1; then
  echo "WARN: 'nsys' (Nsight Systems) not on PATH -> the dimuon timeline will be left as TODO."
  echo "      It ships with the CUDA toolkit; add it to PATH to get that result."
fi

# ---- 3. venv + dependencies (awkward lives in the env) --------------------
echo "-- building venv --"
uv venv "$VENV" --python 3.12
PY="$VENV/bin/python"

echo "-- installing packages --"
uv pip install --python "$PY" \
    "numpy<2.5" matplotlib requests aiohttp \
    "awkward==2.10.0" "uproot==5.7.5" "nvtx==0.2.15" \
    "cuda-cccl==1.1.0" "numba-cuda==0.30.4" "$CUPY_PKG"
    # numpy<2.5: numba needs <=2.4. cuda-cccl 1.1.0 has the serialize/AoT API.

# torch is only for the torch.compile comparison bar; don't abort if it fails
echo "-- installing torch (optional: torch.compile comparison) --"
# shellcheck disable=SC2086
uv pip install --python "$PY" $TORCH_SPEC \
  || echo "WARN: torch install failed; the torch.compile number will be TODO."

# Blackwell (sm_120) fix: torch's cu128 wheel pulls nvidia-nvjitlink-cu12 (12.8),
# which is too old to link sm_120 code -> cuda.compute custom ops fail with
# nvJitLink 'may need newer version'. Force the CUDA 13 nvjitlink (harmless on
# older archs). No-op if torch wasn't installed.
uv pip install --python "$PY" nvidia-nvjitlink-cu13 >/dev/null 2>&1 || true
uv pip uninstall --python "$PY" nvidia-nvjitlink-cu12 >/dev/null 2>&1 || true

# Same class of problem, and the one that actually bites: torch's cu128 wheel
# depends on cuda-bindings ~12.x, so installing it DOWNGRADES cuda-bindings from
# 13.x to 12.9.x and drags in cuda-toolkit 12.8. cuda.compute then fails to build
# any algorithm on sm_120 ("RuntimeError: Failed to build unary transform") and
# every cuda.compute benchmark below silently records TODO while the cupy and
# torch ones still succeed -- which makes it look like a cuda.compute bug rather
# than an environment one. Restore the CUDA 13 bindings after torch.
uv pip install --python "$PY" "cuda-bindings==13.3.1" >/dev/null 2>&1 || true
uv pip uninstall --python "$PY" cuda-toolkit >/dev/null 2>&1 || true

# Fail fast if the stack is broken, rather than emitting a results file of TODOs.
"$PY" - <<'PYCHK' || { echo "ERROR: cuda.compute cannot build on this GPU; fix the env before trusting results."; exit 1; }
import cupy as cp
from cuda.compute import unary_transform
x = cp.arange(8, dtype=cp.float64); y = cp.empty_like(x)
unary_transform(d_in=x, d_out=y, op=lambda v: v * v, num_items=8)
assert float(y[3]) == 9.0
print("  cuda.compute smoke test OK")
PYCHK

# ---- 4. run the benchmarks -------------------------------------------------
echo
echo "-- running benchmarks (this takes a few minutes) --"
export DECK_MACHINE="$MACHINE"
export PY
"$PY" "$HERE/scripts/collect_results.py"

echo
echo "== done =="
echo "Populated: results/$MACHINE/  (fusion, JIT/AoT, ak.argmin, dimuon)"
echo
echo "Still needed: the ADL query benchmark (separate repo columnar_gpu_bench)."
echo "  Run its suite + make_speedup_chart.py, then from marp/:"
echo "    DECK_MACHINE=$MACHINE COLUMNAR_GPU=/path/to/columnar_gpu_bench/columnar_gpu \\"
echo "        python3 scripts/integrate_adl.py"
echo "  (copies the chart + fills the adl.* numbers). See results/B200/README.md."
echo
echo "Then set DEFAULT_MACHINE in scripts/deck_config.py and rebuild."
