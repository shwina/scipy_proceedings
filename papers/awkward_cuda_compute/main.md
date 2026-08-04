---
title: "GPU-Accelerated Awkward Arrays with CUDA Python"
abstract: |
  Awkward Array is a Python library for manipulating nested,
  variable-length ("ragged") data structures with NumPy-like idioms,
  widely used in High-Energy Physics (HEP) and beyond. Accelerating
  these analyses on GPUs has, until now, required hand-written CUDA
  C++ kernels (compiled at runtime with CuPy): code that is hard to
  write and maintain, difficult to tune for peak performance across
  successive GPU architectures, and awkward to build and package
  because it mixes C++ with Python. This paper reports on a
  collaboration between the Awkward Array and NVIDIA teams to rebuild
  Awkward's GPU backend on `cuda.compute`, a Python library that
  provides state-of-the-art, composable parallel primitives 
  such as reductions, scans, and sorts) directly to Python.
  The new backend expresses GPU algorithms as compositions of
  these primitives so contributors write ordinary Python instead of
  CUDA C++. Most of Awkward's GPU kernels now run
  through `cuda.compute`, and on realistic physics-analysis benchmarks
  the new backend substantially outperforms the hand-written kernels.
  The layout redesign the GPU work required also made Awkward's CPU
  kernels five times faster. Physicists obtain CUDA C++-level GPU
  performance from pure Python.
---

## Introduction

Awkward Array [@awkward; @awkward-scipy2020] is a Python library for manipulating
nested, variable-length data structures (lists of differing lengths, records,
missing values, and unions) using the same vectorized, NumPy-like [@numpy] idioms
scientists already know. It grew out of the High-Energy Physics (HEP) community,
where a single collision event contains a variable number of particles, each with a
variable number of measurements [@awkward-numba], and it is now a foundation of the
Scikit-HEP ecosystem, interoperating with ROOT [@root], Uproot [@uproot], Numba,
Dask [@dask-awkward], JAX, and Apache Arrow.

Modern data analysis increasingly runs on GPUs, but the dominant GPU array and
tensor frameworks are built around dense, rectangular arrays. They offer little for
ragged data: forcing variable-length lists into a rectangular shape requires padding
and masking, and, more fundamentally, the operations a physicist needs (per-event
reductions, combinatorics, selections that depend on the jagged structure itself)
are not expressible as a handful of dense-tensor calls. What is needed is not
another tensor library, but a way to build custom, structure-aware GPU algorithms
while staying in Python.

Awkward has supported GPUs for several years through hand-written CUDA kernels,
compiled at runtime with CuPy [@cupy]. While effective, this approach carries three
persistent costs. First, the kernels are CUDA C++: writing and maintaining them
demands GPU-programming expertise that most contributors (physicists and data
scientists) do not have. Second, hand-written kernels are difficult to keep
performant across successive GPU architectures, each of which introduces new
hardware features; vendor libraries are re-tuned for every generation, but a
project's own kernels are not. Third, mixing C++ with Python complicates building,
packaging and deploying the library. A further missed opportunity is *kernel
fusion* (combining steps so intermediate results stay on-chip), which is key
to obtaining good performance.

This paper reports on a collaboration between the Awkward Array and
NVIDIA teams to rebuild Awkward's GPU backend on `cuda.compute`, a
Python library that brings NVIDIA's CUDA C++ building blocks to
Python. `cuda.compute` exposes the algorithms and iterators of the
CUB and Thrust C++ libraries. These are the same battle-tested,
architecture-tuned primitives that power production GPU software.
The algorithms and iterators are exposed as ordinary Python callables,
that can be composed to build a fused kernel that is tuned to provide the best
possible performance for any given GPU architecture. The result lets a
physicist write idiomatic Python and obtain the performance of
expert-written CUDA.

(sec-background)=
## Background

### Awkward Array Layout

An Awkward array stores ragged data as flat one-dimensional buffers rather than as a
rectangular block. A list-of-lists, for example, is represented by a *content*
buffer holding all of the values contiguously and an *offsets* array marking where
each sublist begins and ends ({numref}`fig-layout`). This keeps the data compact (no
padding is wasted on short lists), but it means every operation has to be written in
terms of these buffers rather than a simple multidimensional shape. That requirement
is exactly what made hand-written GPU kernels laborious.

```{figure} layout.png
:label: fig-layout
:align: center
:width: 59%

Awkward represents a ragged list-of-lists as a flat `content` buffer plus an
`offsets` array that delimits each sublist (here the empty middle list spans no
elements). The same idea, applied recursively, encodes arbitrarily nested data.
```

### The Cost of Hand-Written CUDA Kernels

Awkward's GPU backend has been a dictionary of hand-written CUDA C++ kernels,
compiled and cached on first use through CuPy. Consider `ak.min`, the minimum over
each ragged sublist. Its hand-written GPU implementation takes *three* separate
kernels: one to initialize a scratch buffer, one to perform a within-block reduction
using shared memory and explicit thread synchronization, and one to copy the result
out. Each kernel is launched separately and communicates with the next through GPU
global memory ({numref}`fig-akmin`), and a Python dispatcher selects and
parameterizes the right specialization on top of that.

```{figure} akmin_passes.png
:label: fig-akmin
:align: center
:width: 64%

The hand-written `ak.min` runs as three separately launched kernels that pass
partial results through global memory, with a synchronization point between each. A
contributor must reason about CUDA threads, synchronization, and the ragged buffers
all at once.
```

A contributor fixing a subtle bug in this path must reason simultaneously about CUDA
thread indexing, synchronization, and the ragged buffer arithmetic, and must be
fluent in CUDA C++ to begin with. Multiplied across the backend's more than one
hundred GPU kernels, this is a substantial and specialized maintenance burden, and
it is the burden the new backend is designed to remove.

## cuda.compute

`cuda.compute` is a Python library that brings NVIDIA's CUDA C++ parallel-algorithm
building blocks, CUB and Thrust, directly to Python [@cuda-compute-docs]. A reduction
that would otherwise be hand-written in CUDA C++ becomes a single call, with the
reduction operator written as a plain Python function:

```python
import cupy as cp, numpy as np
from cuda.compute import reduce_into

d_in:  cp.ndarray = cp.arange(1_000_000, dtype=cp.float64)
d_out: cp.ndarray = cp.empty(1, dtype=cp.float64)

def add(a: float, b: float) -> float:
    return a + b

reduce_into(d_in, d_out, add, len(d_in),
            h_init=np.array([0.0]))   # d_out -> 499999500000.0
```

Its key features:

- **Pure Python.** Both the algorithms and the operators they take (here `add`) are
  written in Python; no CUDA C++ is involved.
- **Built on CUB and Thrust.** The underlying implementations are the
  state-of-the-art, composable primitives (reductions, scans, sorts, transforms) used
  throughout production GPU software, and they are specialized and re-tuned for each
  GPU architecture as new hardware features (such as the Tensor Memory Accelerator)
  appear [@cub].
- **JIT-compiled.** User operators are compiled to device code with Numba CUDA and
  cached, so the compilation cost is paid once per specialization.
- **Composable.** Algorithms accept *iterators* that compute their elements lazily
  during execution, fusing steps that would otherwise require separate CUDA kernel
  launches.

### Just-in-Time and Ahead-of-Time Compilation

Because the operator and its data types are only known at run time, `cuda.compute`
compiles on first use. This keeps the library a very small dependency (nothing
pre-compiled has to be shipped) and lets the generated code be specialized to the
operator actually supplied, but it puts compilation on the critical path of the first
call. On an NVIDIA RTX PRO 6000 Blackwell Server Edition (the machine used for every
measurement reported in this paper), a first `unary_transform` costs 1211 ms, of which
almost all is compilation; subsequent calls hit the cache and take 0.23 ms.

For workloads where even the first call must be fast, `cuda.compute` supports an
ahead-of-time workflow: an algorithm is built once, `serialize`d to a blob (230 KB for
the transform above), and `deserialize`d in a later session. The deserialized algorithm
skips compilation entirely, so the first call costs 3.1 ms rather than 1211 ms, and the
cached call 0.14 ms. An interactive session may touch a given specialization only a
handful of times, so that first call is most of the cost.

### Kernel Fusion

Iterators are what make kernel fusion possible: composing an algorithm with an
iterator folds an extra map, gather, or post-processing step *into* the algorithm's
own pass, so intermediate values stay on-chip instead of being written out and read
back [@cuda-compute-blog]. The simplest case is a reduction over a *function* of the
input: Awkward's `min_range` kernel pairs `stops` and `starts` with a `ZipIterator`,
subtracts them lazily through a `TransformIterator`, and reduces, so the per-row lengths
are never materialized. A more involved case is locating the position of an extremum (`argmin`/`argmax`), which fuses the per-row search and
the conversion to a global index into one pass: a `CountingIterator` enumerates the
rows, and a single transform maps each row to the global index of its extremum, so no
intermediate index array is ever built:

```python
import numpy as np
from cuda.compute import CountingIterator, unary_transform

def row_argmax(row: int) -> int:
    lo, hi = starts[row], stops[row]
    if lo == hi:                                # empty list -> sentinel
        return -1
    return np.argmax(content[lo:hi]) + lo       # local argmax -> global index

unary_transform(CountingIterator(index_dtype(0)), result, row_argmax, nrows)
```

This assigns one thread to each sublist, which is the right trade-off when sublists are
short, as they typically are in HEP data (a handful of particles per event). For long
sublists the work per thread becomes the bottleneck, and the better formulation is a
`segmented_reduce` returning the global index of the extremum followed by a
`lower_bound` to convert it back to a per-row offset. Choosing between the two is a
matter of composing different primitives, not of writing a different kernel.

In each case several logical steps collapse into a single kernel, which is key to good
performance.

### Fusing a Whole Analysis Expression

The same mechanism scales up from a single operation to an entire physics formula. The
opposite-sign di-muon invariant mass, a canonical HEP reconstruction, is one line of
Awkward:

```python
mu1, mu2 = ak.unzip(ak.combinations(muons, 2))
mass = np.sqrt(2 * mu1.pt * mu2.pt
               * (np.cosh(mu1.eta - mu2.eta) - np.cos(mu1.phi - mu2.phi)))
```

Evaluated eagerly, each arithmetic and trigonometric operation is one or more separate
kernels, and every intermediate is a full-length array written to and read back from
global memory. Written with `cuda.compute`, the whole formula becomes the operator of a
single `binary_transform`: a `gpu_struct` packs the four muon fields, a `ZipIterator`
combines them and a `PermutationIterator` gathers each pair, all lazily, so the operator
sees whole muons while nothing is materialized.

Profiling both versions with Nsight Systems ({numref}`fig-dimuon`) shows what this buys.
On all muon pairs from a CMS open-data sample (`SMHiggsToZZTo4L`, replicated 100 times to
reach a representative size), the eager version issues 88 kernel launches and 212 memory
operations over a 25.7 ms GPU timeline; the fused version issues **one** kernel and 45
memory operations over 10.2 ms, a 2.5x speedup with no intermediate arrays allocated at
all.

```{figure} dimuon_timeline.png
:label: fig-dimuon
:align: center
:width: 64%

Nsight Systems GPU timelines for the di-muon invariant mass over the same data. The eager
Awkward path (top) is 88 kernels interleaved with 212 memory operations; the
`cuda.compute` path (bottom) is a single fused kernel with 45 memory operations, 2.5x
faster end to end.
```

## The New Awkward GPU Backend

The new backend leaves Awkward's user-facing API unchanged and replaces the
hand-written CUDA layer with `cuda.compute` calls ({numref}`fig-architecture`).
Operations that previously required bespoke CUDA C++
are now written in Python, and the work of generating efficient device code (including
fusion and per-architecture tuning) is delegated to `cuda.compute`.

```{figure} architecture.png
:label: fig-architecture
:align: center
:width: 59%

The dispatcher behind Awkward's unchanged `ak.Array` API routes GPU work to the new
pure-Python `cuda.compute` backend instead of the legacy hand-written CUDA C++ path.
There, the user-supplied operators and iterators are JIT-compiled to LTO-IR while
`cuda.compute` independently compiles its CUB/Thrust kernels to LTO-IR; the two are
JIT-linked into a single fused CUDA kernel that runs on the GPU.
```

### From CUDA C++ to Python: `ak.min`

The three-kernel `ak.min` of {numref}`fig-akmin` becomes a few lines of Python:

```python
import cupy as cp, numpy as np
from cuda.compute import OpKind, segmented_reduce

def awkward_reduce_min(
    toptr: cp.ndarray, fromptr: cp.ndarray, offsets: cp.ndarray,
    outlength: int, identity: float,
) -> None:

    toptr[:outlength] = identity
    segmented_reduce(
        fromptr, toptr, offsets[:-1], offsets[1:],
        OpKind.MINIMUM, np.asarray(identity, dtype=fromptr.dtype), outlength)
```

There is no CUDA C++, no manual synchronization, and no scratch-buffer
bookkeeping: the library handles boundary conditions, temporary storage, and fusion.
The same pattern applies across the backend, and the operator (`min_op`) is an
ordinary Python function that can be read and tested without a GPU.

### Replacing C++ with Python

Counting the code that must actually be maintained, the GPU backend is now mostly Python
({numref}`fig-loc`). We count a kernel's CUDA C++ as maintained only while it is still
dispatched as CUDA C++, treat the generated kernel-signature table as machine-produced
rather than hand-written, and include the shared `cuda_common.cu` scaffolding (625 lines,
unchanged) on both sides. On that basis, Awkward 2.8.11 carried 8,288 lines of
hand-written CUDA C++ against 278 lines of Python. In Awkward 2.10.0 the CUDA C++ still
dispatched is 2,170 lines, a 74% reduction, while the `cuda.compute` implementations and
their dispatch come to 4,663 lines of Python. A further 4,013 lines of superseded `.cu`
files remain in the tree pending removal; they are dead weight rather than upkeep, which
is why they are excluded. The CUDA C++ a contributor must read has fallen by nearly
three-quarters, and what replaced it can be read and tested without a GPU.

```{figure} loc_comparison.png
:label: fig-loc
:align: center
:width: 47%

GPU kernel code that must be maintained, by language and Awkward version, counted from
the source of each release. Before `cuda.compute` the backend was almost entirely CUDA
C++ (8,288 lines against 278 of Python); in Awkward 2.10.0 the C++ still dispatched has
fallen 74% to 2,170 lines and the backend is mostly Python.
```

## Results

All GPU measurements reported below were taken on an NVIDIA RTX PRO 6000 Blackwell
Server Edition (GB202, compute capability 12.0, 98 GB) with CUDA 13.2, against
`cuda.compute` 1.1.0 and CuPy 14.1.1. Source-derived counts (kernel coverage and lines of
code) are for the released Awkward 2.10.0; the end-to-end query benchmark builds Awkward
from `main`, which carries migration work not yet in a release. Every benchmark below was
run twice, on independent machines of the same class and in independently built
environments. All structural quantities (kernel counts, memory-operation counts,
serialized-blob size) came out identical, and every timing agreed to within a few percent,
so the figures quoted here are stable rather than single-shot.

### Kernel Coverage

The migration of the GPU backend to `cuda.compute` is tracked publicly in the project's
issue tracker [@awkward-issue-3793]. The figures below are counted from the released
source rather than from the tracker: a kernel counts as migrated when the CUDA backend's
dispatch table maps its name to a `cuda.compute` implementation. On that measure, Awkward
2.10.0 routes 106 of its 133 GPU kernels (80%) through `cuda.compute`
({numref}`tbl-coverage`), up from none in 2.8.11, which had 131 hand-written kernels.

Every reduction (`sum`, `prod`, `min`, `max`, `argmin`, `argmax`, `count`) and the sort
itself now run through `cuda.compute`, with no hand-written implementation remaining. The
27 kernels still dispatched as CUDA C++ are structural rather than numerical: alongside a
few supporting kernels that build groupings and sorting ranges, they are the jagged
`getitem` paths
(`ListArray_getitem_next_*`, `ListArray_getitem_jagged_*`), padding and validity checks,
`RegularArray` combinations, and the `UnionArray` flatten/fill operations. These are the
cases whose control flow depends on the ragged structure itself, and so map least
naturally onto the segmented primitives; further migration work is in progress upstream.

```{table} GPU kernel coverage, counted from the dispatch table of each release. The reductions and sort run exclusively through cuda.compute; what remains in CUDA C++ is structural.
:label: tbl-coverage

| Category | 2.8.11 | 2.10.0 |
|---|---|---|
| GPU kernels with a CUDA implementation | 131 | 133 |
| Running through `cuda.compute` | 0 | 106 (80%) |
| Still dispatched as CUDA C++ | 131 | 27 |
```

### A Migrated Kernel: `ak.argmin`

Taking `ak.argmin` over ragged data as a representative migrated reduction, the
`cuda.compute` implementation is both shorter and faster than the hand-written kernel it
replaces. Over 5,000,000 sublists (37.5M elements in total, 30 repetitions),
the hand-written CUDA kernel takes 0.96 ms per call and the `cuda.compute` backend
0.39 ms, a 2.5x speedup, with the two producing identical output
({numref}`fig-argmin-bench`). The gain comes from the underlying CUB implementation
rather than from anything Awkward-specific: segmented reductions are difficult to write
well by hand, and CUB's have been tuned per architecture for years.

```{figure} bench_ak_argmin.png
:label: fig-argmin-bench
:align: center
:width: 61%

`ak.argmin` over ragged data with 5,000,000 sublists: the `cuda.compute` backend is 2.5x
faster than the hand-written CUDA kernel, with identical results.
```

### End-to-End Analysis Benchmarks

To gauge the impact on realistic workloads rather than microbenchmarks, we ran the
ADL (Analysis Description Language) benchmark queries [@adl-benchmarks] on CMS 2012
open data (Run2012B `SingleMu`), using the GPU port of the queries from
[@columnar-gpu]. These canonical HEP analysis tasks range from simple spectra
(missing transverse energy, jet $p_T$) to combinatoric reconstructions (opposite-sign
di-muon invariant mass, trijet selection). The baseline is released Awkward 2.8.11, the
last version before `cuda.compute`, running the hand-written CuPy `RawKernel` backend.
We measure the GPU compute stage at three event counts ({numref}`fig-adl`).

The figure combines two distinct comparisons. For the combinatoric queries Q5 and Q6 it
is a backend substitution: the same Awkward expression evaluated on the new backend. For
the lighter, host-bound queries Q3, Q4 and Q7 (marked † in the figure), the query itself
was rewritten against `cuda.compute`, collapsing a chain of Awkward operations into one
or two primitives (`DeviceSelect` for Q3; `segmented_reduce` with `DeviceSelect` for Q4;
`segmented_reduce` with a `PermutationIterator` $\Delta R$ for Q7). Their gains therefore
indicate what an analyst can achieve using the library directly, rather than what the
backend provides automatically.

```{figure} adl_speedup_panel.png
:label: fig-adl
:align: center
:width: 70%

GPU compute-stage speedup (Awkward 2.8.11 ÷ this work) for the ADL benchmark queries at
three event counts. Q5 and Q6 are a straight backend comparison; Q3, Q4 and Q7 (†) use
`cuda.compute`-native fused rewrites of the query. Q1 and Q2 are omitted because their
compute stage is essentially zero; Q8 is omitted because the baseline fails at every
size while the new backend runs it.
```

The two categories scale in opposite directions. The backend substitutions improve with
problem size, Q5 rising from 60x at 100k events to 422x at 1M and 3634x at 10M and Q6
from 45x to 250x, the `cuda.compute` time remaining approximately constant while the
hand-written implementation grows super-linearly. The fused rewrites decline, Q7 from 32x
to 6.0x and Q4 from 5.9x to 3.3x, their benefit deriving from eliminated launch and
allocation overhead, which is a diminishing fraction of the total as the data grow. Q3 lies between, at 14x to
18x.

The new backend is also more robust. Q8, an `argmin` over jagged data containing empty
sublists, fails on the hand-written backend at every event count, leaving the CUDA
context unrecoverable so that the process aborts at the next allocation; `cuda.compute`
completes it in 0.18 s to 0.35 s. Q8 is therefore absent from {numref}`fig-adl`, no
baseline time being available for comparison.

Three qualifications apply. First, the comparison isolates the GPU compute stage;
measured end to end, the light queries are read-bound and gain only 1.1x to 2.9x, while
the combinatoric queries retain most of their advantage (Q5 1723x, Q6 232x at 10M).
Second, each cell is a single timed run following a warm-up pass, although repeating the
suite on a second machine of the same class reproduced every query to within a few
percent (Q5 at 10M: 3600x and 3634x). Third, Q6 was evaluated in fixed event batches at
the largest size in order to fit device memory, which does not alter the results.

## Discussion

### Why `cuda.compute`?

Other routes to GPU code from Python exist: hand-written CUDA C++, kernel-generation
frameworks, and domain-specific compilers among them. `cuda.compute` fits Awkward
particularly well for three reasons. A fourth, unanticipated one is that adapting to it
improved the library elsewhere: the `parents`-to-`offsets` layout change that the
segmented primitives required also made Awkward's CPU reducers a geometric mean of 5.0x
faster with 3.2x lower peak memory [@awkward-pr-4056].

- **Maintenance.** It is maintained by NVIDIA and its underlying primitives are
  re-tuned for each new GPU architecture. Awkward inherits good performance on new
  hardware almost as soon as that hardware ships, without any change to Awkward's own
  code.
- **Ecosystem.** The underlying CUDA C++ libraries, CUB and Thrust, are used
  pervasively across GPU-accelerated software, so Awkward builds on a heavily
  exercised, well-tested foundation rather than a bespoke one.
- **Fit.** Awkward's GPU operations map cleanly onto the composable primitives
  `cuda.compute` provides, and the library performs the kernel fusion that would
  otherwise have to be written and maintained by hand.

### Lowering the Barrier to Contribution

A stated goal of the Awkward Array project is to let physics and data analysts write
high-performance code in Python without GPU expertise. The hand-written backend
required contributors to understand CUDA thread hierarchies, atomics, and
shared-memory behavior to add or fix a kernel. The new backend asks only for a scalar
operator written in Python and a call to the appropriate `cuda.compute` primitive.
That the GPU code is now *pure Python* (readable, reviewable, and testable by the
domain scientists who use the library) is as significant as any single performance
number.

### Limitations

`cuda.compute` is still maturing, and 17 of Awkward's structural kernels are implemented
but not yet merged. Just-in-time compilation adds latency on the first use of a given
specialization, which is noticeable in short interactive sessions; the serialization
workflow described above reduces this from roughly a second to a few milliseconds, but it
requires the application to decide in advance which specializations to build and ship.

A further limitation is that the fusion demonstrated here is *within* an operation. A
chain of Awkward calls still materializes an array between each pair of operations,
unless the analyst drops down to `cuda.compute` and composes the iterators by hand, as in
the di-muon example. Removing that requirement is the subject of the next section.

## Future Work

### Lazy Evaluation and Whole-Expression Fusion

The most significant direction under development is a lazy execution layer that performs
the fusion automatically [@awkward-issue-4141; @awkward-pr-4173]. Wrapping an array with
`ak.cuda.lazy` defers evaluation; operations build an expression graph rather than
launching kernels; and a terminal `compute(fuse=True)` lowers a whole chain of
element-wise operations into a single `cuda.compute` kernel:

```python
la = ak.cuda.lazy(array)             # nothing runs yet

expr = la
for _ in range(16):                  # a chain of elementwise ops
    expr = expr * 1.001 + 0.5

expr.compute(fuse=True)              # the whole chain -> ONE kernel
```

Structural operations such as `filter` act as fusion boundaries, so the graph is partitioned into maximal element-wise regions
rather than requiring the whole expression to be fusible. A subexpression used by more
than one consumer is computed once and shared, which yields common-subexpression
elimination without a separate pass. And `fuse=False` runs the same graph through a
per-operation interpreter with numerically identical results, so fusion is a fast path
rather than a correctness dependency; anything the compiler cannot fuse (strings, indexed
layouts, mixed backends) falls back to the eager path automatically. The decisions are
inspectable through `fusion_stats()` and `visualize(fused=True)`, which report how many
element-wise nodes collapsed into how many fused regions.

The performance figures published with that work are substantial (up to roughly 90x on
the GPU for a long element-wise chain, and 2x to 8x on the CPU), and the GPU speedup is
reported to be largely independent of dataset size, consistent with the win coming from
launch dispatch rather than bandwidth. These are the upstream authors' measurements
rather than ours, and we quote them as such; independently reproducing them across
architectures is future work.

### Other Directions

Several other directions extend naturally. The most immediate is **finishing the
migration** of the remaining structural kernels so that the GPU backend is entirely free
of hand-written CUDA C++. For the small inner lists common in HEP data (a handful of
elements per list), **warp- and block-level cooperative algorithms** can be more
efficient than device-wide ones; `cuda.coop` [@cuda-coop] brings exactly these
cooperative primitives to Python and is a natural next step. Finally, **broader use of
ahead-of-time serialization** across Awkward's common dtype and operator combinations
would remove first-use compilation latency from interactive analysis entirely.

## Conclusion

We have presented a collaboration between the Awkward Array and NVIDIA teams that
rebuilds Awkward's GPU backend on `cuda.compute`. The change replaces hand-written
CUDA C++ kernels with compositions of Python-callable primitives drawn from CUB and
Thrust, letting the library handle fusion and per-architecture tuning. Most of Awkward's
GPU kernels, 106 of 133, now run through `cuda.compute` (every reduction and sort
exclusively so), the hand-written CUDA C++ still dispatched has fallen by 74%, and what
remains in C++ is structural rather than numerical.

The abstraction did not cost performance. A migrated
reduction is 2.5x faster than the kernel it replaces; a di-muon mass reconstruction that
took 88 kernel launches becomes a single fused kernel, 2.5x faster with no intermediate
allocations; and on the ADL analysis queries the new backend outperforms its predecessor
by margins that grow with problem size, reaching 3634x on the di-muon query at 10M events,
while also completing queries the old backend could not run at all. The layout redesign
that the GPU work required even made Awkward's CPU kernels five times faster.

Most importantly, the GPU code is now pure Python: high-level abstraction and
hardware-class performance need not be in tension when the underlying library understands
both the hardware and the problem.

## Acknowledgements

This work was supported in part by NSF grants OAC-1450377, OAC-1836650,
OAC-2103945, PHY-2121686, and PHY-2323298. The authors thank Maxym Naumchyk, who
implemented much of the kernel migration described here, the `cuda.compute` and
CUB/Thrust developers at NVIDIA, and the Scikit-HEP community.

Portions of this work were assisted using a generative AI tool (Claude, by
Anthropic). The tool was used for drafting and refining text and for code and
figure assistance. All outputs were reviewed, verified, and revised by the
authors, who take full responsibility for the accuracy and integrity of the final
content.
