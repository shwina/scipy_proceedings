# Why the dimuon fusion speedup was "only" ~3x (investigation)

Ruled OUT: L2 cache / small problem size.
- Working set at the talk size (concat x100) is ~950 MB, FAR past the 96 MB L2.
- Speedup is size-independent: 3.48x at 950 MB, 3.12x at 3.8 GB (bench_sweep_size.py).

Root cause (Nsight Compute on the fused transform_kernel):
- DRAM throughput 8% of peak; SM (compute) throughput 86% -> COMPUTE bound.
- 2.78 billion double-precision FMAs, 74 regs/thread, 46% occupancy.
- The numba op ran in float64 (a bare `2 *`, `** 0.5`, np.cosh/np.cos promote to f64),
  and RTX 6000 Ada is 1:64 FP64 -> the fused kernel is FP64-transcendental bound while
  the cupy baseline runs in float32.

Fix: keep the op in float32 (see mass_cuda_compute.py).
- kernel 8.93 ms -> 3.53 ms; region span 11 ms -> 6.1 ms.
- walltime speedup 3.7x -> 9.4x; nsys-span speedup 2.9x -> 5.3x.
- (numba's cosh/cos/sqrt still compute in f64, so it is still ~86% SM; float32
  transcendental intrinsics would push it toward memory-bound.)
