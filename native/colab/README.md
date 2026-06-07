# Native CUDA backend: build and validation on GPU

The CUDA path (`native/src/cuda/kernels.cu`) is written to mirror the parity-gated
CPU core (`native/src/core.cpp`) term-for-term. It has not been compiled or run in the
development container (no nvcc/GPU). Build and validate it on a GPU machine (Google
Colab or the cluster); the steps below produce a parity report and timings. No GPU
timing is committed to the repo -- record the numbers this harness prints.

## What the GPU path does

- Density, native energy, and the decoy reductions run as CUDA kernels (SoA layout,
  one block per probed unit for the decoy mean/sd shared-memory reduction).
- The decoy random-index stream is generated host-side with the exact glibc `rand()`
  sequence (the same `GlibcRand` as the CPU core), then consumed by the kernels, so
  the GPU result matches the CPU core (the parity reference) rather than diverging on
  a different RNG. Mutational mode is the largest win (the per-(contact, decoy) energy
  evaluation, O(n_contacts x n_decoys x n_res), is the dominant cost).

## Build (Colab or cluster)

```bash
# A CUDA toolkit (nvcc) and a GPU must be present.
nvcc --version
git clone <this repo> && cd <repo>            # or: cd to the worktree
pip install numpy pandas biopython plotly scipy scikit-learn python-igraph leidenalg tqdm
pip install -e .                              # frustrapy (pure Python)
pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_CUDA=ON   # native core + CUDA

python -c "import frustrapy_native as fn; print('cuda:', fn.has_cuda())"   # -> cuda: True
```

If `nvcc` is absent the build falls back to CPU-only and `has_cuda()` returns `False`
(graceful, never a hard failure).

## Validate parity + measure speed

```bash
python native/colab/bench_cuda.py tests/data/1crn.pdb --mode mutational --seq-dist 12
python native/colab/bench_cuda.py <larger>.pdb       --mode mutational --seq-dist 12
python native/colab/bench_cuda.py tests/data/1crn.pdb --mode singleresidue
```

The harness runs the native core on identical inputs with `use_cuda=False` then
`use_cuda=True`, asserts FrstIndex Spearman >= 0.99 and native-energy agreement, and
prints CPU vs CUDA wall-clock and the speedup. The CPU core is already gated bit-for-bit
against the LAMMPS reference (`tests/test_native_parity.py`), so CPU-vs-CUDA parity
closes the loop to the reference.

To route the full FrustraPy pipeline through the GPU, set the env flag (the core must
be a CUDA build):

```bash
FRUSTRAPY_NATIVE_USE_CUDA=1 python -c "import frustrapy; frustrapy.calculate_frustration('tests/data/1crn.pdb', mode='mutational', backend='native')"
```

## compute-sanitizer (where available)

```bash
compute-sanitizer --tool memcheck python native/colab/bench_cuda.py tests/data/1crn.pdb --mode mutational
```

Expect a clean report (no invalid accesses / races). Run `racecheck` and `initcheck`
similarly. The CPU core is ASan/UBSan-clean
(`pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_SANITIZE=ON`).
