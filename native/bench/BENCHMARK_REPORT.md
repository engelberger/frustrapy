# FrustraPy backend benchmark: correctness, speed, scaling, and bottlenecks

This report measures the FrustraPy native frustration core across CPU (serial and multicore)
and Apple Metal GPU, centered on the workload that matters in practice: the single residue
saturation mutational scan (deep mutational scanning, DMS), where every position is mutated to
all 20 amino acids and frustration is recomputed, giving N x 20 per variant calculations. This
is the FrustraMPNN scale, proteome scale use case.

Every speed number is paired with a parity check, because a faster answer that is a different
answer is not a speedup. The headline is that the GPU and multicore paths are numerically
equivalent to the CPU reference and turn multi hour or multi day saturation scans into minutes.

## 1. Methodology and honesty constraints

- Two machines, never mixed on one wall time axis. Host is Apple Silicon arm64 (10 performance
  plus 4 efficiency cores), native. The dev container is x86_64 emulated on that arm64 host, so
  container absolute times are emulation inflated and reported as indicative only; within machine
  ratios and parity remain valid. The real x86 speed sweep is a separate cluster job (paul nodes),
  see section 7.
- Apples to apples: same protein, same n_decoys, same algorithm. Backends differ only in
  implementation. The roughly 1e-6 agreement between CPU and Metal is the evidence that they run
  the same computation.
- Statistics: triplicate timing with the cold first repeat discarded, mean plus sample standard
  deviation. Size axis is a sample of AlphaFold2 monomers spanning 110 to 1863 residues (21
  proteins), so the trend has biological replicates per size bin and a fitted scaling law.
- Five parity metrics against a frozen reference: Spearman, Pearson R squared, max absolute
  difference, RMSE, and class agreement (the contact classes at -1 and 0.78, the single residue
  plot cutoff at 0.58). Spearman and R squared saturate at 1.0 between correct builds, so the
  discriminating metrics are max absolute difference and class agreement.

## 2. Correctness: all backends are equivalent to the CPU reference

Across all three frustration modes and all 21 proteins, Metal and multicore CPU match the serial
CPU reference: Spearman = R squared = class agreement = 1.0000, max absolute FrstIndex difference
about 1e-6 (float32 rounding on Metal, exact 0 on CPU multicore). The CPU core itself is parity
gated to the LAMMPS reference (Spearman 1.0, max difference about 1e-6) from the container run,
so the chain CPU reference to Metal is transitive to LAMMPS.

## 3. Three time bases (the narrative spine)

- Kernel only: the AWSEM compute ceiling. This is where Metal wins, up to about 148x versus
  serial CPU and about 16x versus 14 core CPU on the mutational kernel at 415 residues.
- End to end single protein: what a user feels for one structure. Data prep dominates (section 6),
  so the GPU advantage on a single small structure is muted.
- Batch and proteome scale (the DMS scan): prep amortizes and the kernel dominates again, so the
  multicore and GPU advantage returns and compounds over N x 20 variants. This is the headline.

## 4. Deep mutational scan vs protein size (Fig 4, Fig 5)

Per variant we use the mutational frustration mode (the heavy decoy ensemble), and the full DMS
scan cost is N x 20 x per variant. Projected full scan wall time, host arm64, mutational:

| protein (residues) | CPU x1 | CPU x14 | Metal |
|---|---|---|---|
| CDK2 (298)  | 3.4 h  | 25 min | 1.8 min |
| p53 (393)   | 4.0 h  | 31 min | 2.3 min |
| albumin (609) | 20.8 h | 2.9 h | 8.5 min |
| EGFR (1210) | days   | 12.6 h | 41 min |
| BRCA1 (1863)| days   | 15.7 h | 62 min |

Power law fits t proportional to N to the power b (least squares on log log, fit R squared > 0.98):
CPU x1 b = 1.71, CPU x14 b = 1.46, Metal b = 0.95. The GPU flattens the per variant scaling
exponent to near linear because it parallelizes the per residue work, while the CPU stays
superlinear (the 5 A density and contact work is order N times contacts). Since the DMS scan
multiplies by N x 20, the CPU scan grows about N to the 2.7 and the Metal scan about N to the 1.95,
so the gap widens with size. This is why Metal makes proteome scale DMS tractable.

Mode dependence (important and not hidden): for the light configurational mode the per variant
kernel is sub millisecond, so the roughly 5 to 8 ms GPU launch floor dominates and CPU wins at
every size. Metal helps when the per variant kernel is heavy (mutational), not when it is light.

## 5. CPU scaling efficiency (Fig 7 left)

Mutational kernel, p53 (393 residues), fine thread sweep. Efficiency (speedup divided by cores)
stays about 0.85 to 10 threads (8.5x), then plateaus near 9x and falls at 12 to 14 threads. The
Amdahl serial fraction fit is s = 0.035, a ceiling of about 28x, so the plateau is not Amdahl
limited. It is the host's heterogeneous 10 performance plus 4 efficiency cores: past the 10
performance cores the efficiency cores add little and the mismatch hurts. On a homogeneous HPC
node with equal cores, scaling should continue well past 9x toward the ceiling. Re run this sweep
on a paul node to confirm the homogeneous scaling.

## 6. Bottleneck attribution (Fig 7 right)

For one per variant calculation at 393 residues: data prep is about 1.33 s while the kernel is
about 0.21 s, so prep (the PdbCoords2Lammps subprocess, structure parse, and gamma read) is about
6x the actual compute. The decoy ensemble is the compute cost; the 5 A density part is under a
millisecond. The actionable conclusion matches the maintainer's intuition: for a DMS scan that
currently redoes prep for every mutant, prep is the bottleneck, not the kernel. Amortizing the
prep scaffolding across the N x 20 variants (only the residue identity changes between mutants,
the coordinates and gamma tables are identical) is the single highest value optimization, and it
helps every backend equally. The subprocess based prep is also where IO handling can be improved
(avoid re writing intermediate files per mutant).

## 7. Compiler flags: parity safe, speed flat on arm64, real x86 pending (Fig 6)

Flag sweep on the mutational kernel, comparing every variant against a frozen -O2 reference.

- Parity: -ffast-math, -march=native, and -march=x86-64-v3 do change FrstIndex, but only at max
  absolute difference about 1e-15 (double precision epsilon, from FMA contraction and reassociation),
  and class agreement stays 1.0000. The feared -ffast-math parity break does not occur for this well
  conditioned kernel; the drift never flips a frustration class. These roughly 1e-15 deltas are also
  the proof that the flags reached the compiler. The recommendation for a parity gated tool is to
  verify per flag (as done here) rather than assume, and ship -O3 as the safe default.
- Speed on arm64 (real): -O3 is already optimal. -funroll-loops, -march/-mcpu=native, and
  -ffast-math do not help and -ffast-math or -mcpu=native slightly hurt (0.84 to 0.97x), with higher
  variance. No free lunch from flags on Apple clang for this kernel.
- Speed on x86: the container numbers (-ffast-math about 1.2 to 1.3x, x86-64-v3 about 1.1x) are
  emulation indicative only and are not trustworthy. The real x86 speed sweep runs on the cluster:
  native/bench/cluster/run_flag_sweep_slurm.sh on a paul node (it auto detects AVX-512 to decide
  whether to add x86-64-v4). -march=native produces a non portable binary that can fault on an older
  node, which is the reason production HPC builds either target a portable baseline (x86-64-v3) or
  use runtime dispatch.

## 8. Prep amortization (implemented): prep once per structure

The bottleneck in section 6 (prep about 6x the kernel, redone per mutant) is now addressed.
prepare_structure parses the cleaned PDB, reads the gammas and coefficients, and computes the
geometry cache (per residue density rho and the energy contact list) once per structure;
compute_variant_frustration then scores each variant with a res_type swap that reuses the cached
geometry. The native core takes the precomputed geometry as optional inputs and is bit for bit
identical to the one shot path when they are absent (test_native_geometry_reuse_bit_identical),
so the optimization changes no numbers.

Measured by native/bench/prep_amortization.py (1zni chain B, 30 residues, seq_dist 12, serial,
in container x86_64 emulated so absolute times are indicative, the ratio is the result): a full WT
calculation (prep plus kernel) is about 598 ms, prepare_structure is about 0.6 ms, and an amortized
variant kernel is about 19 ms. For a 10 position scan (200 variants) the current K x (prep plus
kernel) is about 120 s versus prep_once plus K x kernel about 4.3 s, a 27.5x prep amortization
factor (asymptotic ceiling about 32x as the scan grows). The amortized per variant FrstIndex is
bit for bit a fresh native recompute, so the speedup is real work removed, not a different answer.
Run: python native/bench/prep_amortization.py --pdb tests/data/1zni.pdb --chain B --res 25.

## 9. Limitations and next steps

- Real x86 speed (flags and multicore scaling on homogeneous cores) is the paul node job.
- CUDA timing is the maintainer's Colab or cluster step; the harness already includes a CUDA path
  that activates when has_cuda is true.
- Metal is float32; parity holds at the class level here, but a float64 sensitive downstream use
  should prefer the CPU path.

## Reproduce

```
# host (arm64, Metal): build native with Metal + OpenMP, then
python native/bench/paper_bench.py --pdbs benchmark_results/afdb/AF-*.pdb --mode mutational --reps 3 --out bench_sample_mut.json
python native/bench/scaling_bottleneck.py --pdb benchmark_results/afdb/AF-P04637.pdb --out bench_scaling_bottleneck.json
python native/bench/flag_sweep.py --pdbs ... --out bench_flags_arm64.json
python native/bench/make_report.py        # regenerates all figures + scaling_fits.json
# cluster (real x86): sbatch native/bench/cluster/run_flag_sweep_slurm.sh
```
