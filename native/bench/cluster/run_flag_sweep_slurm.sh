#!/usr/bin/env bash
#SBATCH --job-name=frustrapy-flagsweep
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=frustrapy-flagsweep-%j.out
# Real x86_64 compiler-flag SPEED sweep for the FrustraPy native CPU core.
#
# Run this on a real x86 cluster node (the paul nodes). The dev container is x86 EMULATED
# on an arm64 host, so its timings are not representative; only a native x86 CPU gives a
# trustworthy speed sweep. Parity (does a flag change results) is machine-independent and is
# already checked in-container; this job is for the SPEED numbers on real Intel/AMD silicon.
#
# Usage on the cluster:
#   1. Clone/copy the frustrapy repo (this script lives in native/bench/cluster/).
#   2. sbatch native/bench/cluster/run_flag_sweep_slurm.sh        # or run as a plain bash script
#   3. Send the produced bench_flags_x86_cluster.json back; it plots with native/bench/make_report.py.
#
# Pick the config set for the node's CPU: keep -march=x86-64-v3 (AVX2, portable across most
# HPC nodes) and add -march=x86-64-v4 (AVX-512) only on Skylake-SP/Ice Lake/Zen4. -march=native
# targets THIS node exactly (fastest, non-portable: the binary may SIGILL on an older node).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root
cd "$HERE"

module load gcc 2>/dev/null || true        # adjust to the cluster's module system
python3 -m venv /tmp/fp_cluster_venv
. /tmp/fp_cluster_venv/bin/activate
pip install -q --upgrade pip nanobind scikit-build-core numpy pandas biopython scipy

echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "Flags supported: $(grep -o -m1 'avx512[a-z]*\|avx2\|fma' /proc/cpuinfo | sort -u | tr '\n' ' ')"

# Detect AVX-512 to decide whether x86-64-v4 is safe.
if grep -q avx512f /proc/cpuinfo; then
  V4='["O3_x86v4","-O3 -march=x86-64-v4",true],'
else
  V4=''
fi
export FLAG_SWEEP_CONFIGS="[\
[\"O2\",\"-O2\",true],\
[\"O3\",\"-O3\",true],\
[\"O3_x86v2\",\"-O3 -march=x86-64-v2\",true],\
[\"O3_x86v3\",\"-O3 -march=x86-64-v3\",true],\
${V4}\
[\"O3_march_native\",\"-O3 -march=native\",false],\
[\"O3_native_ffast\",\"-O3 -march=native -ffast-math\",false]]"

# A small, fast structure panel (ships with the repo); add larger AF2 models for heavier signal.
PDBS="tests/data/1crn.pdb"
[ -d benchmark_results/afdb ] && PDBS="$PDBS $(ls benchmark_results/afdb/AF-P24941.pdb benchmark_results/afdb/AF-P04637.pdb 2>/dev/null)"

python3 native/bench/flag_sweep.py \
  --pdbs $PDBS --mode mutational --reps 5 \
  --out benchmark_results/bench_flags_x86_cluster.json

echo "DONE. Return benchmark_results/bench_flags_x86_cluster.json for plotting."
