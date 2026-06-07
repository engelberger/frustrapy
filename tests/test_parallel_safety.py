"""Live resource-bound tests for the nested parallel paths.

The structural sizing math is covered in ``test_parallelism.py``; these tests
instead RUN the dangerous nest and watch the live process tree, proving that the
single shared core budget is enforced end to end rather than only by convention.

The fork-bomb path is ``dir_frustration(n_procs=K)`` over single-residue
structures: each outer structure worker opens its OWN mutation scan pool, so
without a shared budget the live LAMMPS process count would reach
``K x min(cores, 20)``. With the budget the outer pool takes ``K`` and each inner
pool gets ``cores // K``, so the total CPU-bound (``lmp_serial``) processes alive
at once can never exceed ``cores`` (see ``frustrapy/utils/concurrency.py`` and
``docs/parallelism_inventory.md``).

These tests require ``psutil`` (the project's ``perf`` extra) and, like the rest
of the suite, the project venv on PATH (a calculation spawns a bare ``python3``).
"""

import os
import shutil
import threading
import time

import pytest

from frustrapy.utils.concurrency import (
    cpu_budget,
    get_pool_context,
    pool_worker_initializer,
)

psutil = pytest.importorskip("psutil")

# A handful of extra processes are legitimate transients: the per-structure
# `PdbCoords2Lammps.sh` briefly spawns `python3` helpers, and a worker may be
# exiting as another's `lmp_serial` starts. The budget bounds CPU-bound work, not
# these short-lived helpers, so allow a small slack over the core budget.
SLACK = 3


def _count_lammps_descendants(proc):
    """Number of live ``lmp_serial`` (AWSEM/LAMMPS) processes under ``proc``.

    ``lmp_serial`` is the single-threaded CPU unit of every frustration calc, so
    its live count is exactly the number of cores being consumed at that instant.
    """
    n = 0
    for child in proc.children(recursive=True):
        try:
            if "lmp_serial" in child.name():
                n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return n


class _PeakSampler(threading.Thread):
    """Daemon thread that polls the live ``lmp_serial`` descendant count."""

    def __init__(self, interval=0.03):
        super().__init__(daemon=True)
        self._interval = interval
        self._stop_event = threading.Event()
        self.peak = 0
        self.samples = 0
        self._me = psutil.Process(os.getpid())

    def run(self):
        while not self._stop_event.is_set():
            try:
                n = _count_lammps_descendants(self._me)
            except psutil.Error:
                n = 0
            if n > self.peak:
                self.peak = n
            self.samples += 1
            time.sleep(self._interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)


def _make_batch(crn_pdb, tmp_path, names):
    batch = tmp_path / "batch"
    batch.mkdir()
    for n in names:
        shutil.copyfile(crn_pdb, batch / f"prot{n}.pdb")
    return str(batch)


@pytest.mark.slow
def test_nested_singleresidue_batch_stays_within_core_budget(crn_pdb, tmp_path):
    """The K-structure x per-residue-scan nest never oversubscribes the cores.

    Runs ``dir_frustration(n_procs=2, mode='singleresidue', graphics=True)`` over
    two structures, each scanning ONE residue (20 LAMMPS evals). This is the exact
    nest that, unbounded, would run ``2 x min(cores, 20)`` concurrent LAMMPS
    processes; the shared budget caps the outer pool at 2 and each inner pool at
    ``cores // 2``, so the live ``lmp_serial`` count must stay at or below the
    core budget. We assert the sampled peak is within ``cores + SLACK``.
    """
    from frustrapy.analysis.frustration import dir_frustration

    cores = cpu_budget()
    pdbs_dir = _make_batch(crn_pdb, tmp_path, ["a", "b"])
    results_dir = str(tmp_path / "out")

    me = psutil.Process(os.getpid())
    assert _count_lammps_descendants(me) == 0, "stray lmp_serial before the run"

    sampler = _PeakSampler()
    sampler.start()
    try:
        plots, _density = dir_frustration(
            pdbs_dir=pdbs_dir,
            mode="singleresidue",
            residues={"A": [1]},
            graphics=True,
            visualization=False,
            results_dir=results_dir,
            n_procs=2,
        )
    finally:
        sampler.stop()

    # The nest actually executed end to end.
    assert sorted(plots) == ["prota", "protb"]
    for n in ("a", "b"):
        table = os.path.join(
            results_dir, f"prot{n}.done", "FrustrationData", f"prot{n}.pdb_singleresidue"
        )
        assert os.path.exists(table), f"missing singleresidue output for prot{n}"

    # Core guarantee: live CPU-bound (lmp_serial) processes never exceeded the
    # shared core budget. A regression that stacks two cpu_count()-sized pools
    # would push this toward 2 x min(cores, 20).
    assert sampler.samples > 0, "sampler never ran"
    assert sampler.peak <= cores + SLACK, (
        f"nested run peaked at {sampler.peak} concurrent lmp_serial "
        f"(> core budget {cores} + slack {SLACK})"
    )

    # Clean shutdown: the pools closed/joined and no LAMMPS child leaked.
    leftover = _count_lammps_descendants(me)
    assert leftover == 0, f"{leftover} orphan lmp_serial processes after the run"


def _raise_worker(_x):
    """Top-level (picklable) task that always fails, to induce a worker error."""
    raise RuntimeError("induced worker failure")


def _noop_worker(x):
    """Top-level (picklable) no-op, used to warm up the pool context."""
    return x


def test_pool_closes_cleanly_on_worker_error():
    """A worker exception still drains and joins the pool, leaving no orphans.

    Mirrors the real cleanup contract in ``mutate_res_scan_parallel``: the pool is
    created from the shared fork-safe context with the thread-pinning initializer,
    and ``close()``/``join()`` run in a ``finally`` so a raising worker can never
    leak child processes.
    """
    me = psutil.Process(os.getpid())
    ctx = get_pool_context()

    # Warm up first: a forkserver/spawn context lazily starts a persistent server
    # and resource-tracker process on first pool use. Those are intentional,
    # bounded singletons (not leaked workers), so start them and fold them into
    # the baseline before measuring what the failing pool leaves behind.
    warm = ctx.Pool(processes=2, initializer=pool_worker_initializer)
    try:
        list(warm.imap_unordered(_noop_worker, range(2)))
    finally:
        warm.close()
        warm.join()
    before = {c.pid for c in me.children(recursive=True)}

    pool = ctx.Pool(processes=2, initializer=pool_worker_initializer)
    with pytest.raises(RuntimeError):
        try:
            for _ in pool.imap_unordered(_raise_worker, range(8)):
                pass
        finally:
            pool.close()
            pool.join()

    # Give the OS a moment to reap the joined workers, then confirm none linger.
    for _ in range(50):
        now = {c.pid for c in me.children(recursive=True)}
        if not (now - before):
            break
        time.sleep(0.05)
    leftover = {c.pid for c in me.children(recursive=True)} - before
    assert not leftover, f"{len(leftover)} worker process(es) leaked after error"
