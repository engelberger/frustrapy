"""Single shared concurrency budget for every FrustraPy backend.

This module is the one place that decides how many worker processes and how
many native-math threads FrustraPy may use at once. Every pool — the mutation
scan (``analysis/mutations.py``), the structure/frame batch
(``analysis/frustration.py``), and the evolution precompute
(``evolution/information_content.py``) — draws from the helpers here instead of
calling :func:`multiprocessing.cpu_count` independently. That guarantees the
nested pools can never multiply into a fork bomb:

* The total number of live worker processes never exceeds the physical core
  budget, because an outer pool of ``outer`` items hands each item an inner
  budget of ``cores // outer`` (see :func:`resolve_concurrency`), so
  ``outer * inner <= cores`` on every nesting path.
* Each worker process runs its native math libraries (OpenBLAS/MKL/OpenMP)
  single-threaded (see :func:`apply_thread_limits`), so ``cores`` processes can
  no longer spin ``cores`` threads each and reach ``cores**2`` live threads.
* Pools are created from a start method that is safe to use from a possibly
  multi-threaded parent (see :func:`get_pool_context`), avoiding the
  fork-after-threads deadlock and the resulting orphaned LAMMPS children.

The user-facing summary of the resulting limits lives in the docs
("Parallelism and resource limits"); the audit baseline that motivated this is
``docs/parallelism_inventory.md``.
"""

import multiprocessing
from typing import Optional, Tuple

# `apply_thread_limits` (and its variable list) live in the dependency-free leaf
# module `frustrapy._threadlimits` so they can run before numpy is imported,
# at the very top of `frustrapy/__init__.py`, without dragging in this module's
# imports. Re-exported here so callers have one import surface for concurrency.
from .._threadlimits import THREAD_ENV_VARS, apply_thread_limits  # noqa: F401


def cpu_budget() -> int:
    """The shared physical-core budget every FrustraPy pool draws from.

    A single definition so no call site invents its own (possibly larger)
    bound. Never returns less than 1.
    """
    try:
        return max(1, multiprocessing.cpu_count())
    except NotImplementedError:  # pragma: no cover - platform-dependent
        return 1


def resolve_concurrency(
    n_procs: Optional[int],
    n_items: int,
    cores: Optional[int] = None,
) -> Tuple[int, int]:
    """Split the shared core budget into an ``(outer, inner)`` pair.

    ``outer`` is how many items (structures, trajectory frames, evolution jobs)
    run concurrently; ``inner`` is how many CPUs each item's own inner pool may
    use. The invariant ``outer * inner <= cores`` holds for any inputs, so two
    nested pools sized this way never oversubscribe past the core budget.

    Args:
        n_procs: requested outer width. ``None``/0 means "use the full budget".
        n_items: number of items to process; caps ``outer`` (never spawn more
            outer workers than there is work).
        cores: override the core budget (defaults to :func:`cpu_budget`).

    Returns:
        ``(outer, inner)`` with both at least 1.
    """
    if cores is None:
        cores = cpu_budget()
    cores = max(1, int(cores))
    n_items = max(1, int(n_items))
    requested = cores if not n_procs else int(n_procs)
    outer = max(1, min(requested, n_items, cores))
    inner = max(1, cores // outer)
    return outer, inner


def resolve_pool_size(
    requested: Optional[int],
    n_tasks: int,
    cores: Optional[int] = None,
) -> int:
    """Resolve the worker count for a single (non-nested) pool.

    ``min(requested_or_budget, cores, n_tasks)`` — never oversubscribes the core
    budget and never spawns more workers than tasks. This is the inner-pool /
    flat-pool sizing; for the two-level batch case use
    :func:`resolve_concurrency` to get ``(outer, inner)`` first and pass
    ``inner`` here as ``requested``.
    """
    if cores is None:
        cores = cpu_budget()
    requested = cores if requested is None else int(requested)
    return max(1, min(requested, cores, n_tasks))


def get_pool_context():
    """Return a multiprocessing context with a fork-safe start method.

    Prefers ``forkserver`` (a clean, single-threaded server forks each worker,
    so a multi-threaded parent cannot deadlock the child), then ``spawn``, then
    the platform default. FrustraPy's pool payloads are already picklable (Pool
    pickles task args regardless of start method), so this switch does not
    change results — only the safety of the fork.
    """
    for method in ("forkserver", "spawn"):
        try:
            if method in multiprocessing.get_all_start_methods():
                return multiprocessing.get_context(method)
        except (ValueError, RuntimeError):  # pragma: no cover - platform-dependent
            continue
    return multiprocessing.get_context()


def pool_worker_initializer() -> None:
    """Pool-worker initializer: re-pin native-math threads in the worker.

    Belt-and-suspenders for spawn/forkserver workers; the environment is already
    inherited from the parent (which called :func:`apply_thread_limits` at import
    time), but re-applying here keeps the guarantee local to the pool.
    """
    apply_thread_limits()
