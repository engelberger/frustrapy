"""Pluggable frustration-engine backends.

The energy model is the swappable part of a frustration run (see :mod:`.base`).
``lammps`` is the reference backend and the default; future backends (a native CPU
core, GPU) register here behind the same :class:`FrustrationBackend` interface.

Select a backend by name through the public API, e.g.
``calculate_frustration(..., backend="lammps")``. The default is ``lammps`` and the
on-disk output contract is identical regardless of backend.
"""

from __future__ import annotations

from typing import Dict, Type, Union

from .atomic import AtomicBackend
from .base import FrustrationBackend
from .lammps import LammpsBackend
from .native import NativeBackend

#: Name of the backend used when none is requested.
DEFAULT_BACKEND = "lammps"

_REGISTRY: Dict[str, Type[FrustrationBackend]] = {
    LammpsBackend.name: LammpsBackend,
    NativeBackend.name: NativeBackend,
    AtomicBackend.name: AtomicBackend,
}


def available_backends() -> list:
    """Return the sorted list of registered backend names."""
    return sorted(_REGISTRY)


def register_backend(cls: Type[FrustrationBackend]) -> Type[FrustrationBackend]:
    """Register a backend class under its ``name``. Returns the class (usable as a
    decorator)."""
    if not isinstance(cls, type) or not issubclass(cls, FrustrationBackend):
        raise TypeError("backend must be a FrustrationBackend subclass")
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(
    backend: Union[str, FrustrationBackend, None] = None,
) -> FrustrationBackend:
    """Resolve a backend selector to a :class:`FrustrationBackend` instance.

    Args:
        backend: a registered backend name (``str``), an already-constructed
            :class:`FrustrationBackend` instance (returned as-is), or ``None`` for
            the default (:data:`DEFAULT_BACKEND`).
    """
    if backend is None:
        backend = DEFAULT_BACKEND
    if isinstance(backend, FrustrationBackend):
        return backend
    if isinstance(backend, str):
        try:
            return _REGISTRY[backend]()
        except KeyError:
            raise ValueError(
                f"Unknown frustration backend {backend!r}. "
                f"Available: {available_backends()}"
            )
    raise TypeError(
        "backend must be a backend name (str), a FrustrationBackend instance, or None"
    )


__all__ = [
    "FrustrationBackend",
    "LammpsBackend",
    "NativeBackend",
    "AtomicBackend",
    "DEFAULT_BACKEND",
    "available_backends",
    "register_backend",
    "get_backend",
]
