"""GPU setup helpers for TensorFlow.

TensorFlow's CUDA libraries ship as separate ``nvidia-*`` wheels that install
into ``site-packages/nvidia/*/lib``. TensorFlow ``dlopen``s them by soname and
does **not** add those directories to the loader path, so on a plain
``uv sync --extra gpu`` it reports "Cannot dlopen some GPU libraries" and falls
back to CPU.

:func:`preload_cuda_libraries` fixes that from inside the process: loading each
library with ``RTLD_GLOBAL`` puts its soname in the process namespace, so
TensorFlow's later ``dlopen`` resolves against the already-loaded object. Call
it *before* importing ``tensorflow``.

The equivalent from outside the process is to point ``LD_LIBRARY_PATH`` at the
same directories; see ``CLAUDE.md``.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys

__all__ = ["cuda_library_paths", "preload_cuda_libraries"]


def cuda_library_paths() -> list[str]:
    """Absolute paths of the CUDA shared objects shipped by the nvidia wheels."""
    site_packages = os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )
    paths: list[str] = []
    for lib_dir in sorted(glob.glob(os.path.join(site_packages, "nvidia", "*", "lib"))):
        paths.extend(sorted(glob.glob(os.path.join(lib_dir, "*.so*"))))
    return paths


def preload_cuda_libraries() -> int:
    """Load the pip-installed CUDA libraries into the global symbol namespace.

    Must be called before ``import tensorflow``. Safe to call when no CUDA
    wheels are present (returns 0) and safe to call more than once.

    Returns:
        The number of shared objects successfully loaded.
    """
    loaded = 0
    for path in cuda_library_paths():
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Some objects are stubs or have unmet optional deps; TensorFlow
            # only needs the ones it asks for, so a failure here is not fatal.
            continue
        loaded += 1
    return loaded
