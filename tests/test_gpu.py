"""Tests for the CUDA preload helper.

These must pass on a CPU-only machine (CI), so they assert behaviour that holds
with or without the ``gpu`` extra installed.
"""

from __future__ import annotations

from nfl.gpu import cuda_library_paths, preload_cuda_libraries


def test_cuda_library_paths_returns_absolute_paths() -> None:
    paths = cuda_library_paths()
    assert all(path.startswith("/") for path in paths)
    assert all(".so" in path for path in paths)


def test_preload_is_safe_without_cuda_and_counts_what_it_loads() -> None:
    loaded = preload_cuda_libraries()
    assert 0 <= loaded <= len(cuda_library_paths())


def test_preload_is_idempotent() -> None:
    assert preload_cuda_libraries() == preload_cuda_libraries()
