"""Make NVIDIA cuBLAS from pip wheels loadable for CTranslate2 on Windows.

CTranslate2's Windows wheels (CUDA 12.8 build, Blackwell-capable) load ``cublas64_12.dll``
with plain ``LoadLibrary`` on first GPU use. The ``nvidia-cublas-cu12`` wheel installs the
DLLs under ``site-packages/nvidia/cublas/bin``, which is not on the default search path, so
we put that folder on PATH and preload the DLLs by absolute path.
"""

from __future__ import annotations

import ctypes
import os
import sys
from importlib.util import find_spec
from pathlib import Path

_prepared = False
_loaded: list[str] = []


def _nvidia_roots() -> list[Path]:
    spec = find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    return [Path(p) for p in spec.submodule_search_locations]


def prepare_cuda_libraries() -> list[str]:
    """Idempotent. Returns the DLLs that were preloaded (empty on non-Windows or no wheels)."""
    global _prepared
    if _prepared or sys.platform != "win32":
        return list(_loaded)
    _prepared = True
    for root in _nvidia_roots():
        for sub in ("cublas", "cuda_nvrtc", "cuda_runtime", "cudnn"):
            bin_dir = root / sub / "bin"
            if bin_dir.is_dir():
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                os.add_dll_directory(str(bin_dir))
        for dll in ("cublasLt64_12.dll", "cublas64_12.dll"):
            path = root / "cublas" / "bin" / dll
            if path.is_file() and str(path) not in _loaded:
                ctypes.WinDLL(str(path))
                _loaded.append(str(path))
    return list(_loaded)
