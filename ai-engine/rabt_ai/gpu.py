"""GPU facts from ``nvidia-smi`` (no CUDA libraries needed)."""

from __future__ import annotations

import shutil
import subprocess
import sys

from rabt_shared import GpuStatus

_QUERY = "name,driver_version,memory.total,memory.used,compute_cap"


def query_gpu() -> GpuStatus:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return GpuStatus(available=False)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        proc = subprocess.run(
            [exe, f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return GpuStatus(available=False)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        return GpuStatus(available=False)
    try:
        name, driver, total, used, cc = (part.strip() for part in lines[0].split(","))
        return GpuStatus(
            available=True,
            name=name,
            driver_version=driver,
            total_vram_mb=int(float(total)),
            used_vram_mb=int(float(used)),
            compute_capability=cc,
        )
    except ValueError:
        return GpuStatus(available=True, name=lines[0][:100])
