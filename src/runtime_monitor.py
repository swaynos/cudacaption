from __future__ import annotations

import gc
import subprocess
import time


def _torch_cuda_stats() -> dict:
    try:
        import torch
    except Exception:
        return {"torch_cuda_available": False}

    if not torch.cuda.is_available():
        return {"torch_cuda_available": False}

    return {
        "torch_cuda_available": True,
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
    }


def _nvidia_smi_snapshot() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return "nvidia-smi unavailable"

    if proc.returncode != 0:
        return f"nvidia-smi error: {proc.stderr.strip() or proc.stdout.strip()}"
    return proc.stdout.strip().replace("\n", " | ")


def phase_marker(phase: str) -> None:
    print(f"PHASE: {phase}")
    log_vram(f"phase={phase}")


def log_vram(context: str) -> None:
    stats = _torch_cuda_stats()
    smi = _nvidia_smi_snapshot()
    print(f"VRAM [{context}] torch={stats} nvidia_smi='{smi}'")


def cleanup_cuda(context: str) -> None:
    before = _torch_cuda_stats()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize()
    except Exception:
        pass
    time.sleep(0.1)
    after = _torch_cuda_stats()
    print(f"VRAM cleanup [{context}] before={before} after={after}")
