import os
from dataclasses import dataclass
from typing import Optional, Union

import torch


def _parse_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _parse_dtype(dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if dtype is None:
        return torch.double
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower().replace("torch.", "")
    aliases = {
        "double": "float64",
        "single": "float32",
    }
    name = aliases.get(name, name)
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported dtype: {dtype}")
    parsed = getattr(torch, name)
    if not isinstance(parsed, torch.dtype):
        raise ValueError(f"Unsupported dtype: {dtype}")
    return parsed


def polyround_backend_from_cobra_solver(cobra_solver: str) -> str:
    """Map the configured Cobra solver name to the PolyRound backend name."""
    normalized = str(cobra_solver).strip().lower()
    aliases = {
        "glpk_exact": "glpk",
    }
    return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class SBMFIConfig:
    device: torch.device
    dtype: torch.dtype
    batch_size: int
    cobra_solver: str
    cvxpy_solver: str

    @classmethod
    def from_env(cls) -> "SBMFIConfig":
        return cls(
            device=_parse_device(os.environ.get("SBMFI_TORCH_DEVICE", "cpu")),
            dtype=_parse_dtype(os.environ.get("SBMFI_TORCH_DTYPE", "double")),
            batch_size=int(os.environ.get("SBMFI_BATCH_SIZE", "1")),
            cobra_solver=os.environ.get("SBMFI_COBRA_SOLVER", "glpk"),
            cvxpy_solver=os.environ.get("SBMFI_CVXPY_SOLVER", "CLARABEL"),
        )
