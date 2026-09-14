from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(requested)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                nested_key: nested_value.to(device)
                if hasattr(nested_value, "to")
                else nested_value
                for nested_key, nested_value in value.items()
            }
        elif hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
