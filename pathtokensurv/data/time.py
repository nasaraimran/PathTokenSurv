from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch


@dataclass
class TimeDiscretizer:
    edges: np.ndarray | None = None

    def fit(self, times: Sequence[float], events: Sequence[float], num_bins: int) -> "TimeDiscretizer":
        times_arr = np.asarray(times, dtype=float)
        events_arr = np.asarray(events, dtype=int)
        valid = np.isfinite(times_arr) & (times_arr > 0)
        times_arr = times_arr[valid]
        events_arr = events_arr[valid]
        event_times = times_arr[events_arr == 1]
        if event_times.size < 2:
            event_times = times_arr
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.quantile(event_times, quantiles)
        edges[0] = 0.0
        edges[-1] = max(float(times_arr.max()), float(edges[-1])) + 1e-6
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.linspace(0.0, max(float(times_arr.max()), 1.0) + 1e-6, max(3, num_bins + 1))
        self.edges = edges.astype(np.float32)
        return self

    @property
    def num_bins(self) -> int:
        self._check()
        assert self.edges is not None
        return len(self.edges) - 1

    @property
    def right_edges(self) -> np.ndarray:
        self._check()
        assert self.edges is not None
        return self.edges[1:]

    def make_targets(self, times: torch.Tensor, events: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check()
        assert self.edges is not None
        edges = torch.as_tensor(self.edges, dtype=times.dtype, device=times.device)
        right = edges[1:]
        left = edges[:-1]
        t = times.unsqueeze(1)
        e = events.unsqueeze(1) > 0.5
        event_in_interval = e & (t > left.unsqueeze(0)) & (t <= right.unsqueeze(0))
        survived_full_interval = t >= right.unsqueeze(0)
        observed = survived_full_interval | event_in_interval
        labels = event_in_interval.to(times.dtype)
        return labels, observed.to(times.dtype)

    def save(self, path: str | Path) -> None:
        self._check()
        assert self.edges is not None
        Path(path).write_text(json.dumps({"edges": self.edges.tolist()}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TimeDiscretizer":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(edges=np.asarray(raw["edges"], dtype=np.float32))

    def _check(self) -> None:
        if self.edges is None:
            raise RuntimeError("TimeDiscretizer.fit must be called first.")
