from __future__ import annotations

import numpy as np
import torch

from pathtokensurv.data.time import TimeDiscretizer
from pathtokensurv.metrics import harrell_c_index
from pathtokensurv.models.survival import CancerStratifiedDiscreteSurvivalHead


def test_discrete_targets_and_survival_monotonicity() -> None:
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    events = np.array([1, 0, 1, 0, 1], dtype=int)
    discretizer = TimeDiscretizer().fit(times, events, num_bins=4)
    labels, observed = discretizer.make_targets(
        torch.tensor(times, dtype=torch.float32),
        torch.tensor(events, dtype=torch.float32),
    )
    assert labels.shape == observed.shape == (5, discretizer.num_bins)
    assert torch.all(labels <= observed)

    logits = torch.randn(5, discretizer.num_bins)
    survival = CancerStratifiedDiscreteSurvivalHead.survival_from_logits(logits)
    assert torch.all((survival >= 0) & (survival <= 1))
    assert torch.all(survival[:, 1:] <= survival[:, :-1] + 1e-7)


def test_harrell_c_index_perfect_ordering() -> None:
    times = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.ones(4, dtype=int)
    risk = np.array([4.0, 3.0, 2.0, 1.0])
    assert harrell_c_index(times, events, risk) == 1.0
