from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def joint_strata(cancers: Sequence, events: Sequence) -> np.ndarray:
    cancers = np.asarray(cancers).astype(str)
    events = np.asarray(events).astype(int).astype(str)
    return np.char.add(np.char.add(cancers, "__"), events)


def _can_stratify(labels: np.ndarray, n_splits: int = 2) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    return len(counts) > 1 and counts.min() >= n_splits


def train_val_test_split(
    cancers: Sequence,
    events: Sequence,
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(cancers)
    indices = np.arange(n)
    labels = joint_strata(cancers, events)
    stratify = labels if _can_stratify(labels, 2) else None
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=seed,
        stratify=stratify,
    )
    temp_labels = labels[temp_idx]
    relative_val = val_fraction / (1.0 - train_fraction)
    stratify_temp = temp_labels if _can_stratify(temp_labels, 2) else None
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val,
        random_state=seed + 1,
        stratify=stratify_temp,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def outer_folds(
    cancers: Sequence,
    events: Sequence,
    n_splits: int,
    seed: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    labels = joint_strata(cancers, events)
    indices = np.arange(len(labels))
    if _can_stratify(labels, n_splits):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(indices, labels)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(indices)
