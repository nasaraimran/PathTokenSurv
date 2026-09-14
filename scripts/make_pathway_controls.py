from __future__ import annotations

# Allow direct execution of scripts from a source checkout without requiring
# an editable installation first (e.g., ``python scripts/run_synthetic.py``).
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def shuffled_assignments(payload: dict, rng: np.random.Generator) -> dict:
    result = deepcopy(payload)
    pathways = result["pathways"]
    for modality, mapping in result["modality_features"].items():
        sizes = [len(mapping.get(pathway, [])) for pathway in pathways]
        features = [feature for pathway in pathways for feature in mapping.get(pathway, [])]
        rng.shuffle(features)
        cursor = 0
        for pathway, size in zip(pathways, sizes):
            mapping[pathway] = features[cursor : cursor + size]
            cursor += size
    return result


def degree_preserving_randomization(adjacency: np.ndarray, rng: np.random.Generator, swaps: int) -> np.ndarray:
    graph = np.asarray(adjacency, dtype=int).copy()
    graph = np.triu(graph, 1)
    graph = graph + graph.T
    edges = [tuple(edge) for edge in np.argwhere(np.triu(graph, 1) == 1)]
    if len(edges) < 2:
        return graph
    edge_set = {tuple(sorted(edge)) for edge in edges}
    attempts = 0
    completed = 0
    while completed < swaps and attempts < swaps * 50:
        attempts += 1
        first, second = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[first]
        c, d = edges[second]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            new1, new2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        else:
            new1, new2 = tuple(sorted((a, c))), tuple(sorted((b, d)))
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        if new1 in edge_set or new2 in edge_set:
            continue
        old1, old2 = tuple(sorted((a, b))), tuple(sorted((c, d)))
        edge_set.remove(old1)
        edge_set.remove(old2)
        edge_set.add(new1)
        edge_set.add(new2)
        edges[first] = new1
        edges[second] = new2
        completed += 1
    randomized = np.zeros_like(graph)
    for a, b in edge_set:
        randomized[a, b] = randomized[b, a] = 1
    return randomized


def main() -> None:
    parser = argparse.ArgumentParser(description="Create shuffled and degree-matched pathway controls.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--swaps", type=int, default=1000)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    shuffled = shuffled_assignments(payload, rng)
    (output / "pathways_shuffled_assignments.json").write_text(
        json.dumps(shuffled, indent=2), encoding="utf-8"
    )
    random_graph = deepcopy(payload)
    random_graph["adjacency"] = degree_preserving_randomization(
        np.asarray(payload["adjacency"]), rng, args.swaps
    ).tolist()
    (output / "pathways_degree_matched_random_graph.json").write_text(
        json.dumps(random_graph, indent=2), encoding="utf-8"
    )
    print(f"Control pathway files written to {output}")


if __name__ == "__main__":
    main()
