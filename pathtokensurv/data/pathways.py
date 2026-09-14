from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence

import numpy as np

from pathtokensurv.constants import MOLECULAR_MODALITIES


_MIRNA_ARM_RE = re.compile(r"^(?P<base>.+)-(?:3p|5p)$", flags=re.IGNORECASE)


@dataclass
class PathwaySpec:
    pathway_names: List[str]
    feature_names: Dict[str, List[str]]
    feature_indices: Dict[str, List[List[int]]]
    residual_indices: Dict[str, List[int]]
    adjacency: np.ndarray
    # Number of fold-selected features that were eligible for each pathway before
    # modality-specific top-K truncation. This is QC metadata and is not consumed
    # by the neural network.
    eligible_feature_counts: Dict[str, List[int]] = field(default_factory=dict)
    # Outcome-independent normalized multimodal coverage score used to rank the
    # retained pathway candidates before the optional source-balancing step.
    pathway_selection_scores: List[float] = field(default_factory=list)

    @property
    def num_pathways(self) -> int:
        return len(self.pathway_names)

    def to_dict(self) -> dict:
        return {
            "pathway_names": self.pathway_names,
            "feature_names": self.feature_names,
            "feature_indices": self.feature_indices,
            "residual_indices": self.residual_indices,
            "adjacency": self.adjacency.tolist(),
            "eligible_feature_counts": self.eligible_feature_counts,
            "pathway_selection_scores": self.pathway_selection_scores,
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> "PathwaySpec":
        feature_indices = {
            k: [list(map(int, indices)) for indices in value]
            for k, value in raw["feature_indices"].items()
        }
        eligible = raw.get("eligible_feature_counts")
        if not eligible:
            eligible = {
                k: [len(indices) for indices in value]
                for k, value in feature_indices.items()
            }
        scores = list(map(float, raw.get("pathway_selection_scores", [])))
        if not scores:
            scores = [0.0] * len(raw["pathway_names"])
        return cls(
            pathway_names=list(raw["pathway_names"]),
            feature_names={k: list(v) for k, v in raw["feature_names"].items()},
            feature_indices=feature_indices,
            residual_indices={k: list(map(int, v)) for k, v in raw["residual_indices"].items()},
            adjacency=np.asarray(raw["adjacency"], dtype=np.float32),
            eligible_feature_counts={
                k: list(map(int, v)) for k, v in eligible.items()
            },
            pathway_selection_scores=scores,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PathwaySpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def canonical_feature_name(modality: str, value: object) -> str:
    """Canonicalize identifiers for robust annotation matching.

    TCGA PanCancer mRNA matrices can use either gene symbols or Entrez IDs,
    depending on the export. CNV matrices generally use gene symbols and miRNA
    matrices use mature miRNA names. The original spelling is preserved elsewhere;
    canonicalization is used only for matching.
    """
    text = str(value).strip()
    if modality == "mrna":
        text = text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text
        return text if text.isdigit() else text.upper()
    if modality == "cnv":
        return text.upper()
    if modality == "mirna":
        return text.lower()
    return text


def _mirna_precursor_key(value: object) -> str:
    """Return a mature-arm-independent key without guessing between 3p and 5p."""
    name = canonical_feature_name("mirna", value)
    match = _MIRNA_ARM_RE.match(name)
    return match.group("base") if match else name


def conservative_mirna_alias_map(annotation_names: Sequence[str]) -> Dict[str, str]:
    """Map an arm-unspecified miRNA name to one unique mature-arm annotation.

    A precursor-style alias (for example ``hsa-miR-375``) is generated only when
    the complete annotation contains exactly one arm-specific form for that base
    and does not already contain the arm-unspecified identifier. If both 3p and 5p
    forms exist, no alias is created. This avoids inventing an arm assignment.
    """
    canonical = {canonical_feature_name("mirna", x) for x in annotation_names if str(x).strip()}
    by_base: Dict[str, set[str]] = {}
    for name in canonical:
        match = _MIRNA_ARM_RE.match(name)
        if match:
            by_base.setdefault(match.group("base"), set()).add(name)
    return {
        base: next(iter(arms))
        for base, arms in by_base.items()
        if len(arms) == 1 and base not in canonical
    }


def mapping_conservative_mirna_alias_map(mapping: Mapping) -> Dict[str, str]:
    """Use provider-wide aliases stored by v1.5.4; derive only for legacy maps."""
    stored = mapping.get("metadata", {}).get("mirna_conservative_aliases")
    if isinstance(stored, Mapping):
        return {
            canonical_feature_name("mirna", k): canonical_feature_name("mirna", v)
            for k, v in stored.items()
            if str(k).strip() and str(v).strip()
        }
    return conservative_mirna_alias_map(_raw_annotation_names(mapping, "mirna"))


def _raw_annotation_names(mapping: Mapping, modality: str) -> List[str]:
    values: List[str] = []
    for names in mapping.get("modality_features", {}).get(modality, {}).values():
        values.extend(map(str, names))
    return values


def annotation_feature_sets(
    mapping: Mapping,
    conservative_mirna_alias_harmonization: bool = True,
) -> Dict[str, set[str]]:
    """Return canonical annotation identifiers for each molecular modality.

    When enabled, the miRNA annotation set additionally contains conservative
    precursor aliases only for bases that resolve to exactly one mature arm.
    """
    result = {m: set() for m in MOLECULAR_MODALITIES}
    modality_features = mapping.get("modality_features", {})
    for modality in MOLECULAR_MODALITIES:
        for names in modality_features.get(modality, {}).values():
            result[modality].update(canonical_feature_name(modality, x) for x in names)
    if conservative_mirna_alias_harmonization:
        alias_map = mapping_conservative_mirna_alias_map(mapping)
        result["mirna"].update(alias_map.keys())
    return result


def selected_annotation_coverage(
    mapping: Mapping,
    selected_features: Mapping[str, Sequence[str]],
    conservative_mirna_alias_harmonization: bool = True,
) -> Dict[str, dict]:
    """Compute selected-feature annotation coverage without using outcomes."""
    ann = annotation_feature_sets(mapping, conservative_mirna_alias_harmonization)
    stats: Dict[str, dict] = {}
    for modality in MOLECULAR_MODALITIES:
        selected = {canonical_feature_name(modality, x) for x in selected_features[modality]}
        matched = selected & ann[modality]
        stats[modality] = {
            "selected_features": len(selected),
            "matched_selected_features": len(matched),
            "selected_annotation_coverage_pct": 100.0 * len(matched) / max(len(selected), 1),
        }
    return stats


def validate_selected_annotation_coverage(
    mapping: Mapping,
    selected_features: Mapping[str, Sequence[str]],
    minimum_pct: Mapping[str, float] | None = None,
    conservative_mirna_alias_harmonization: bool = True,
) -> Dict[str, dict]:
    """Fail early when a molecular modality is effectively disconnected from annotations.

    This guard is intentionally outcome-independent. It catches identifier mismatches
    (for example, gene symbols being compared with Entrez IDs) before training starts.
    """
    stats = selected_annotation_coverage(
        mapping,
        selected_features,
        conservative_mirna_alias_harmonization=conservative_mirna_alias_harmonization,
    )
    minimum_pct = minimum_pct or {}
    failures = []
    for modality, record in stats.items():
        threshold = float(minimum_pct.get(modality, 0.0))
        if record["selected_annotation_coverage_pct"] + 1e-12 < threshold:
            failures.append(
                f"{modality}: {record['matched_selected_features']}/{record['selected_features']} "
                f"({record['selected_annotation_coverage_pct']:.2f}%) < required {threshold:.2f}%"
            )
    if failures:
        raise ValueError(
            "Selected-feature pathway annotation coverage is implausibly low. "
            "Check feature identifier formats before training. " + "; ".join(failures)
        )
    return stats


def load_pathway_mapping(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "pathways" not in raw or "modality_features" not in raw:
        raise ValueError("Pathway JSON must contain 'pathways' and 'modality_features'.")
    return raw


def _resolve_caps(
    max_features_per_pathway: int,
    max_features_per_pathway_by_modality: Mapping[str, int] | None,
) -> Dict[str, int]:
    caps = {m: int(max_features_per_pathway) for m in MOLECULAR_MODALITIES}
    if max_features_per_pathway_by_modality:
        for modality, value in max_features_per_pathway_by_modality.items():
            if modality not in caps:
                raise ValueError(f"Unknown modality in max_features_per_pathway_by_modality: {modality}")
            if int(value) < 1:
                raise ValueError("All modality-specific pathway caps must be positive.")
            caps[modality] = int(value)
    return caps


def _mirna_score_lookup(mapping: Mapping, pathway: str, names: Sequence[str]) -> List[float]:
    """Read aligned enrichment scores, falling back to annotation order for legacy maps."""
    metadata = (
        mapping.get("modality_feature_metadata", {})
        .get("mirna", {})
        .get(pathway, {})
    )
    scores = metadata.get("enrichment_score", []) if isinstance(metadata, Mapping) else []
    if len(scores) != len(names):
        # v1.5.3 already stores miRNA names strongest-first. Decreasing synthetic
        # scores preserve that ordering while keeping old pathway JSON compatible.
        return [float(len(names) - i) for i in range(len(names))]
    return [float(x) for x in scores]


def build_pathway_spec(
    mapping: Mapping,
    selected_features: Mapping[str, Sequence[str]],
    min_features_per_pathway: int = 1,
    max_features_per_pathway: int = 128,
    min_modalities_per_pathway: int = 1,
    max_pathways: int | None = None,
    balance_pathway_sources: bool = False,
    max_features_per_pathway_by_modality: Mapping[str, int] | None = None,
    conservative_mirna_alias_harmonization: bool = True,
) -> PathwaySpec:
    """Build fold-specific pathway-index mappings from selected feature names.

    v1.5.4 uses modality-specific feature caps, score-aware top-K selection for
    miRNAs, and normalized multimodal pathway ranking. No survival outcomes enter
    any of these operations.

    For miRNA, the strongest available pathway-enrichment scores are retained when
    a pathway has more matched features than its cap. If a legacy pathway map does
    not contain explicit scores, the builder preserves its existing annotation
    order. Arm-unspecified dataset miRNAs can be matched to an arm-specific
    annotation only when the complete annotation provides exactly one mature arm.
    """
    if min_features_per_pathway < 1:
        raise ValueError("min_features_per_pathway must be positive.")
    if not 1 <= min_modalities_per_pathway <= len(MOLECULAR_MODALITIES):
        raise ValueError("min_modalities_per_pathway is out of range.")

    caps = _resolve_caps(max_features_per_pathway, max_features_per_pathway_by_modality)
    raw_pathways = [str(x) for x in mapping["pathways"]]
    modality_map = mapping["modality_features"]

    # Global annotation context is required for conservative miRNA precursor aliases.
    mirna_aliases: Dict[str, str] = {}
    if conservative_mirna_alias_harmonization:
        mirna_aliases = mapping_conservative_mirna_alias_map(mapping)

    feature_to_index: Dict[str, Dict[str, int]] = {}
    for modality in MOLECULAR_MODALITIES:
        canonical: Dict[str, int] = {}
        for idx, name in enumerate(selected_features[modality]):
            key = canonical_feature_name(modality, name)
            # Keep the first occurrence deterministically if canonicalization
            # collapses duplicate spelling variants.
            canonical.setdefault(key, idx)
        if modality == "mirna" and mirna_aliases:
            # Add the unique mature arm as an alternative lookup key for a dataset
            # feature that omits 3p/5p. Exact mature-arm features always take priority.
            for precursor, mature_arm in mirna_aliases.items():
                if precursor in canonical:
                    canonical.setdefault(mature_arm, canonical[precursor])
        feature_to_index[modality] = canonical

    pathway_sources = mapping.get("metadata", {}).get("pathway_sources", {})
    candidates = []
    selected_denominators = {
        m: max(len({canonical_feature_name(m, x) for x in selected_features[m]}), 1)
        for m in MOLECULAR_MODALITIES
    }

    for raw_order, pathway in enumerate(raw_pathways):
        candidate: Dict[str, List[int]] = {}
        eligible_counts: Dict[str, int] = {}
        total = 0
        modalities_with_matches = 0

        for modality in MOLECULAR_MODALITIES:
            names = list(modality_map.get(modality, {}).get(pathway, []))
            matched: Dict[int, tuple[float, int]] = {}
            mirna_scores = _mirna_score_lookup(mapping, pathway, names) if modality == "mirna" else []
            for order, name in enumerate(names):
                key = canonical_feature_name(modality, name)
                if key not in feature_to_index[modality]:
                    continue
                idx = feature_to_index[modality][key]
                score = mirna_scores[order] if modality == "mirna" else 0.0
                prior = matched.get(idx)
                if prior is None or score > prior[0] or (score == prior[0] and order < prior[1]):
                    matched[idx] = (float(score), int(order))

            eligible_counts[modality] = len(matched)
            if modality == "mirna":
                ordered = sorted(matched.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0]))
                indices = [idx for idx, _ in ordered[:caps[modality]]]
            else:
                ordered = sorted(matched.items(), key=lambda kv: (kv[1][1], kv[0]))
                indices = [idx for idx, _ in ordered[:caps[modality]]]

            candidate[modality] = indices
            total += len(indices)
            if indices:
                modalities_with_matches += 1

        if total < min_features_per_pathway or modalities_with_matches < min_modalities_per_pathway:
            continue

        normalized_by_modality = {
            m: len(candidate[m]) / selected_denominators[m]
            for m in MOLECULAR_MODALITIES
        }
        normalized_score = float(np.mean(list(normalized_by_modality.values())))
        candidates.append({
            "pathway": pathway,
            "indices": candidate,
            "eligible_counts": eligible_counts,
            "total": total,
            "modalities": modalities_with_matches,
            "normalized_coverage": normalized_score,
            "normalized_by_modality": normalized_by_modality,
            "source": str(pathway_sources.get(pathway, "Unknown")),
            "raw_order": raw_order,
        })

    if not candidates:
        raise ValueError("No pathways retained after intersecting the pathway map with selected features.")

    # Equal-weight normalized modality coverage prevents a densely annotated
    # modality from dominating the final pathway-token selection. Survival labels
    # and event times are never used.
    ranked = sorted(
        candidates,
        key=lambda r: (-r["modalities"], -r["normalized_coverage"], r["pathway"]),
    )
    if max_pathways is not None and len(ranked) > max_pathways:
        if balance_pathway_sources:
            sources = sorted({r["source"] for r in ranked if r["source"] != "Unknown"})
            if len(sources) > 1:
                quota = max_pathways // len(sources)
                chosen = []
                chosen_ids = set()
                for source in sources:
                    items = [r for r in ranked if r["source"] == source][:quota]
                    chosen.extend(items)
                    chosen_ids.update(r["pathway"] for r in items)
                for record in ranked:
                    if len(chosen) >= max_pathways:
                        break
                    if record["pathway"] not in chosen_ids:
                        chosen.append(record)
                        chosen_ids.add(record["pathway"])
                ranked = chosen
            else:
                ranked = ranked[:max_pathways]
        else:
            ranked = ranked[:max_pathways]

    # Restore raw database order after selection to make adjacency extraction stable.
    ranked = sorted(ranked, key=lambda r: r["raw_order"])
    retained_pathways = [r["pathway"] for r in ranked]
    indices_by_modality: Dict[str, List[List[int]]] = {m: [] for m in MOLECULAR_MODALITIES}
    eligible_by_modality: Dict[str, List[int]] = {m: [] for m in MOLECULAR_MODALITIES}
    selection_scores: List[float] = []
    for record in ranked:
        selection_scores.append(float(record["normalized_coverage"]))
        for modality in MOLECULAR_MODALITIES:
            indices_by_modality[modality].append(record["indices"][modality])
            eligible_by_modality[modality].append(int(record["eligible_counts"][modality]))

    mapped_by_modality = {m: set() for m in MOLECULAR_MODALITIES}
    for modality in MOLECULAR_MODALITIES:
        for indices in indices_by_modality[modality]:
            mapped_by_modality[modality].update(indices)
    residual = {
        modality: [idx for idx in range(len(selected_features[modality])) if idx not in mapped_by_modality[modality]]
        for modality in MOLECULAR_MODALITIES
    }

    raw_adjacency = np.asarray(mapping.get("adjacency", np.eye(len(raw_pathways))), dtype=np.float32)
    if raw_adjacency.shape != (len(raw_pathways), len(raw_pathways)):
        raise ValueError("Pathway adjacency must be square and match the raw pathway count.")
    raw_index = {name: idx for idx, name in enumerate(raw_pathways)}
    keep = [raw_index[name] for name in retained_pathways]
    adjacency = raw_adjacency[np.ix_(keep, keep)].astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)

    return PathwaySpec(
        pathway_names=retained_pathways,
        feature_names={m: list(map(str, selected_features[m])) for m in MOLECULAR_MODALITIES},
        feature_indices=indices_by_modality,
        residual_indices=residual,
        adjacency=adjacency,
        eligible_feature_counts=eligible_by_modality,
        pathway_selection_scores=selection_scores,
    )
