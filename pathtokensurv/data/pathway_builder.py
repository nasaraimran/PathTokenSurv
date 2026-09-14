from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


_MIRNA_ARM_RE = re.compile(r"^(?P<base>.+)-(?:3p|5p)$", flags=re.IGNORECASE)


def _conservative_mirna_aliases_from_names(names) -> dict[str, str]:
    """Unique-arm aliases derived from the complete provider miRNA name set."""
    canonical = {str(x).strip().lower() for x in names if str(x).strip()}
    by_base: dict[str, set[str]] = {}
    for name in canonical:
        match = _MIRNA_ARM_RE.match(name)
        if match:
            by_base.setdefault(match.group("base"), set()).add(name)
    return {
        base: next(iter(arms))
        for base, arms in by_base.items()
        if len(arms) == 1 and base not in canonical
    }

def _clean_entrez(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith('.0') and text[:-2].isdigit() else text


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment for a one-dimensional p-value array."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.size == 0:
        return pvalues
    order = np.argsort(pvalues, kind="mergesort")
    ranked = pvalues[order]
    adjusted = ranked * pvalues.size / np.arange(1, pvalues.size + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _enrichment_score(qvalue: float) -> float:
    """Numerically stable score used only to rank significant miRNA links."""
    return float(-math.log10(max(float(qvalue), np.finfo(float).tiny)))


def build_pathway_json_from_tables(
    gene_membership_file: str | Path,
    output_file: str | Path,
    mirna_targets_file: str | Path | None = None,
    pathway_edges_file: str | Path | None = None,
    delimiter: str = "\t",
    min_genes_per_pathway: int = 1,
    max_genes_per_pathway: int | None = None,
    mirna_pathway_mode: str = "enrichment",
    mirna_min_target_overlap: int = 3,
    mirna_enrichment_fdr: float = 0.05,
) -> dict:
    """Build PathTokenSurv ``pathways.json`` from local annotation tables.

    Gene membership requires ``pathway_id``, ``gene_symbol`` and ``entrez_id``.
    Optional ``pathway_name`` is retained as metadata. The pathway-size filter is
    applied before modality mappings are created. Size is based on the number of
    unique annotated genes in a pathway (unique Entrez IDs where available,
    otherwise gene symbols).

    miRNA targets require ``mirna`` and ``gene_symbol``. By default, miRNA-to-
    pathway links are retained only when target genes are enriched in a pathway
    (minimum target overlap plus Benjamini-Hochberg FDR correction performed
    separately for each miRNA). ``mirna_pathway_mode='any_target'`` reproduces the
    legacy permissive mapping for ablation/sensitivity analysis.

    v1.5.4 stores the enrichment score, q-value, and target overlap aligned with
    every retained miRNA-pathway link. The fold-specific builder can therefore
    select the strongest top-K miRNAs for each pathway rather than truncating an
    arbitrary list.

    mRNA pathway membership stores both Entrez IDs and gene-symbol aliases because
    TCGA PanCancer exports occur in both identifier spaces. This function performs
    no online download so exact database releases can be archived with the study.
    """
    if min_genes_per_pathway < 1:
        raise ValueError("min_genes_per_pathway must be >= 1")
    if max_genes_per_pathway is not None and max_genes_per_pathway < min_genes_per_pathway:
        raise ValueError("max_genes_per_pathway must be >= min_genes_per_pathway")
    mirna_pathway_mode = str(mirna_pathway_mode).strip().lower()
    if mirna_pathway_mode not in {"enrichment", "any_target"}:
        raise ValueError("mirna_pathway_mode must be 'enrichment' or 'any_target'")
    if mirna_min_target_overlap < 1:
        raise ValueError("mirna_min_target_overlap must be >= 1")
    if not 0.0 < float(mirna_enrichment_fdr) <= 1.0:
        raise ValueError("mirna_enrichment_fdr must lie in (0, 1]")

    genes = pd.read_csv(gene_membership_file, sep=delimiter, dtype=str).fillna("")
    required = {"pathway_id", "gene_symbol", "entrez_id"}
    missing = required - set(genes.columns)
    if missing:
        raise ValueError(f"Gene membership table is missing columns: {sorted(missing)}")
    genes["pathway_id"] = genes["pathway_id"].str.strip()
    genes["gene_symbol"] = genes["gene_symbol"].str.strip()
    genes["entrez_id"] = genes["entrez_id"].map(_clean_entrez)
    if "source" in genes.columns:
        genes["source"] = genes["source"].astype(str).str.strip()
    genes = genes[(genes["pathway_id"] != "") & ((genes["gene_symbol"] != "") | (genes["entrez_id"] != ""))]

    # Count unique genes per pathway. Prefer Entrez ID but fall back to symbol.
    genes = genes.copy()
    genes["__gene_key"] = np.where(
        genes["entrez_id"] != "",
        "E:" + genes["entrez_id"],
        "S:" + genes["gene_symbol"].str.upper(),
    )
    pathway_sizes = genes.groupby("pathway_id")["__gene_key"].nunique().sort_index()
    keep = pathway_sizes[pathway_sizes >= int(min_genes_per_pathway)]
    if max_genes_per_pathway is not None:
        keep = keep[keep <= int(max_genes_per_pathway)]
    retained_set = set(keep.index.astype(str))
    genes_retained = genes[genes["pathway_id"].isin(retained_set)].copy()
    pathways = list(dict.fromkeys(genes_retained["pathway_id"].tolist()))
    if not pathways:
        raise ValueError("Pathway-size filtering removed every pathway.")

    modality_features = {"mrna": {}, "mirna": {}, "cnv": {}}
    # Aligned arrays keep the JSON compact while making score-aware top-K explicit.
    modality_feature_metadata = {"mirna": {}}
    for pathway in pathways:
        group = genes_retained.loc[genes_retained["pathway_id"] == pathway]
        entrez_aliases = {x for x in group["entrez_id"] if x}
        symbol_aliases = {str(x).upper() for x in group["gene_symbol"] if x}
        modality_features["mrna"][pathway] = sorted(entrez_aliases | symbol_aliases)
        modality_features["cnv"][pathway] = sorted(symbol_aliases)
        modality_features["mirna"][pathway] = []
        modality_feature_metadata["mirna"][pathway] = {
            "enrichment_score": [],
            "q_value": [],
            "target_overlap": [],
        }

    mirna_metadata = {
        "mirna_target_rows": 0,
        "unique_mirnas": 0,
        "unique_target_genes": 0,
        "target_genes_overlapping_retained_pathways": 0,
        "target_genes_not_in_retained_pathways": 0,
        "mirna_pathway_mode": mirna_pathway_mode,
        "mirna_min_target_overlap": int(mirna_min_target_overlap),
        "mirna_enrichment_fdr": float(mirna_enrichment_fdr),
        "mirna_enrichment_score_definition": "-log10(BH-adjusted q-value); target-overlap descending then miRNA identifier for ties",
        "mirna_pathway_candidate_links": 0,
        "mirna_pathway_retained_links": 0,
        "mirna_pathways_with_links": 0,
        "mirna_pathway_links_median_per_pathway": 0.0,
        "mirna_pathway_links_max_per_pathway": 0,
        "mirna_conservative_aliases": {},
        "mirna_conservative_alias_count": 0,
    }
    if mirna_targets_file is not None:
        targets = pd.read_csv(
            mirna_targets_file, sep=delimiter, dtype=str,
            usecols=lambda c: c in {"mirna", "gene_symbol"},
        ).fillna("")
        req = {"mirna", "gene_symbol"}
        missing = req - set(targets.columns)
        if missing:
            raise ValueError(f"miRNA target table is missing columns: {sorted(missing)}")
        targets["mirna"] = targets["mirna"].str.strip()
        targets["__mirna_lower"] = targets["mirna"].str.lower()
        targets["gene_symbol"] = targets["gene_symbol"].str.strip().str.upper()
        targets = targets[(targets["__mirna_lower"] != "") & (targets["gene_symbol"] != "")].copy()
        targets = targets.drop_duplicates(["__mirna_lower", "gene_symbol"])
        targets["__gene_upper"] = targets["gene_symbol"]
        mirna_display = targets.drop_duplicates("__mirna_lower").set_index("__mirna_lower")["mirna"].to_dict()

        conservative_aliases = _conservative_mirna_aliases_from_names(targets["__mirna_lower"].unique())

        retained_symbols = set(genes_retained["gene_symbol"].str.upper()) - {""}
        target_symbols = set(targets["__gene_upper"])
        universe = retained_symbols & target_symbols

        pathway_gene_sets = {
            pathway: (set(genes_retained.loc[genes_retained["pathway_id"] == pathway, "gene_symbol"].str.upper()) - {""}) & universe
            for pathway in pathways
        }
        gene_to_pathways: dict[str, list[str]] = {}
        for pathway, symbols in pathway_gene_sets.items():
            for gene in symbols:
                gene_to_pathways.setdefault(gene, []).append(pathway)

        targets_u = targets[targets["__gene_upper"].isin(universe)].copy()
        target_sets = targets_u.groupby("__mirna_lower")["__gene_upper"].agg(lambda x: set(x))

        candidate_links = 0
        retained_links = 0
        # tuple = (score, qvalue, overlap, display_name)
        per_pathway_records: dict[str, list[tuple[float, float, int, str]]] = {p: [] for p in pathways}

        if mirna_pathway_mode == "any_target":
            for mirna, gene_set in target_sets.items():
                counts: dict[str, int] = {}
                for gene in gene_set:
                    for pathway in gene_to_pathways.get(gene, []):
                        counts[pathway] = counts.get(pathway, 0) + 1
                for pathway, overlap in counts.items():
                    if overlap >= mirna_min_target_overlap:
                        candidate_links += 1
                        retained_links += 1
                        display = str(mirna_display.get(str(mirna), mirna))
                        # For the legacy rule, target overlap is the only available strength.
                        per_pathway_records[pathway].append((float(overlap), 1.0, int(overlap), display))
        else:
            population = len(universe)
            pathway_sizes_u = {p: len(g) for p, g in pathway_gene_sets.items()}
            for mirna, gene_set in target_sets.items():
                n_targets = len(gene_set)
                if n_targets < mirna_min_target_overlap or population == 0:
                    continue
                counts: dict[str, int] = {}
                for gene in gene_set:
                    for pathway in gene_to_pathways.get(gene, []):
                        counts[pathway] = counts.get(pathway, 0) + 1
                candidates = [
                    (pathway, overlap) for pathway, overlap in counts.items()
                    if overlap >= mirna_min_target_overlap and pathway_sizes_u[pathway] > 0
                ]
                if not candidates:
                    continue
                candidate_links += len(candidates)
                pvals = np.asarray([
                    hypergeom.sf(overlap - 1, population, pathway_sizes_u[pathway], n_targets)
                    for pathway, overlap in candidates
                ], dtype=float)
                qvals = _bh_adjust(pvals)
                for (pathway, overlap), qvalue in zip(candidates, qvals):
                    if float(qvalue) <= float(mirna_enrichment_fdr):
                        retained_links += 1
                        display = str(mirna_display.get(str(mirna), mirna))
                        per_pathway_records[pathway].append(
                            (_enrichment_score(float(qvalue)), float(qvalue), int(overlap), display)
                        )

        pathway_link_counts = []
        for pathway in pathways:
            # Strongest enrichment score first; target overlap and identifier are
            # deterministic tie-breakers. The exact score is retained in JSON.
            records = sorted(per_pathway_records[pathway], key=lambda r: (-r[0], -r[2], r[3].lower()))
            modality_features["mirna"][pathway] = [r[3] for r in records]
            modality_feature_metadata["mirna"][pathway] = {
                "enrichment_score": [float(r[0]) for r in records],
                "q_value": [float(r[1]) for r in records],
                "target_overlap": [int(r[2]) for r in records],
            }
            pathway_link_counts.append(len(records))

        mirna_metadata = {
            "mirna_target_rows": int(len(targets)),
            "unique_mirnas": int(targets["__mirna_lower"].nunique()),
            "unique_target_genes": int(len(target_symbols)),
            "target_genes_overlapping_retained_pathways": int(len(universe)),
            "target_genes_not_in_retained_pathways": int(len(target_symbols - retained_symbols)),
            "mirna_pathway_mode": mirna_pathway_mode,
            "mirna_min_target_overlap": int(mirna_min_target_overlap),
            "mirna_enrichment_fdr": float(mirna_enrichment_fdr),
            "mirna_enrichment_universe_genes": int(len(universe)),
            "mirna_enrichment_score_definition": "-log10(BH-adjusted q-value); target-overlap descending then miRNA identifier for ties",
            "mirna_pathway_candidate_links": int(candidate_links),
            "mirna_pathway_retained_links": int(retained_links),
            "mirna_pathways_with_links": int(sum(x > 0 for x in pathway_link_counts)),
            "mirna_pathway_links_median_per_pathway": float(np.median(pathway_link_counts)) if pathway_link_counts else 0.0,
            "mirna_pathway_links_max_per_pathway": int(max(pathway_link_counts)) if pathway_link_counts else 0,
            "mirna_conservative_aliases": conservative_aliases,
            "mirna_conservative_alias_count": int(len(conservative_aliases)),
        }

    adjacency = np.zeros((len(pathways), len(pathways)), dtype=np.float32)
    retained_edge_count = 0
    if pathway_edges_file is not None:
        edges = pd.read_csv(pathway_edges_file, sep=delimiter, dtype=str).fillna("")
        req = {"source_pathway", "target_pathway"}
        missing = req - set(edges.columns)
        if missing:
            raise ValueError(f"Pathway edge table is missing columns: {sorted(missing)}")
        pidx = {p: i for i, p in enumerate(pathways)}
        for source, target in edges[["source_pathway", "target_pathway"]].itertuples(index=False, name=None):
            source, target = str(source).strip(), str(target).strip()
            if source in pidx and target in pidx and source != target:
                if adjacency[pidx[source], pidx[target]] == 0:
                    retained_edge_count += 1
                adjacency[pidx[source], pidx[target]] = 1.0
                adjacency[pidx[target], pidx[source]] = 1.0

    metadata = {
        "source_gene_membership_rows": int(len(genes)),
        "source_unique_pathways": int(pathway_sizes.size),
        "retained_pathways_after_size_filter": int(len(pathways)),
        "pathways_removed_by_size_filter": int(pathway_sizes.size - len(pathways)),
        "min_genes_per_pathway": int(min_genes_per_pathway),
        "max_genes_per_pathway": None if max_genes_per_pathway is None else int(max_genes_per_pathway),
        "retained_pathway_gene_size_min": int(keep.min()),
        "retained_pathway_gene_size_median": float(keep.median()),
        "retained_pathway_gene_size_max": int(keep.max()),
        "unique_gene_symbols": int(genes_retained.loc[genes_retained['gene_symbol'] != '', 'gene_symbol'].str.upper().nunique()),
        "unique_entrez_ids": int(genes_retained.loc[genes_retained['entrez_id'] != '', 'entrez_id'].nunique()),
        "mrna_identifier_alias_mode": "gene_symbol_or_entrez",
        "retained_pathway_edges": int(retained_edge_count),
        **mirna_metadata,
    }
    if "pathway_name" in genes_retained.columns:
        metadata["pathway_names"] = {
            p: next((x for x in genes_retained.loc[genes_retained["pathway_id"] == p, "pathway_name"] if x), p)
            for p in pathways
        }
    if "source" in genes_retained.columns:
        metadata["pathway_sources"] = {
            p: next((x for x in genes_retained.loc[genes_retained["pathway_id"] == p, "source"] if x), "Unknown")
            for p in pathways
        }
        metadata["retained_pathways_by_source"] = {
            str(k): int(v) for k, v in genes_retained.drop_duplicates("pathway_id").groupby("source")["pathway_id"].nunique().items()
        }
    metadata["pathway_gene_sizes"] = {str(k): int(v) for k, v in keep.items()}

    payload = {
        "pathways": pathways,
        "modality_features": modality_features,
        "modality_feature_metadata": modality_feature_metadata,
        "adjacency": adjacency.tolist(),
        "metadata": metadata,
    }
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
