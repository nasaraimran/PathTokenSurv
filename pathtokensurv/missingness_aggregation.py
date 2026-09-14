from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from pathtokensurv.utils.io import save_json


METRICS = [
    "c_index",
    "cancer_stratified_c_index",
    "ibs",
    "mean_time_dependent_auc",
]

DROP_CONDITION_ORDER = [
    "all_observed",
    "drop_mrna",
    "drop_mirna",
    "drop_cnv",
    "clinical_only",
]


def _mean_ci(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(stats.t.ppf(0.975, df=n - 1) * sd / np.sqrt(n))
        low, high = mean - half, mean + half
    else:
        low = high = float("nan")
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _fold_number(fold_dir: Path, fallback: int) -> int:
    manifest = fold_dir / "experiment_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        value = payload.get("outer_fold")
        if value is not None:
            return int(value)
    match = re.search(r"fold[_-]?(\d+)", fold_dir.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else int(fallback)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _wide_group_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for key, block in frame.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(
            {
                "folds_present": int(block["fold"].nunique()),
                "total_patients": int(block["patients"].sum()),
                "total_events": int(block["events"].sum()),
                "mean_patients_per_fold": float(block["patients"].mean()),
                "mean_events_per_fold": float(block["events"].mean()),
                "min_patients_per_fold": int(block["patients"].min()),
                "min_events_per_fold": int(block["events"].min()),
            }
        )
        for metric in metrics:
            summary = _mean_ci(block[metric].to_numpy(dtype=float))
            row.update({f"{metric}_{name}": value for name, value in summary.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _long_delta_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        summary = _mean_ci(frame[column].to_numpy(dtype=float))
        rows.append({"contrast": column, **summary})
    return pd.DataFrame(rows)


def _format_mean_sd(mean: float, sd: float) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(sd):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {sd:.4f}"


def _write_markdown_report(
    natural_summary: pd.DataFrame,
    natural_gap_summary: pd.DataFrame,
    dropout_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# PathTokenSurv v1.5.8 — Missingness Robustness Summary",
        "",
        "This report aggregates held-out outer-fold results only. No model training, checkpoint selection, preprocessing, or pathway construction is performed by the aggregator.",
        "",
        "## Natural missingness (observational groups)",
        "",
        "Complete and incomplete groups contain different patients. Their differences are descriptive and must not be interpreted as causal test-time ablations.",
        "",
        "| Group | Folds | Total N | Events | Pooled C | Stratified C | IBS | Mean tdAUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    natural_agg = natural_summary[natural_summary["group_type"] == "aggregate"].copy()
    order = {"complete": 0, "incomplete": 1}
    natural_agg["_order"] = natural_agg["group"].map(order).fillna(99)
    for _, row in natural_agg.sort_values("_order").iterrows():
        lines.append(
            "| {group} | {folds} | {n} | {events} | {c} | {sc} | {ibs} | {auc} |".format(
                group=str(row["group"]),
                folds=int(row["folds_present"]),
                n=int(row["total_patients"]),
                events=int(row["total_events"]),
                c=_format_mean_sd(row["c_index_mean"], row["c_index_sd"]),
                sc=_format_mean_sd(
                    row["cancer_stratified_c_index_mean"],
                    row["cancer_stratified_c_index_sd"],
                ),
                ibs=_format_mean_sd(row["ibs_mean"], row["ibs_sd"]),
                auc=_format_mean_sd(
                    row["mean_time_dependent_auc_mean"],
                    row["mean_time_dependent_auc_sd"],
                ),
            )
        )

    lines.extend(
        [
            "",
            "### Complete-versus-incomplete observational fold gaps",
            "",
            "Positive values for C-index/tdAUC contrasts mean the complete group scored higher. Positive `incomplete_minus_complete_ibs` means the incomplete group had worse (higher) IBS.",
            "",
            "| Contrast | Mean ± SD | 95% cross-fold CI |",
            "|---|---:|---:|",
        ]
    )
    for _, row in natural_gap_summary.iterrows():
        lines.append(
            f"| {row['contrast']} | {_format_mean_sd(row['mean'], row['sd'])} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )

    lines.extend(
        [
            "",
            "## Controlled complete-case modality dropout",
            "",
            "All conditions use the same naturally complete held-out patients within each fold. Degradation is defined so positive values always mean worse performance after removing modalities.",
            "",
            "| Condition | Folds | Total N | Pooled C | Stratified C | IBS | Mean tdAUC | ΔC degradation | Δstrat C degradation | ΔIBS degradation | ΔtdAUC degradation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    order = {name: i for i, name in enumerate(DROP_CONDITION_ORDER)}
    dropout_summary = dropout_summary.copy()
    dropout_summary["_order"] = dropout_summary["condition"].map(order).fillna(99)
    for _, row in dropout_summary.sort_values("_order").iterrows():
        lines.append(
            "| {condition} | {folds} | {n} | {c} | {sc} | {ibs} | {auc} | {dc} | {dsc} | {dibs} | {dauc} |".format(
                condition=row["condition"],
                folds=int(row["folds_present"]),
                n=int(row["total_patients"]),
                c=_format_mean_sd(row["c_index_mean"], row["c_index_sd"]),
                sc=_format_mean_sd(
                    row["cancer_stratified_c_index_mean"],
                    row["cancer_stratified_c_index_sd"],
                ),
                ibs=_format_mean_sd(row["ibs_mean"], row["ibs_sd"]),
                auc=_format_mean_sd(
                    row["mean_time_dependent_auc_mean"],
                    row["mean_time_dependent_auc_sd"],
                ),
                dc=_format_mean_sd(
                    row["degradation_c_index_mean"], row["degradation_c_index_sd"]
                ),
                dsc=_format_mean_sd(
                    row["degradation_cancer_stratified_c_index_mean"],
                    row["degradation_cancer_stratified_c_index_sd"],
                ),
                dibs=_format_mean_sd(
                    row["degradation_ibs_mean"], row["degradation_ibs_sd"]
                ),
                dauc=_format_mean_sd(
                    row["degradation_mean_time_dependent_auc_mean"],
                    row["degradation_mean_time_dependent_auc_sd"],
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Statistical note",
            "",
            "Means, standard deviations, and two-sided 95% Student-t intervals are computed across outer-fold point estimates. These are cross-fold intervals, not patient-level bootstrap intervals and not multi-seed ensemble confidence intervals.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_missingness_robustness(
    fold_dirs: Iterable[str | Path], output_dir: str | Path
) -> dict[str, pd.DataFrame]:
    fold_paths = [Path(path) for path in fold_dirs]
    if not fold_paths:
        raise ValueError("At least one fold directory is required.")

    natural_blocks = []
    dropout_blocks = []
    fold_ids = []
    for position, fold_dir in enumerate(fold_paths, start=1):
        fold = _fold_number(fold_dir, position)
        fold_ids.append(fold)
        natural_path = fold_dir / "natural_missingness_metrics.csv"
        dropout_path = fold_dir / "complete_case_modality_drop_metrics.csv"
        if not natural_path.exists():
            raise FileNotFoundError(f"Missing natural-missingness metrics: {natural_path}")
        if not dropout_path.exists():
            raise FileNotFoundError(f"Missing controlled-dropout metrics: {dropout_path}")

        natural = pd.read_csv(natural_path)
        dropout = pd.read_csv(dropout_path)
        _require_columns(
            natural,
            ["group_type", "group", "pattern_name", "patients", "events", *METRICS],
            natural_path,
        )
        _require_columns(
            dropout,
            ["condition", "dropped_modalities", "patients", "events", *METRICS],
            dropout_path,
        )
        natural.insert(0, "fold", fold)
        dropout.insert(0, "fold", fold)
        natural_blocks.append(natural)
        dropout_blocks.append(dropout)

    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError(f"Duplicate fold IDs detected: {fold_ids}")

    natural_fold = pd.concat(natural_blocks, ignore_index=True).sort_values(
        ["fold", "group_type", "group"]
    )
    dropout_fold = pd.concat(dropout_blocks, ignore_index=True).sort_values(
        ["fold", "condition"]
    )

    # Validate aggregate natural groups and exact-pattern accounting per fold.
    for fold, block in natural_fold.groupby("fold"):
        aggregate = block[block["group_type"] == "aggregate"].set_index("group")
        for required in ["complete", "incomplete"]:
            if required not in aggregate.index:
                raise ValueError(f"Fold {fold} lacks natural aggregate group '{required}'.")
        exact = block[block["group_type"] == "exact_pattern"]
        if int(exact["patients"].sum()) != int(aggregate.loc["complete", "patients"] + aggregate.loc["incomplete", "patients"]):
            raise ValueError(f"Fold {fold} exact-pattern patient counts do not sum to the aggregate test count.")

    # Validate controlled experiment identity of patient composition and all-observed reference.
    degradation_rows = []
    for fold, block in dropout_fold.groupby("fold"):
        if "all_observed" not in set(block["condition"]):
            raise ValueError(f"Fold {fold} controlled dropout table lacks all_observed.")
        if block["patients"].nunique() != 1 or block["events"].nunique() != 1:
            raise ValueError(f"Fold {fold} controlled dropout conditions do not use identical patients/events.")
        reference = block.loc[block["condition"] == "all_observed"].iloc[0]
        # Cross-check natural complete-case metrics with controlled all-observed.
        natural_complete = natural_fold[
            (natural_fold["fold"] == fold)
            & (natural_fold["group_type"] == "aggregate")
            & (natural_fold["group"] == "complete")
        ]
        if len(natural_complete) != 1:
            raise ValueError(f"Fold {fold} must contain exactly one natural complete aggregate row.")
        natural_complete = natural_complete.iloc[0]
        for metric in METRICS:
            left, right = float(reference[metric]), float(natural_complete[metric])
            if np.isfinite(left) and np.isfinite(right) and not np.isclose(left, right, rtol=1e-6, atol=1e-8):
                raise ValueError(
                    f"Fold {fold} all_observed {metric} ({left}) does not match natural complete {metric} ({right})."
                )
        for _, row in block.iterrows():
            payload = row.to_dict()
            payload.update(
                {
                    "degradation_c_index": float(reference["c_index"] - row["c_index"]),
                    "degradation_cancer_stratified_c_index": float(
                        reference["cancer_stratified_c_index"]
                        - row["cancer_stratified_c_index"]
                    ),
                    "degradation_ibs": float(row["ibs"] - reference["ibs"]),
                    "degradation_mean_time_dependent_auc": float(
                        reference["mean_time_dependent_auc"]
                        - row["mean_time_dependent_auc"]
                    ),
                }
            )
            degradation_rows.append(payload)
    dropout_degradation = pd.DataFrame(degradation_rows)

    natural_summary = _wide_group_summary(
        natural_fold,
        ["group_type", "group", "pattern_name"],
        METRICS,
    )

    # Fold-wise complete vs incomplete contrasts are observational because patient composition differs.
    natural_gap_rows = []
    for fold, block in natural_fold.groupby("fold"):
        aggregate = block[block["group_type"] == "aggregate"].set_index("group")
        complete = aggregate.loc["complete"]
        incomplete = aggregate.loc["incomplete"]
        natural_gap_rows.append(
            {
                "fold": int(fold),
                "complete_patients": int(complete["patients"]),
                "incomplete_patients": int(incomplete["patients"]),
                "complete_events": int(complete["events"]),
                "incomplete_events": int(incomplete["events"]),
                "complete_minus_incomplete_c_index": float(
                    complete["c_index"] - incomplete["c_index"]
                ),
                "complete_minus_incomplete_cancer_stratified_c_index": float(
                    complete["cancer_stratified_c_index"]
                    - incomplete["cancer_stratified_c_index"]
                ),
                "incomplete_minus_complete_ibs": float(incomplete["ibs"] - complete["ibs"]),
                "complete_minus_incomplete_mean_time_dependent_auc": float(
                    complete["mean_time_dependent_auc"]
                    - incomplete["mean_time_dependent_auc"]
                ),
            }
        )
    natural_gaps = pd.DataFrame(natural_gap_rows).sort_values("fold")
    natural_gap_columns = [
        "complete_minus_incomplete_c_index",
        "complete_minus_incomplete_cancer_stratified_c_index",
        "incomplete_minus_complete_ibs",
        "complete_minus_incomplete_mean_time_dependent_auc",
    ]
    natural_gap_summary = _long_delta_summary(natural_gaps, natural_gap_columns)

    dropout_summary = _wide_group_summary(
        dropout_degradation,
        ["condition", "dropped_modalities"],
        METRICS
        + [
            "degradation_c_index",
            "degradation_cancer_stratified_c_index",
            "degradation_ibs",
            "degradation_mean_time_dependent_auc",
        ],
    )

    # Publication-oriented compact table; exact patterns remain in the supplementary summary.
    publication_rows = []
    natural_agg = natural_summary[natural_summary["group_type"] == "aggregate"]
    for _, row in natural_agg.iterrows():
        publication_rows.append(
            {
                "panel": "natural_missingness_observational",
                "condition": row["group"],
                "folds": row["folds_present"],
                "total_patients": row["total_patients"],
                "total_events": row["total_events"],
                "c_index_mean": row["c_index_mean"],
                "c_index_sd": row["c_index_sd"],
                "cancer_stratified_c_index_mean": row["cancer_stratified_c_index_mean"],
                "cancer_stratified_c_index_sd": row["cancer_stratified_c_index_sd"],
                "ibs_mean": row["ibs_mean"],
                "ibs_sd": row["ibs_sd"],
                "mean_time_dependent_auc_mean": row["mean_time_dependent_auc_mean"],
                "mean_time_dependent_auc_sd": row["mean_time_dependent_auc_sd"],
                "degradation_c_index_mean": np.nan,
                "degradation_cancer_stratified_c_index_mean": np.nan,
                "degradation_ibs_mean": np.nan,
                "degradation_mean_time_dependent_auc_mean": np.nan,
            }
        )
    for _, row in dropout_summary.iterrows():
        publication_rows.append(
            {
                "panel": "controlled_complete_case_dropout",
                "condition": row["condition"],
                "folds": row["folds_present"],
                "total_patients": row["total_patients"],
                "total_events": row["total_events"],
                "c_index_mean": row["c_index_mean"],
                "c_index_sd": row["c_index_sd"],
                "cancer_stratified_c_index_mean": row["cancer_stratified_c_index_mean"],
                "cancer_stratified_c_index_sd": row["cancer_stratified_c_index_sd"],
                "ibs_mean": row["ibs_mean"],
                "ibs_sd": row["ibs_sd"],
                "mean_time_dependent_auc_mean": row["mean_time_dependent_auc_mean"],
                "mean_time_dependent_auc_sd": row["mean_time_dependent_auc_sd"],
                "degradation_c_index_mean": row["degradation_c_index_mean"],
                "degradation_cancer_stratified_c_index_mean": row[
                    "degradation_cancer_stratified_c_index_mean"
                ],
                "degradation_ibs_mean": row["degradation_ibs_mean"],
                "degradation_mean_time_dependent_auc_mean": row[
                    "degradation_mean_time_dependent_auc_mean"
                ],
            }
        )
    publication = pd.DataFrame(publication_rows)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    natural_fold.to_csv(out / "natural_missingness_fold_metrics.csv", index=False)
    natural_summary.to_csv(out / "natural_missingness_summary.csv", index=False)
    natural_gaps.to_csv(out / "natural_complete_vs_incomplete_fold_gaps.csv", index=False)
    natural_gap_summary.to_csv(out / "natural_complete_vs_incomplete_gap_summary.csv", index=False)
    dropout_degradation.to_csv(out / "complete_case_modality_drop_fold_metrics.csv", index=False)
    dropout_summary.to_csv(out / "complete_case_modality_drop_summary.csv", index=False)
    publication.to_csv(out / "missingness_robustness_publication_table.csv", index=False)
    _write_markdown_report(
        natural_summary,
        natural_gap_summary,
        dropout_summary,
        out / "MISSINGNESS_ROBUSTNESS_SUMMARY.md",
    )
    save_json(
        {
            "status": "PASS",
            "folds": sorted(int(v) for v in fold_ids),
            "fold_count": int(len(fold_ids)),
            "training_performed": False,
            "checkpoint_modified": False,
            "primary_protocol_modified": False,
            "natural_missingness_interpretation": (
                "Observational complete/incomplete and exact-pattern groups contain different patients; "
                "between-group gaps are descriptive, not causal ablations."
            ),
            "controlled_dropout_interpretation": (
                "All controlled conditions use identical naturally complete held-out patients within each fold; "
                "positive degradation values indicate worse performance after modality removal."
            ),
            "ci_method": "two-sided 95% Student-t interval across outer-fold point estimates",
            "outputs": [
                "natural_missingness_fold_metrics.csv",
                "natural_missingness_summary.csv",
                "natural_complete_vs_incomplete_fold_gaps.csv",
                "natural_complete_vs_incomplete_gap_summary.csv",
                "complete_case_modality_drop_fold_metrics.csv",
                "complete_case_modality_drop_summary.csv",
                "missingness_robustness_publication_table.csv",
                "MISSINGNESS_ROBUSTNESS_SUMMARY.md",
            ],
        },
        out / "missingness_aggregation_manifest.json",
    )

    return {
        "natural_fold": natural_fold,
        "natural_summary": natural_summary,
        "natural_gaps": natural_gaps,
        "natural_gap_summary": natural_gap_summary,
        "dropout_fold": dropout_degradation,
        "dropout_summary": dropout_summary,
        "publication": publication,
    }
