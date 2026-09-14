import argparse
from pathlib import Path
import shutil, zipfile

EXPECTED = {
    "outer": ["outer_cv_summary.csv", "outer_fold_metrics.csv", "baseline_fold_metrics.csv"],
    "missing": [
        "missingness_robustness_publication_table.csv",
        "natural_missingness_summary.csv",
        "natural_missingness_fold_metrics.csv",
        "complete_case_modality_drop_summary.csv",
        "complete_case_modality_drop_fold_metrics.csv",
        "natural_complete_vs_incomplete_gap_summary.csv",
    ],
    "ablation": [
        "architecture_ablation_publication_table.csv",
        "paired_fold_differences.csv",
        "primary_fold_metrics.csv",
        "ablation_fold_metrics.csv",
    ],
    "pathway": [
        "pathway_token_control_publication_table.csv",
        "pathway_token_control_summary.csv",
        "paired_fold_differences.csv",
        "primary_fold_metrics.csv",
        "control_fold_metrics.csv",
    ],
}

RENAME = {
    ("ablation","paired_fold_differences.csv"): "ablation_paired_fold_differences.csv",
    ("ablation","primary_fold_metrics.csv"): "ablation_primary_fold_metrics.csv",
    ("pathway","paired_fold_differences.csv"): "pathway_control_paired_fold_differences.csv",
    ("pathway","primary_fold_metrics.csv"): "pathway_control_primary_fold_metrics.csv",
}

def extract(key, zpath, outdir):
    with zipfile.ZipFile(zpath) as z:
        found = set()
        for member in z.namelist():
            base = Path(member).name
            if base in EXPECTED[key]:
                target = RENAME.get((key,base), base)
                with z.open(member) as src, open(outdir/target, "wb") as dst:
                    shutil.copyfileobj(src,dst)
                found.add(base)
        missing = set(EXPECTED[key]) - found
        if missing:
            print(f"WARNING {key}: missing {sorted(missing)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", required=True)
    ap.add_argument("--missingness", required=True)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--pathway-control", required=True)
    ap.add_argument("--output-dir", default="data")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    extract("outer", a.outer, out)
    extract("missing", a.missingness, out)
    extract("ablation", a.ablation, out)
    extract("pathway", a.pathway_control, out)
    print("Prepared:", out.resolve())

if __name__ == "__main__":
    main()
