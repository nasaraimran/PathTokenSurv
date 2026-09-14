from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

from pathtokensurv.config import ExperimentConfig
from pathtokensurv.data.raw import load_raw_cohort
from pathtokensurv.data.tcga import load_tcga_clinical_tables
from pathtokensurv.data.qc import patient_availability_table, modality_pattern_summary, cancer_specific_missingness


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect native TCGA PanCancer files and write QC metadata.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/tcga_raw_qc.json")
    parser.add_argument("--clinical-only", action="store_true", help="Inspect only the clinical workbook; do not load large omics matrices.")
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    if args.clinical_only:
        outcomes, clinical, qc = load_tcga_clinical_tables(cfg.data, require_outcomes=True)
        payload = {"clinical_qc": qc, "eligible_patient_count": len(outcomes)}
    else:
        raw = load_raw_cohort(cfg.data, require_outcomes=True)
        availability = patient_availability_table(raw, cfg.data)
        patterns = modality_pattern_summary(availability)
        cancer_missing = cancer_specific_missingness(availability)
        payload = dict(raw.metadata)
        payload["availability_pattern_summary"] = patterns.to_dict(orient="records")
        payload["cancer_specific_missingness"] = cancer_missing.to_dict(orient="records")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved QC report to {output}")


if __name__ == "__main__":
    main()
