import argparse
from pathlib import Path
import subprocess, sys

CORE = [
    "fig03_main_performance.py",
    "fig04_missingness.py",
    "fig05_ablation.py",
    "fig06_pathway_control.py",
    "supp01_fold_stability.py",
    "supp02_calibration_ece.py",
    "supp03_modality_patterns.py",
    "supp04_missingness_foldwise.py",
    "supp05_ablation_foldwise.py",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--km-dir", default=None)
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    for script in CORE:
        cmd = [
            sys.executable, str(here / script),
            "--data-dir", str(Path(a.data_dir).resolve()),
            "--output-dir", str(Path(a.output_dir).resolve()),
        ]
        print("RUN", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if a.km_dir:
        cmd = [
            sys.executable, str(here / "fig07_kaplan_meier.py"),
            "--km-dir", str(Path(a.km_dir).resolve()),
            "--output-dir", str(Path(a.output_dir).resolve()),
        ]
        print("RUN", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
