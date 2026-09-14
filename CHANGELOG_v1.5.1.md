# PathTokenSurv v1.5.1

Hotfix for miRTarBase pathway construction.

- Reports the original automatic-download failure immediately instead of later emitting only a generic zero-pair error.
- Adds the official miRTarBase release page to the error message.
- Auto-detects common local human miRTarBase filenames, including `hsa_MTI.csv`.
- Recognizes the common `Target Gene (Entrez ID)` header.
- Adds `scripts/validate_mirtarbase.py` for preflight validation of a manually downloaded file.
- Keeps `--allow-empty-mirna` restricted to deliberate ablation runs.
