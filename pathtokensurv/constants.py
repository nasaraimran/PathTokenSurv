from __future__ import annotations

MODALITIES = ("clinical", "mrna", "mirna", "cnv")
MOLECULAR_MODALITIES = ("mrna", "mirna", "cnv")
MODALITY_TO_INDEX = {name: idx for idx, name in enumerate(MODALITIES)}
