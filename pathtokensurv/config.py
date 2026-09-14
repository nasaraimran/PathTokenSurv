from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


@dataclass
class DataConfig:
    """File names and preprocessing rules for a cohort."""

    data_dir: str = "data"
    # ``patient_wide`` is the original prepared format. ``tcga_pancancer_raw``
    # reads the native PanCancer matrices directly (features x samples) plus
    # the TCGA-CDR clinical workbook.
    source_format: str = "patient_wide"
    outcomes_file: str = "outcomes.tsv"
    clinical_file: str = "clinical.tsv"
    modality_files: Dict[str, str] = field(
        default_factory=lambda: {
            "mrna": "mrna.tsv",
            "mirna": "mirna.tsv",
            "cnv": "cnv.tsv",
        }
    )
    pathway_file: str = "pathways.json"
    delimiter: str = "\t"
    patient_id_col: str = "patient_id"
    time_col: str = "time"
    event_col: str = "event"
    cancer_col: str = "cancer_type"
    clinical_continuous: List[str] = field(default_factory=lambda: ["age"])
    clinical_categorical: List[str] = field(
        default_factory=lambda: ["sex", "race", "histology"]
    )
    variance_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"mrna": 7.0, "mirna": 0.0, "cnv": 0.2}
    )
    min_features_per_modality: int = 8
    min_features_per_pathway: int = 1
    max_features_per_pathway: int = 128
    # v1.5.4: modality-specific caps override the legacy scalar when provided.
    max_features_per_pathway_by_modality: Dict[str, int] = field(
        default_factory=lambda: {"mrna": 128, "mirna": 32, "cnv": 128}
    )
    # Conservative miRNA aliasing maps an arm-unspecified name to 3p/5p only
    # when exactly one mature arm exists in the complete annotation space.
    mirna_conservative_alias_harmonization: bool = True
    min_modalities_per_pathway: int = 1
    # Cap pathway tokens after fold-specific feature intersection to keep attention tractable.
    # Selection is unsupervised: no survival outcomes are used.
    max_pathways: int | None = 256
    balance_pathway_sources: bool = True
    # Fail before training if selected molecular features barely intersect the
    # annotation space; this catches identifier mismatches early.
    min_selected_annotation_coverage_pct: Dict[str, float] = field(
        default_factory=lambda: {"mrna": 20.0, "mirna": 10.0, "cnv": 20.0}
    )
    clip_scaled_values: bool = True
    preprocessing_feature_chunk_size: int = 2048

    # Raw TCGA PanCancer settings. These fields are ignored for patient_wide data.
    clinical_sheet: str = "TCGA-CDR"
    tcga_clinical_patient_id_col: str = "bcr_patient_barcode"
    tcga_clinical_cancer_col: str = "type"
    tcga_clinical_time_col: str = "OS.time"
    tcga_clinical_event_col: str = "OS"
    tcga_age_col: str = "age_at_initial_pathologic_diagnosis"
    tcga_sex_col: str = "gender"
    tcga_race_col: str = "race"
    tcga_pathologic_stage_col: str = "ajcc_pathologic_tumor_stage"
    tcga_clinical_stage_col: str = "clinical_stage"
    tcga_histology_col: str = "histological_type"
    tcga_patient_barcode_length: int = 12
    tcga_allowed_sample_types: List[str] = field(default_factory=lambda: ["01", "03", "09", "06"])
    tcga_duplicate_patient_strategy: str = "mean"
    tcga_duplicate_feature_strategy: str = "mean"
    tcga_exclude_zero_survival_time: bool = True
    tcga_raw_float_dtype: str = "float32"
    tcga_molecular_feature_columns: Dict[str, str] = field(
        default_factory=lambda: {"mrna": "sample", "mirna": "sample", "cnv": "Sample"}
    )


@dataclass
class ModelConfig:
    """Architecture settings for PathTokenSurv."""

    d_model: int = 128
    n_heads: int = 8
    intramodal_layers: int = 1
    fusion_layers: int = 2
    reconstruction_layers: int = 1
    ffn_multiplier: int = 4
    dropout: float = 0.15
    num_time_bins: int = 20
    pathway_same_bias_init: float = 0.5
    pathway_graph_bias_init: float = 0.25
    use_pathway_bias: bool = True
    use_reconstruction: bool = True
    use_subset_consistency: bool = True
    use_cancer_deviation: bool = True
    tokenizer_pathway_chunk_size: int = 32

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.num_time_bins < 2:
            raise ValueError("num_time_bins must be at least 2.")
        if self.tokenizer_pathway_chunk_size < 1:
            raise ValueError("tokenizer_pathway_chunk_size must be positive.")


@dataclass
class TrainingConfig:
    """Optimization, masking, and stopping settings."""

    seed: int = 123
    device: str = "auto"
    batch_size: int = 32
    num_workers: int = 0
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    lambda_reconstruction: float = 0.25
    lambda_consistency: float = 0.10
    lambda_baseline: float = 0.01
    baseline_smoothness: float = 0.25
    modality_drop_probability: float = 0.35
    always_keep_clinical: bool = False
    scheduler_t_max: int = 50
    min_learning_rate: float = 1e-6
    ensemble_size: int = 1


@dataclass
class EvaluationConfig:
    bootstrap_repetitions: int = 200
    calibration_bins: int = 10
    evaluation_horizons: List[float] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "outputs/experiment"

    def validate(self) -> None:
        self.model.validate()
        if not 0.0 <= self.training.modality_drop_probability < 1.0:
            raise ValueError("modality_drop_probability must lie in [0, 1).")
        if self.training.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.data.preprocessing_feature_chunk_size < 1:
            raise ValueError("preprocessing_feature_chunk_size must be positive.")
        if not 1 <= self.data.min_modalities_per_pathway <= 3:
            raise ValueError("min_modalities_per_pathway must lie between 1 and 3.")
        if self.data.min_features_per_pathway < 1:
            raise ValueError("min_features_per_pathway must be positive.")
        if self.data.max_features_per_pathway < 1:
            raise ValueError("max_features_per_pathway must be positive.")
        for modality, cap in self.data.max_features_per_pathway_by_modality.items():
            if modality not in {"mrna", "mirna", "cnv"}:
                raise ValueError(f"Unknown modality in max_features_per_pathway_by_modality: {modality}")
            if int(cap) < 1:
                raise ValueError("All modality-specific pathway caps must be positive.")
        if self.data.max_pathways is not None and self.data.max_pathways < 1:
            raise ValueError("max_pathways must be positive or null.")
        for modality, threshold in self.data.min_selected_annotation_coverage_pct.items():
            if modality not in {"mrna", "mirna", "cnv"}:
                raise ValueError(f"Unknown modality in min_selected_annotation_coverage_pct: {modality}")
            if not 0.0 <= float(threshold) <= 100.0:
                raise ValueError("Annotation coverage thresholds must lie in [0, 100].")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        cfg = cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            output_dir=raw.get("output_dir", "outputs/experiment"),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)
