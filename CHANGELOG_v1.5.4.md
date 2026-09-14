# PathTokenSurv v1.5.4 Changelog

## Purpose

v1.5.4 finalizes the fold-specific pathway-token selection logic after the v1.5.3 real-data QC showed that mRNA identifier harmonization was repaired but many retained pathways still reached the common 128-feature miRNA cap.

## Changes

- Added **score-aware top-K miRNA assignment**. The pathway builder now stores aligned miRNA enrichment scores, BH-adjusted q-values, and target overlaps. Fold-specific pathway construction selects the strongest miRNAs by enrichment score rather than arbitrary list position.
- Added **modality-specific pathway caps** through `max_features_per_pathway_by_modality`.
  - TCGA primary configuration: mRNA = 128, miRNA = 32, CNV = 128.
  - The legacy scalar `max_features_per_pathway` remains supported as a fallback.
- Added **normalized pathway ranking**. Candidate pathways are ranked by equal-weight mean coverage across mRNA, miRNA, and CNV after modality-specific top-K selection, rather than by raw total matched features.
- Retained optional **KEGG/Reactome source balancing** after normalized ranking.
- Added **conservative miRNA alias harmonization**. An arm-unspecified dataset name is mapped to `-3p` or `-5p` only when the complete provider miRNA set contains exactly one mature arm for that base. Ambiguous two-arm cases are left unmatched.
- Added provider-wide conservative aliases to pathway metadata so alias decisions are based on the complete TarBase/miRTarBase input, not only links surviving pathway enrichment.
- Extended `PathwaySpec` with outcome-independent QC metadata:
  - pre-cap eligible feature counts by pathway/modality;
  - normalized pathway-selection scores.
- Extended `retained_pathway_sizes.csv` with:
  - `*_eligible_before_cap`;
  - per-modality feature caps;
  - `*_at_feature_cap`;
  - `*_truncated_by_cap`;
  - `normalized_multimodal_coverage_score`.
- Extended miRNA annotation validation with the number of conservative aliases available and selected features matched specifically through those aliases.
- Kept all pathway selection, top-K assignment, alias harmonization, and ranking **outcome-independent**.

## Backward compatibility

v1.5.4 can load v1.5.3 pathway JSON files. When explicit miRNA enrichment scores are absent, the existing miRNA annotation order is preserved. For the main TCGA analysis, rebuild `pathways.json` with v1.5.4 so explicit enrichment scores and provider-wide alias metadata are stored.
