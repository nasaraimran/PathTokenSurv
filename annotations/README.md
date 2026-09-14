# Pathway Annotations

`raw/` and `processed/` contain locally acquired third-party resources and their
derived tables. They are excluded from version control because redistribution
terms and release sizes vary by provider.

Use `scripts/construct_pathways.py` and the versioned pathway-construction guides
to rebuild these files. `raw/source_manifest.json` records the source URLs,
retrieval dates, and checksums used for the study where available.
