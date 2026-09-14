# PathTokenSurv v1.5 Validation Report

Validation performed in the development environment:

- Unit/integration tests: **12 passed**.
- NCBI gene-info parser: passed on a synthetic human fixture.
- Reactome membership parser: passed.
- Reactome human relation parser: passed.
- KEGG pathway-list and gene-link parser: passed.
- miRTarBase CSV parser: passed.
- Combined KEGG + Reactome pathway JSON construction: passed.
- miRNA -> target gene -> KEGG/Reactome mapping: passed.
- Source-balanced pathway cap: passed.
- Existing synthetic end-to-end PathTokenSurv training/inference workflow: passed after v1.5 changes.

The development environment cannot make ordinary Python internet requests, so the online download stage itself was not executed end-to-end there. The downloader uses Python's standard library, retries, rate limiting for KEGG, source caching, and SHA-256 hashing. Database parser behavior was validated independently from network transport.
