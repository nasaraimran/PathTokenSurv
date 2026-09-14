from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

REACTOME_BASE = "https://reactome.org/download/current"
NCBI_GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
KEGG_BASE = "https://rest.kegg.jp"

# miRNA-target providers. TarBase v9 is the primary provider in v1.5.2.
# The TarBase publication documents unrestricted local retrieval but does not
# publish a stable bulk-file URL that we can safely hard-code. Users therefore
# export/download the human interaction table from the official portal and pass
# it with --mirna-target-file / --tarbase-file, or provide the exact export URL
# with --tarbase-url. miRTarBase remains supported as an optional provider.
TARBASE_VERSION = "9.0"
TARBASE_PORTAL = "https://dianalab.e-ce.uth.gr/tarbasev9"
TARBASE_ALT_PORTAL = "http://62.217.122.56/"
MIRTARBASE_TEMPLATE = "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/cache/download/{release}/hsa_MTI.csv"
MIRTARBASE_RELEASE_PAGE = "https://awi.cuhk.edu.cn/miRTarBase/downloads/releases/{release}/"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, retries: int = 3, timeout: int = 180) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PathTokenSurv/1.5.2 academic research"})
            with urllib.request.urlopen(req, timeout=timeout) as response, dest.open("wb") as out:
                shutil.copyfileobj(response, out)
                headers = dict(response.headers.items())
            return {
                "url": url,
                "path": str(dest),
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "last_modified": headers.get("Last-Modified"),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_source_bundle(
    raw_dir: str | Path,
    *,
    reactome_all_levels: bool = True,
    include_kegg: bool = True,
    include_mirtarbase: bool = False,
    mirtarbase_release: str = "10.0",
    tarbase_url: str | None = None,
    tarbase_file: str | Path | None = None,
) -> dict:
    """Download and checksum the external annotation sources used by PathTokenSurv.

    TarBase v9 is the default miRNA-target provider in v1.5.2, but its publication
    does not specify a stable direct bulk-download URL. If ``tarbase_url`` is
    supplied (for example an export URL copied from the TarBase interface), that
    file is downloaded and checksummed. Otherwise only NCBI/Reactome/KEGG are
    downloaded automatically. miRTarBase download support is retained for
    backward compatibility and optional analyses.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    reactome_name = "NCBI2Reactome_All_Levels.txt" if reactome_all_levels else "NCBI2Reactome.txt"
    for key, url, name in [
        ("ncbi_gene_info", NCBI_GENE_INFO_URL, "Homo_sapiens.gene_info.gz"),
        ("reactome_membership", f"{REACTOME_BASE}/{reactome_name}", reactome_name),
        ("reactome_pathways", f"{REACTOME_BASE}/ReactomePathways.txt", "ReactomePathways.txt"),
        ("reactome_relations", f"{REACTOME_BASE}/ReactomePathwaysRelation.txt", "ReactomePathwaysRelation.txt"),
    ]:
        manifest[key] = _download(url, raw_dir / name)

    if include_kegg:
        for key, endpoint, name in [
            ("kegg_info", "info/kegg", "kegg_info.txt"),
            ("kegg_pathways", "list/pathway/hsa", "kegg_hsa_pathways.txt"),
            ("kegg_gene_links", "link/pathway/hsa", "kegg_hsa_gene_pathway_links.txt"),
        ]:
            manifest[key] = _download(f"{KEGG_BASE}/{endpoint}", raw_dir / name)
            time.sleep(0.4)

    local_tarbase = Path(tarbase_file) if tarbase_file is not None else None
    if local_tarbase is not None and local_tarbase.exists() and local_tarbase.stat().st_size > 0:
        manifest["tarbase"] = {
            "status": "local_file_registered",
            "provider": "DIANA-TarBase",
            "version": TARBASE_VERSION,
            "portal": TARBASE_PORTAL,
            "path": str(local_tarbase.resolve()),
            "bytes": local_tarbase.stat().st_size,
            "sha256": sha256_file(local_tarbase),
            "registered_utc": datetime.now(timezone.utc).isoformat(),
        }
    elif tarbase_url:
        try:
            manifest["tarbase"] = _download(tarbase_url, raw_dir / "TarBase_v9_human_interactions.csv")
            manifest["tarbase"]["provider"] = "DIANA-TarBase"
            manifest["tarbase"]["version"] = TARBASE_VERSION
        except RuntimeError as exc:
            manifest["tarbase"] = {
                "url": tarbase_url,
                "status": "download_failed",
                "message": str(exc),
                "portal": TARBASE_PORTAL,
            }
    else:
        manifest["tarbase"] = {
            "status": "manual_export_required",
            "provider": "DIANA-TarBase",
            "version": TARBASE_VERSION,
            "portal": TARBASE_PORTAL,
            "alternate_portal": TARBASE_ALT_PORTAL,
            "manual_action": (
                "Download/export Homo sapiens miRNA-gene interactions from the official TarBase v9 portal, "
                "save the CSV/TSV/XLSX file under annotations/raw, and pass it with --tarbase-file."
            ),
        }

    if include_mirtarbase:
        url = MIRTARBASE_TEMPLATE.format(release=mirtarbase_release)
        try:
            manifest["mirtarbase"] = _download(url, raw_dir / f"miRTarBase_{mirtarbase_release}_hsa_MTI.csv")
        except RuntimeError as exc:  # server can occasionally be unavailable
            manifest["mirtarbase"] = {
                "url": url,
                "status": "download_failed",
                "message": str(exc),
                "release_page": MIRTARBASE_RELEASE_PAGE.format(release=mirtarbase_release),
                "manual_action": "Download the Homo sapiens hsa_MTI.csv and rerun with --mirtarbase-file.",
            }

    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    (raw_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_ncbi_gene_map(path: str | Path) -> pd.DataFrame:
    """Return unique human Entrez ID -> official/current gene symbol mappings."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        df = pd.read_csv(handle, sep="\t", dtype=str, low_memory=False)
    if "#tax_id" in df.columns:
        df = df[df["#tax_id"].astype(str).eq("9606")]
    if "GeneID" not in df.columns or "Symbol" not in df.columns:
        raise ValueError("NCBI gene_info must contain GeneID and Symbol columns.")
    out = df[["GeneID", "Symbol"]].rename(columns={"GeneID": "entrez_id", "Symbol": "gene_symbol"}).copy()
    out["entrez_id"] = out["entrez_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["gene_symbol"] = out["gene_symbol"].astype(str).str.strip()
    out = out[(out.entrez_id != "") & (out.gene_symbol != "-") & (out.gene_symbol != "")]
    return out.drop_duplicates("entrez_id", keep="first").reset_index(drop=True)


def parse_reactome_membership(path: str | Path, gene_map: pd.DataFrame) -> pd.DataFrame:
    names = ["entrez_id", "reactome_id", "url", "pathway_name", "evidence", "species"]
    df = pd.read_csv(path, sep="\t", header=None, names=names, dtype=str, usecols=range(6), low_memory=False).fillna("")
    df = df[df["species"].eq("Homo sapiens") & df["reactome_id"].str.startswith("R-HSA-")].copy()
    df["entrez_id"] = df["entrez_id"].str.strip().str.replace(r"\.0$", "", regex=True)
    df = df.merge(gene_map, on="entrez_id", how="left")
    df["gene_symbol"] = df["gene_symbol"].fillna("")
    df["pathway_id"] = "REACTOME:" + df["reactome_id"].str.strip()
    df["source"] = "Reactome"
    out = df[["pathway_id", "pathway_name", "gene_symbol", "entrez_id", "source"]]
    return out.drop_duplicates().reset_index(drop=True)


def parse_reactome_relations(path: str | Path, valid_pathways: Iterable[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["parent", "child"], dtype=str).fillna("")
    df = df[df.parent.str.startswith("R-HSA-") & df.child.str.startswith("R-HSA-")].copy()
    df["source_pathway"] = "REACTOME:" + df.parent.str.strip()
    df["target_pathway"] = "REACTOME:" + df.child.str.strip()
    if valid_pathways is not None:
        valid = set(map(str, valid_pathways))
        df = df[df.source_pathway.isin(valid) & df.target_pathway.isin(valid)]
    df["relation"] = "parent_child"
    df["source"] = "Reactome"
    return df[["source_pathway", "target_pathway", "relation", "source"]].drop_duplicates().reset_index(drop=True)


def _strip_kegg_prefix(text: str) -> str:
    text = str(text).strip()
    return text.split(":", 1)[1] if ":" in text else text


def parse_kegg_membership(pathway_list_file: str | Path, link_file: str | Path, gene_map: pd.DataFrame) -> pd.DataFrame:
    names = {}
    with Path(pathway_list_file).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            pid, name = line.rstrip("\n").split("\t", 1)
            pid = _strip_kegg_prefix(pid)
            name = re.sub(r"\s+-\s+Homo sapiens \(human\)\s*$", "", name).strip()
            names[pid] = name

    pairs = []
    with Path(link_file).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            left, right = line.rstrip("\n").split("\t", 1)
            left, right = left.strip(), right.strip()
            if left.startswith("hsa:") and left.split(":", 1)[1].isdigit():
                gene_id = left.split(":", 1)[1]
                pid = _strip_kegg_prefix(right)
            elif right.startswith("hsa:") and right.split(":", 1)[1].isdigit():
                gene_id = right.split(":", 1)[1]
                pid = _strip_kegg_prefix(left)
            else:
                continue
            if pid.startswith("hsa") and pid[3:].isdigit():
                pairs.append((pid, gene_id))
    links = pd.DataFrame(pairs, columns=["kegg_id", "entrez_id"]).drop_duplicates()
    links = links.merge(gene_map, on="entrez_id", how="left")
    links["gene_symbol"] = links["gene_symbol"].fillna("")
    links["pathway_id"] = "KEGG:" + links["kegg_id"]
    links["pathway_name"] = links["kegg_id"].map(names).fillna(links["kegg_id"])
    links["source"] = "KEGG"
    return links[["pathway_id", "pathway_name", "gene_symbol", "entrez_id", "source"]].drop_duplicates().reset_index(drop=True)


def _normalize_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _find_col(columns: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    for normalized, original in columns.items():
        if any(c in normalized for c in candidates):
            return original
    return None


def _detect_delimiter(path: Path) -> str:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return ","
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(16384)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def _read_header(path: Path, sep: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=5, dtype=str).fillna("")
    sep = sep or _detect_delimiter(path)
    return pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig", low_memory=False, nrows=5, compression="infer").fillna("")


def _is_human_species(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip().str.lower()
    return (
        s.eq("9606")
        | s.eq("human")
        | s.eq("hsa")
        | s.str.contains("homo sapiens", regex=False)
        | s.str.contains("h. sapiens", regex=False)
    )


def _clean_entrez_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(s.str.fullmatch(r"\d+", na=False), "")


def _standardize_target_chunk(
    df: pd.DataFrame,
    *,
    provider: str,
    mirna_col: str,
    gene_col: str | None,
    entrez_col: str | None,
    species_col: str | None,
    support_col: str | None,
    experiments_col: str | None,
    gene_map: pd.DataFrame | None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["mirna"] = df[mirna_col].astype(str).str.strip()
    out["gene_symbol"] = df[gene_col].astype(str).str.strip() if gene_col else ""
    out["entrez_id"] = _clean_entrez_series(df[entrez_col]) if entrez_col else ""
    out["support_type"] = df[support_col].astype(str).str.strip() if support_col else ""
    out["experiments"] = df[experiments_col].astype(str).str.strip() if experiments_col else ""

    if species_col is not None:
        out = out.loc[_is_human_species(df.loc[out.index, species_col])].copy()
    # PathTokenSurv uses human TCGA miRNAs. This also removes viral miRNAs from
    # TarBase v9 even when the source export contains them.
    out = out[out.mirna.str.lower().str.startswith("hsa-")].copy()

    if gene_map is not None and len(out):
        gm = gene_map[["entrez_id", "gene_symbol"]].drop_duplicates().copy()
        e2s = dict(zip(gm.entrez_id.astype(str), gm.gene_symbol.astype(str)))
        # Symbol -> Entrez can be non-unique for aliases; gene_info Symbol is official,
        # so use the first official mapping deterministically.
        s2e = dict(zip(gm.gene_symbol.astype(str).str.upper(), gm.entrez_id.astype(str)))
        missing_symbol = out.gene_symbol.eq("") | out.gene_symbol.eq("-")
        out.loc[missing_symbol, "gene_symbol"] = out.loc[missing_symbol, "entrez_id"].map(e2s).fillna("")
        missing_entrez = out.entrez_id.eq("")
        out.loc[missing_entrez, "entrez_id"] = out.loc[missing_entrez, "gene_symbol"].str.upper().map(s2e).fillna("")

    out = out[(out.mirna != "") & (out.gene_symbol != "")].copy()
    out["source"] = provider
    return out[["mirna", "gene_symbol", "entrez_id", "support_type", "experiments", "source"]]


def parse_tarbase_targets(
    path: str | Path,
    gene_map: pd.DataFrame | None = None,
    *,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Parse a TarBase v9 human export into unique miRNA -> target-gene pairs.

    TarBase v9 exports can be large (millions of rows) and the web interface may
    evolve. The parser therefore auto-detects common TarBase column labels and
    reads delimited files in chunks. CSV, TSV/TXT, gzipped delimited files and
    XLSX exports are accepted. If a species column is present, only Homo sapiens
    rows are kept; in all cases only ``hsa-`` miRNAs are retained.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"TarBase file not found or empty: {path}")

    header = _read_header(path)
    cmap = {_normalize_column_name(c): c for c in header.columns}
    mirna_col = _find_col(cmap, ["mirna", "mirna_name", "mirna_id", "mature_mirna", "mirbase_id"])
    gene_col = _find_col(cmap, ["gene_symbol", "target_gene_symbol", "target_gene", "gene_name"])
    # Avoid treating identifier fields such as ``NCBI Gene ID`` as a gene symbol.
    if gene_col is None:
        gene_col = cmap.get("gene")
    entrez_col = _find_col(cmap, ["entrez_id", "entrez_gene_id", "ncbi_gene_id", "target_gene_entrez_id", "target_entrez"])
    species_col = _find_col(cmap, ["species", "species_name", "organism", "organism_name", "tax_id", "taxonomy_id"])
    support_col = _find_col(cmap, ["experimental_type", "experiment_type", "support_type", "direct_indirect", "interaction_type", "support"])
    experiments_col = _find_col(cmap, ["experimental_method", "experiment_method", "method", "technique", "experiments", "experiment"])
    if mirna_col is None or (gene_col is None and entrez_col is None):
        raise ValueError(
            "Could not identify TarBase miRNA/target columns. "
            f"Columns seen in {path.name}: {list(header.columns)}"
        )

    usecols = [x for x in [mirna_col, gene_col, entrez_col, species_col, support_col, experiments_col] if x is not None]
    usecols = list(dict.fromkeys(usecols))
    parts: list[pd.DataFrame] = []

    if path.suffix.lower() in {".xlsx", ".xls"}:
        iterator = [pd.read_excel(path, dtype=str, usecols=usecols).fillna("")]
    else:
        sep = _detect_delimiter(path)
        iterator = pd.read_csv(
            path,
            sep=sep,
            dtype=str,
            encoding="utf-8-sig",
            low_memory=False,
            compression="infer",
            usecols=usecols,
            chunksize=max(1, int(chunksize)),
        )

    for chunk in iterator:
        chunk = chunk.fillna("")
        standardized = _standardize_target_chunk(
            chunk,
            provider="TarBase-v9.0",
            mirna_col=mirna_col,
            gene_col=gene_col,
            entrez_col=entrez_col,
            species_col=species_col,
            support_col=support_col,
            experiments_col=experiments_col,
            gene_map=gene_map,
        )
        if len(standardized):
            parts.append(standardized.drop_duplicates(["mirna", "gene_symbol"]))

    if not parts:
        return pd.DataFrame(columns=["mirna", "gene_symbol", "entrez_id", "support_type", "experiments", "source"])
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["mirna", "gene_symbol"], keep="first").reset_index(drop=True)


def parse_mirtarbase_targets(path: str | Path, gene_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """Parse human miRTarBase CSV/TSV exports into miRNA -> target gene mappings."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"miRTarBase file not found or empty: {path}")
    header = _read_header(path)
    cmap = {_normalize_column_name(c): c for c in header.columns}
    mirna_col = _find_col(cmap, ["mirna", "mirna_name"])
    gene_col = _find_col(cmap, ["target_gene", "target_gene_symbol", "gene_symbol"])
    entrez_col = _find_col(cmap, ["target_gene_entrez_gene_id", "target_gene_entrez_id", "entrez_gene_id", "entrez_id", "target_entrez"])
    support_col = _find_col(cmap, ["support_type", "support"])
    experiments_col = _find_col(cmap, ["experiments", "experiment"])
    if mirna_col is None or (gene_col is None and entrez_col is None):
        raise ValueError(f"Could not identify miRNA/target columns in {path}. Columns: {list(header.columns)}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str).fillna("")
    else:
        df = pd.read_csv(path, sep=_detect_delimiter(path), dtype=str, encoding="utf-8-sig", low_memory=False, compression="infer").fillna("")
    out = _standardize_target_chunk(
        df,
        provider="miRTarBase",
        mirna_col=mirna_col,
        gene_col=gene_col,
        entrez_col=entrez_col,
        species_col=None,
        support_col=support_col,
        experiments_col=experiments_col,
        gene_map=gene_map,
    )
    return out.drop_duplicates(["mirna", "gene_symbol"], keep="first").reset_index(drop=True)


def _auto_find_mirna_target_file(raw_dir: Path, provider: str) -> Path | None:
    provider = provider.lower()
    if provider == "tarbase":
        patterns = [
            "TarBase*v9*.csv", "TarBase*v9*.tsv", "TarBase*v9*.txt", "TarBase*v9*.xlsx",
            "tarbase*v9*.csv", "tarbase*v9*.tsv", "tarbase*v9*.txt", "tarbase*v9*.xlsx",
            "*tarbase*.csv", "*tarbase*.tsv", "*tarbase*.txt", "*tarbase*.xlsx",
            "TarBase*.csv.gz", "tarbase*.csv.gz",
        ]
    elif provider == "mirtarbase":
        patterns = ["miRTarBase_*_hsa_MTI.csv", "hsa_MTI.csv", "*hsa*MTI*.csv", "miRTarBase_MTI.csv", "*mirtarbase*.tsv"]
    else:
        return None
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(raw_dir.glob(pattern))
    possible = sorted({p.resolve() for p in candidates if p.is_file() and p.stat().st_size > 0})
    if provider == "tarbase":
        # Do not accidentally treat an optional miRTarBase file as a TarBase export.
        possible = [p for p in possible if "mirtarbase" not in p.name.lower()]
    return possible[-1] if possible else None


def prepare_unified_annotations(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    mirna_target_source: str = "tarbase",
    mirna_target_file: str | Path | None = None,
    tarbase_file: str | Path | None = None,
    mirtarbase_file: str | Path | None = None,
    mirna_target_chunksize: int = 250_000,
    include_kegg: bool = True,
    include_reactome: bool = True,
) -> dict:
    """Convert cached database downloads into PathTokenSurv annotation tables.

    ``mirna_target_source`` accepts ``tarbase`` (default), ``mirtarbase`` or
    ``none``. Provider-specific file arguments are kept for convenient CLI and
    backward compatibility. The standardized output schema is identical for both
    providers, so downstream pathway construction is unchanged.
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_info = raw_dir / "Homo_sapiens.gene_info.gz"
    if not gene_info.exists():
        raise FileNotFoundError(f"Missing {gene_info}. Run with --download first or provide the source bundle.")
    gene_map = load_ncbi_gene_map(gene_info)

    membership_parts = []
    edge_parts = []
    if include_reactome:
        candidates = [raw_dir / "NCBI2Reactome_All_Levels.txt", raw_dir / "NCBI2Reactome.txt"]
        reactome_file = next((p for p in candidates if p.exists()), None)
        if reactome_file is None:
            raise FileNotFoundError("Reactome NCBI mapping file not found.")
        react = parse_reactome_membership(reactome_file, gene_map)
        membership_parts.append(react)
        relation_file = raw_dir / "ReactomePathwaysRelation.txt"
        if relation_file.exists():
            edge_parts.append(parse_reactome_relations(relation_file, react.pathway_id.unique()))

    if include_kegg:
        kegg_list = raw_dir / "kegg_hsa_pathways.txt"
        kegg_links = raw_dir / "kegg_hsa_gene_pathway_links.txt"
        if not kegg_list.exists() or not kegg_links.exists():
            raise FileNotFoundError("KEGG cache files not found. Run with --download first.")
        membership_parts.append(parse_kegg_membership(kegg_list, kegg_links, gene_map))

    if not membership_parts:
        raise ValueError("At least one pathway database must be enabled.")
    membership = pd.concat(membership_parts, ignore_index=True).drop_duplicates()
    membership = membership.sort_values(["source", "pathway_id", "entrez_id", "gene_symbol"]).reset_index(drop=True)
    membership_path = output_dir / "gene_pathway_membership.tsv"
    membership.to_csv(membership_path, sep="\t", index=False)

    edges = pd.concat(edge_parts, ignore_index=True).drop_duplicates() if edge_parts else pd.DataFrame(
        columns=["source_pathway", "target_pathway", "relation", "source"]
    )
    edges_path = output_dir / "pathway_edges.tsv"
    edges.to_csv(edges_path, sep="\t", index=False)

    source = str(mirna_target_source).strip().lower()
    if source not in {"tarbase", "mirtarbase", "none"}:
        raise ValueError("mirna_target_source must be one of: tarbase, mirtarbase, none")

    chosen_file: Path | None = None
    if mirna_target_file is not None:
        chosen_file = Path(mirna_target_file)
    elif source == "tarbase" and tarbase_file is not None:
        chosen_file = Path(tarbase_file)
    elif source == "mirtarbase" and mirtarbase_file is not None:
        chosen_file = Path(mirtarbase_file)
    elif source != "none":
        chosen_file = _auto_find_mirna_target_file(raw_dir, source)

    if source == "tarbase" and chosen_file is not None and chosen_file.exists():
        targets = parse_tarbase_targets(chosen_file, gene_map, chunksize=mirna_target_chunksize)
    elif source == "mirtarbase" and chosen_file is not None and chosen_file.exists():
        targets = parse_mirtarbase_targets(chosen_file, gene_map)
    else:
        targets = pd.DataFrame(columns=["mirna", "gene_symbol", "entrez_id", "support_type", "experiments", "source"])

    targets_path = output_dir / "mirna_targets.tsv"
    targets.to_csv(targets_path, sep="\t", index=False)

    source_counts = membership.groupby("source").agg(pathways=("pathway_id", "nunique"), rows=("pathway_id", "size"), genes=("entrez_id", "nunique")).reset_index()
    qc = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gene_map_rows": int(len(gene_map)),
        "membership_rows": int(len(membership)),
        "unique_pathways": int(membership.pathway_id.nunique()),
        "unique_entrez_ids": int(membership.entrez_id.nunique()),
        "unique_gene_symbols": int(membership.loc[membership.gene_symbol != "", "gene_symbol"].str.upper().nunique()),
        "pathway_edges": int(len(edges)),
        "mirna_target_provider": source,
        "mirna_target_provider_label": (
            "DIANA-TarBase v9.0" if source == "tarbase" else "miRTarBase" if source == "mirtarbase" else "None"
        ),
        "mirna_target_source_file": str(chosen_file) if chosen_file is not None else None,
        "mirna_target_source_sha256": sha256_file(chosen_file) if chosen_file is not None and chosen_file.exists() else None,
        "mirna_target_pairs": int(len(targets)),
        "unique_mirnas": int(targets.mirna.str.lower().nunique()) if len(targets) else 0,
        "unique_mirna_target_genes": int(targets.gene_symbol.str.upper().nunique()) if len(targets) else 0,
        "source_counts": source_counts.to_dict(orient="records"),
        "edge_policy": "Reactome human parent-child relations; KEGG contributes pathway membership but no inferred KEGG pathway-pathway edges.",
    }
    (output_dir / "annotation_source_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return {"membership": membership_path, "edges": edges_path, "mirna_targets": targets_path, "qc": qc}
