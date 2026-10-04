"""Download, cache and parse public data (Ken French Data Library, FRED).

All returns are converted from percent to decimal. Every month is indexed by its
month-end Timestamp. Raw files are cached in data/raw/ together with a manifest
(URL, SHA-256, download time, CRSP vintage line) so a run can be audited later.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

_DATA_ROW = re.compile(r"^\s*(\d{6})\s*,")


def _fetch(url: str, path: Path) -> bytes:
    """Return file bytes, downloading once and recording it in the manifest."""
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    manifest_path = RAW / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[path.name] = {
        "url": url,
        "sha256": hashlib.sha256(resp.content).hexdigest(),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return resp.content


def _french_text(name: str) -> str:
    blob = _fetch(FRENCH_URL.format(name=name), RAW / f"{name}_CSV.zip")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return zf.read(zf.namelist()[0]).decode("latin-1")


def parse_french_csv(text: str, table: int = 0) -> pd.DataFrame:
    """Parse the `table`-th block of monthly (YYYYMM) rows in a French CSV.

    French files stack several tables (value-weighted, equal-weighted, annual...).
    Each block is a header line starting with ',' followed by data rows. Annual
    blocks use 4-digit years and are skipped by the 6-digit row pattern.
    """
    lines = text.splitlines()
    blocks, header, rows = [], None, []
    for line in lines:
        if _DATA_ROW.match(line):
            rows.append(line)
            continue
        if rows:
            blocks.append((header, rows))
            rows = []
        if line.lstrip().startswith(","):
            header = line
    if rows:
        blocks.append((header, rows))
    if table >= len(blocks):
        raise ValueError(f"only {len(blocks)} monthly tables found")
    header, rows = blocks[table]
    cols = [c.strip() for c in header.split(",")[1:]]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=["date", *cols])
    df.index = pd.to_datetime(df.pop("date").astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.astype(float)
    df = df.mask(df <= -99.99)  # French's missing-value codes: -99.99 and -999
    return df / 100.0


def french(name: str, table: int = 0) -> pd.DataFrame:
    return parse_french_csv(_french_text(name), table)


def french_vintage(name: str) -> str:
    """The 'created using the YYYYMM CRSP database' line (data vintage)."""
    first = _french_text(name).splitlines()[0]
    m = re.search(r"(\d{6}) CRSP", first)
    return m.group(1) if m else first.strip()


def fred(series: str) -> pd.Series:
    """Monthly FRED series indexed by month-end. Yields stay in percent units."""
    blob = _fetch(FRED_URL.format(series=series), RAW / f"FRED_{series}.csv")
    df = pd.read_csv(io.BytesIO(blob), na_values=".")
    s = df.set_index(pd.to_datetime(df.iloc[:, 0]) + pd.offsets.MonthEnd(0)).iloc[:, 1]
    return s.astype(float).rename(series)


# --- named datasets used in the project -------------------------------------

def ff3() -> pd.DataFrame:
    """Mkt-RF, SMB, HML, RF (US, July 1926 onward)."""
    return french("F-F_Research_Data_Factors")


def ff5() -> pd.DataFrame:
    """Mkt-RF, SMB, HML, RMW, CMA, RF (US, July 1963 onward)."""
    return french("F-F_Research_Data_5_Factors_2x3")


def umd() -> pd.Series:
    return french("F-F_Momentum_Factor")["Mom"].rename("UMD")


def six_portfolios() -> pd.DataFrame:
    """Value-weighted 2x3 size/BE-ME portfolios (building blocks of SMB, HML)."""
    return french("6_Portfolios_2x3")


def portfolios_25(region: str = "US") -> pd.DataFrame:
    """Value-weighted 5x5 size/BE-ME portfolios; columns renamed S1B1..S5B5."""
    name = "25_Portfolios_5x5" if region == "US" else f"{region}_25_Portfolios_ME_BE-ME"
    df = french(name)
    df.columns = [f"S{i}B{j}" for i in range(1, 6) for j in range(1, 6)]
    return df


def regional_factors(region: str) -> pd.DataFrame:
    """Mkt-RF, SMB, HML, RF for an international region (USD returns)."""
    return french(f"{region}_3_Factors")


def industries_49() -> pd.DataFrame:
    """Value-weighted returns on 49 industry portfolios."""
    return french("49_Industry_Portfolios")
