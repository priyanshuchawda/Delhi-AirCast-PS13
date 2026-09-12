#!/usr/bin/env python3
"""Inventory downloaded source files without requiring pandas or a database."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_JSON = DATA / "SOURCE_INVENTORY.json"
OUT_MD = DATA / "SOURCE_INVENTORY.md"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_profile(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = 0
    header: list[str] = []
    first_row: list[str] = []
    with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {"header": [], "rows": 0}
        for row in reader:
            rows += 1
            if not first_row:
                first_row = row
    return {
        "header": header,
        "first_row": first_row,
        "rows": rows,
    }


def profile(path: Path) -> dict:
    relative = path.relative_to(ROOT)
    record = {
        "path": str(relative),
        "source": relative.parts[1] if len(relative.parts) > 1 else "unknown",
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }
    if path.suffix in {".csv", ".gz"} and path.name.endswith((".csv", ".csv.gz")):
        record["csv"] = csv_profile(path)
    elif path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record["json_keys"] = sorted(payload.keys()) if isinstance(payload, dict) else []
            if isinstance(payload, dict) and isinstance(payload.get("hourly"), dict):
                record["hourly_points"] = len(payload["hourly"].get("time", []))
        except (OSError, json.JSONDecodeError):
            record["json_error"] = True
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            record["zip_files"] = len(archive.infolist())
            record["zip_uncompressed_bytes"] = sum(item.file_size for item in archive.infolist())
    return record


def main() -> None:
    paths = sorted(
        path
        for path in DATA.rglob("*")
        if path.is_file()
        and ".cache" not in path.parts
        and not path.name.endswith((".part", ".lock", ".metadata"))
        and path.name not in {OUT_JSON.name, OUT_MD.name}
    )
    records = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.relative_to(ROOT)}", flush=True)
        records.append(profile(path))
    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    OUT_JSON.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Downloaded source inventory",
        "",
        f"Generated: `{inventory['generated_at']}`",
        f"Files: **{inventory['file_count']}**",
        f"Bytes: **{inventory['total_bytes']:,}**",
        "",
        "| Source | File | Size | SHA-256 | Details |",
        "|---|---|---:|---|---|",
    ]
    for record in records:
        details = ""
        if "csv" in record:
            details = f"{record['csv']['rows']:,} rows; {', '.join(record['csv']['header'][:6])}"
        elif "hourly_points" in record:
            details = f"{record['hourly_points']:,} hourly points"
        elif "zip_files" in record:
            details = f"{record['zip_files']:,} members; {record['zip_uncompressed_bytes']:,} uncompressed bytes"
        elif "json_keys" in record:
            details = ", ".join(record["json_keys"][:6])
        lines.append(
            f"| {record['source']} | `{record['path']}` | {record['bytes']:,} | `{record['sha256'][:16]}…` | {details} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
