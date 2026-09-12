"""Fetch the Seoul Bike Sharing Demand dataset from the UCI repository."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import zipfile
from pathlib import Path

import requests

from ..config import ensure_dir, load_config, resolve

META_NAME = "dataset_meta.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_csv_path() -> Path:
    cfg = load_config()["data"]
    return resolve(cfg["raw_dir"]) / cfg["csv_name"]


def meta_path() -> Path:
    return resolve(load_config()["data"]["raw_dir"]) / META_NAME


def download_raw(force: bool = False, timeout: int = 120) -> Path:
    """Download and unpack the raw CSV. Idempotent unless `force` is set."""
    cfg = load_config()["data"]
    raw_dir = ensure_dir(cfg["raw_dir"])
    csv_path = raw_dir / cfg["csv_name"]

    if csv_path.exists() and not force:
        actual = sha256_of(csv_path)
        expected = cfg["expected_sha256"]
        if actual != expected:
            raise RuntimeError(
                f"Raw dataset SHA256 mismatch: expected {expected}, got {actual}. "
                "Remove the file or re-run download with --force."
            )
        print(f"[download] already present: {csv_path}")
        return csv_path

    print(f"[download] GET {cfg['url']}")
    response = requests.get(cfg["url"], timeout=timeout)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        if cfg["csv_name"] not in names:
            raise RuntimeError(f"{cfg['csv_name']} not found in archive; members are {names}")
        payload = archive.read(cfg["csv_name"])

    actual = hashlib.sha256(payload).hexdigest()
    expected = cfg["expected_sha256"]
    if actual != expected:
        raise RuntimeError(f"Downloaded dataset SHA256 mismatch: expected {expected}, got {actual}")

    csv_path.write_bytes(payload)

    n_lines = payload.count(b"\n")
    meta = {
        "url": cfg["url"],
        "file": cfg["csv_name"],
        "sha256": actual,
        "bytes": len(payload),
        "approx_rows": n_lines - 1,
        "downloaded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (raw_dir / META_NAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[download] saved {csv_path} ({len(payload)} bytes, sha256={meta['sha256'][:12]}…)")
    return csv_path


def data_sha256() -> str | None:
    """Hash of the raw dataset, recorded in every model artifact."""
    csv_path = raw_csv_path()
    if not csv_path.exists():
        return None
    actual = sha256_of(csv_path)
    expected = load_config()["data"]["expected_sha256"]
    if actual != expected:
        raise RuntimeError(f"Raw dataset SHA256 mismatch: expected {expected}, got {actual}")
    return actual
