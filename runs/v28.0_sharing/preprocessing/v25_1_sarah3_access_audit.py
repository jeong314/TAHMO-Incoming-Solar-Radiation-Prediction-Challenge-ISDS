from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pandas as pd
import requests


RUN_DIR = Path("runs/v25.1_sarah3_audit")
DIAG_DIR = RUN_DIR / "diagnostics"
DATA_DIR = RUN_DIR / "data"
LOG_PATH = RUN_DIR / "train.log"

TRAIN_PATH = Path("Train.csv")
TEST_PATH = Path("Test.csv")

SARAH3_COLLECTION_ID = "EO:EUM:DAT:0863"
EUMETSAT_BROWSE_URL = f"https://api.eumetsat.int/data/browse/collections/{quote(SARAH3_COLLECTION_ID, safe='')}"
EUMETSAT_DATES_URL = f"https://api.eumetsat.int/data/browse/1.0.0/collections/{quote(SARAH3_COLLECTION_ID, safe='')}/dates?format=json"
EUMETSAT_OSDD_URL = f"https://api.eumetsat.int/data/search-products/1.0.0/osdd?pi={quote(SARAH3_COLLECTION_ID, safe='')}"
CMSAF_WUI_URL = "https://wui.cmsaf.eu/"


def log(message: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message, flush=True)


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_tahmo_catalog(smoke_test: bool = False) -> pd.DataFrame:
    usecols = ["timestamp", "station", "station_name", "country", "latitude", "longitude", "elevation"]
    train = pd.read_csv(TRAIN_PATH, usecols=usecols)
    test = pd.read_csv(TEST_PATH, usecols=usecols)
    df = pd.concat([train, test], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cat = (
        df.groupby("station", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            country=("country", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            elevation=("elevation", "first"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            row_count=("timestamp", "size"),
        )
        .sort_values("station")
    )
    if smoke_test:
        cat = cat.head(6).copy()
    cat.to_csv(DIAG_DIR / "tahmo_station_catalog.csv", index=False)
    return cat


def fetch_json(url: str, timeout: int = 60) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def fetch_text(url: str, timeout: int = 60) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def build_station_request_plan(tahmo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in tahmo.itertuples(index=False):
        start = pd.Timestamp(row.start_time).floor("D")
        end = pd.Timestamp(row.end_time).ceil("D")
        rows.append(
            {
                "station": row.station,
                "station_name": row.station_name,
                "country": row.country,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "bbox_deg_0p10": f"{row.longitude - 0.10:.4f},{row.latitude - 0.10:.4f},{row.longitude + 0.10:.4f},{row.latitude + 0.10:.4f}",
                "bbox_deg_0p25": f"{row.longitude - 0.25:.4f},{row.latitude - 0.25:.4f},{row.longitude + 0.25:.4f},{row.latitude + 0.25:.4f}",
                "recommended_products": "SIS,SID,SDU,CAL",
                "expected_role": "15/30-min satellite solar reference or daily/monthly cloud albedo context",
            }
        )
    plan = pd.DataFrame(rows)
    plan.to_csv(DIAG_DIR / "sarah3_station_request_plan.csv", index=False)
    return plan


def inspect_netcdf(path: Path) -> dict[str, Any]:
    try:
        import h5py

        with h5py.File(path, "r") as h5:
            return {"reader": "h5py", "keys": list(h5.keys())[:50]}
    except Exception as h5_exc:
        try:
            from scipy.io import netcdf_file

            with netcdf_file(str(path), "r", mmap=False) as nc:
                return {
                    "reader": "scipy.io.netcdf_file",
                    "dimensions": {k: int(v) for k, v in nc.dimensions.items()},
                    "variables": list(nc.variables.keys())[:80],
                }
        except Exception as scipy_exc:
            return {"reader": None, "h5py_error": str(h5_exc), "scipy_error": str(scipy_exc)}


def download_sample_entry(detail: dict[str, Any], product_id: str, timeout: int = 60) -> dict[str, Any]:
    entries = detail.get("properties", {}).get("links", {}).get("sip-entries", [])
    nc_links = [link for link in entries if str(link.get("title", "")).endswith(".nc")]
    if not nc_links:
        return {"attempted": False, "reason": "no NetCDF entry link"}
    link = nc_links[0]
    href = link["href"]
    dest = DATA_DIR / f"{product_id}.nc"
    report: dict[str, Any] = {"attempted": True, "url": href, "path": str(dest)}
    try:
        with requests.get(href, timeout=timeout, stream=True) as response:
            report.update(
                {
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type"),
                    "content_length": response.headers.get("content-length"),
                }
            )
            response.raise_for_status()
            tmp = dest.with_suffix(".nc.tmp")
            with tmp.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
        report["downloaded_size"] = dest.stat().st_size
        report["netcdf_inspection"] = inspect_netcdf(dest)
    except Exception as exc:
        report["error"] = str(exc)
    return report


def sample_product_search(plan: pd.DataFrame, timeout: int = 60, download_sample: bool = False) -> dict[str, Any]:
    if plan.empty:
        return {"reachable": False, "error": "empty station plan"}
    row = plan.iloc[0]
    start = pd.Timestamp(row["start_date"])
    end = start + pd.Timedelta(days=1)
    params = {
        "pi": SARAH3_COLLECTION_ID,
        "c": "5",
        "bbox": row["bbox_deg_0p25"],
        "dtstart": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dtend": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "type": "SIS",
        "compositeType": "PT30M",
        "set": "brief",
        "format": "json",
    }
    url = "https://api.eumetsat.int/data/search-products/1.0.0/os?" + urlencode(params)
    try:
        result = fetch_json(url, timeout=timeout)
        (RUN_DIR / "sarah3_sample_search.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        count = result.get("totalResults") or result.get("totalResults".lower()) or result.get("properties", {}).get("totalResults")
        download_probe: dict[str, Any] = {"attempted": False}
        features = result.get("features") or []
        if features:
            links = features[0].get("properties", {}).get("links", {}).get("data", [])
            if links:
                href = links[0].get("href")
                if href:
                    product_id = features[0].get("id")
                    product_detail_candidates = [
                        "https://api.eumetsat.int/data/browse/collections/"
                        + quote(SARAH3_COLLECTION_ID, safe="")
                        + "/products/"
                        + quote(str(product_id), safe="")
                        + "?format=json",
                        "https://api.eumetsat.int/data/browse/1.0.0/collections/"
                        + quote(SARAH3_COLLECTION_ID, safe="")
                        + "/products/"
                        + quote(str(product_id), safe="")
                        + "?format=json",
                    ]
                    fixed_href = (
                        "https://api.eumetsat.int/data/download/1.0.0/collections/"
                        + quote(SARAH3_COLLECTION_ID, safe="")
                        + "/products/"
                        + quote(str(product_id), safe="")
                    )
                    detail_probe: list[dict[str, Any]] = []
                    first_detail: dict[str, Any] | None = None
                    for detail_url in product_detail_candidates:
                        detail_report: dict[str, Any] = {"url": detail_url}
                        try:
                            detail = fetch_json(detail_url, timeout=timeout)
                            if first_detail is None:
                                first_detail = detail
                            detail_report.update(
                                {
                                    "reachable": True,
                                    "top_level_keys": sorted(list(detail.keys())),
                                }
                            )
                            (RUN_DIR / "sarah3_sample_product_detail.json").write_text(
                                json.dumps(detail, indent=2), encoding="utf-8"
                            )
                        except Exception as exc:
                            detail_report.update({"reachable": False, "error": str(exc)})
                        detail_probe.append(detail_report)
                    sample_download = (
                        download_sample_entry(first_detail, str(product_id), timeout=timeout)
                        if download_sample and first_detail is not None
                        else {"attempted": False}
                    )
                    download_probe = {
                        "attempted": True,
                        "product_id": product_id,
                        "detail_candidates": detail_probe,
                        "sample_download": sample_download,
                        "candidates": [],
                    }
                    for candidate_url in [href, fixed_href]:
                        candidate_report: dict[str, Any] = {"url": candidate_url}
                        try:
                            head = requests.head(candidate_url, timeout=timeout, allow_redirects=True)
                            candidate_report.update(
                                {
                                    "status_code": head.status_code,
                                    "content_type": head.headers.get("content-type"),
                                    "content_length": head.headers.get("content-length"),
                                    "auth_header_present": "www-authenticate" in {k.lower(): v for k, v in head.headers.items()},
                                }
                            )
                        except Exception as exc:
                            candidate_report.update({"head_error": str(exc)})
                        try:
                            get_resp = requests.get(
                                candidate_url,
                                timeout=timeout,
                                allow_redirects=True,
                                stream=True,
                                headers={"Range": "bytes=0-0"},
                            )
                            candidate_report.update(
                                {
                                    "range_get_status_code": get_resp.status_code,
                                    "range_get_content_type": get_resp.headers.get("content-type"),
                                    "range_get_content_length": get_resp.headers.get("content-length"),
                                    "range_get_auth_header_present": "www-authenticate"
                                    in {k.lower(): v for k, v in get_resp.headers.items()},
                                }
                            )
                            get_resp.close()
                        except Exception as exc:
                            candidate_report.update({"range_get_error": str(exc)})
                        download_probe["candidates"].append(candidate_report)
        return {
            "reachable": True,
            "url": url,
            "station": row["station"],
            "start": params["dtstart"],
            "end": params["dtend"],
            "top_level_keys": sorted(list(result.keys())),
            "reported_count": count,
            "download_probe": download_probe,
        }
    except Exception as exc:
        return {"reachable": False, "url": url, "error": str(exc)}


def credential_probe() -> dict[str, Any]:
    candidate_files = [
        Path("eumdac_credentials.local.json"),
        Path("cmsaf_credentials.local.json"),
        Path("sarah3_credentials.local.json"),
    ]
    return {
        "eumdac_installed": importlib.util.find_spec("eumdac") is not None,
        "credential_files_present": [str(p) for p in candidate_files if p.exists()],
        "note": "Credentials are not read or printed. This probe only checks whether a local file exists inside PDS.",
    }


def write_next_steps(metadata_status: dict[str, Any], credential_status: dict[str, Any]) -> None:
    lines = [
        "# v25.1 SARAH-3 Access Audit",
        "",
        "## Purpose",
        "",
        "SARAH-3 is a direct solar-radiation satellite climate data record. It is relevant because it can provide surface incoming shortwave radiation, direct irradiance, sunshine duration, and effective cloud albedo signals that are physically closer to the target than generic station weather.",
        "",
        "## Current access status",
        "",
        f"- EUMETSAT browse metadata reachable: {metadata_status.get('reachable')}",
        f"- `eumdac` installed in the project virtualenv: {credential_status.get('eumdac_installed')}",
        f"- Local credential files present: {len(credential_status.get('credential_files_present', []))}",
        "",
        "## Recommended extraction design",
        "",
        "1. Request SARAH-3 files covering each station-year period and a small bounding box around every TAHMO station.",
        "2. Extract nearest or distance-weighted grid-cell values at the station coordinates.",
        "3. Convert native timestamp to the TAHMO 15-minute grid with explicit missingness flags.",
        "4. Build features: SARAH SIS, direct irradiance, sunshine duration, effective cloud albedo, SARAH clearness index, SARAH minus LSASAF DSSF, and SARAH minus CAMS/NASA clear-sky references.",
        "5. Validate first as a residual feature against existing v16.6/v22.2/v15.9.2 OOF predictions before any expensive full retraining.",
        "",
        "## Blocker if any",
        "",
        "If only the browse metadata is reachable but file download requires a CM SAF/EUMETSAT authenticated order, create the required credential file inside PDS and rerun this script after installing/authorizing the official client.",
    ]
    (RUN_DIR / "SARAH3_NEXT_STEPS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_config(args: argparse.Namespace) -> None:
    config = {
        "version": "v25.1_sarah3_audit",
        "objective": "Check SARAH-3 access and create a station/time extraction plan.",
        "collection_id": SARAH3_COLLECTION_ID,
        "eumetsat_browse_url": EUMETSAT_BROWSE_URL,
        "eumetsat_dates_url": EUMETSAT_DATES_URL,
        "eumetsat_osdd_url": EUMETSAT_OSDD_URL,
        "cmsaf_wui_url": CMSAF_WUI_URL,
        "args": vars(args),
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--download-sample", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    write_config(args)
    log("[start] v25.1 SARAH-3 access audit")

    tahmo = load_tahmo_catalog(smoke_test=args.smoke_test)
    plan = build_station_request_plan(tahmo)
    log(f"[plan] station request rows={len(plan)}")

    metadata_status: dict[str, Any] = {"reachable": False, "url": EUMETSAT_BROWSE_URL}
    try:
        metadata = fetch_json(EUMETSAT_BROWSE_URL, timeout=args.timeout)
        (RUN_DIR / "sarah3_collection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata_status.update(
            {
                "reachable": True,
                "top_level_keys": sorted(list(metadata.keys())),
                "title": metadata.get("title") or metadata.get("properties", {}).get("title"),
            }
        )
        log("[metadata] EUMETSAT browse metadata fetched")
    except Exception as exc:
        metadata_status.update({"error": str(exc)})
        log(f"[warn] EUMETSAT browse metadata fetch failed: {exc}")

    dates_status: dict[str, Any] = {"reachable": False, "url": EUMETSAT_DATES_URL}
    try:
        dates = fetch_json(EUMETSAT_DATES_URL, timeout=args.timeout)
        (RUN_DIR / "sarah3_dates.json").write_text(json.dumps(dates, indent=2), encoding="utf-8")
        dates_status.update({"reachable": True, "top_level_keys": sorted(list(dates.keys()))})
        log("[dates] EUMETSAT browse dates fetched")
    except Exception as exc:
        dates_status.update({"error": str(exc)})
        log(f"[warn] EUMETSAT dates fetch failed: {exc}")

    osdd_status: dict[str, Any] = {"reachable": False, "url": EUMETSAT_OSDD_URL}
    try:
        osdd = fetch_text(EUMETSAT_OSDD_URL, timeout=args.timeout)
        (RUN_DIR / "sarah3_osdd.xml").write_text(osdd, encoding="utf-8")
        osdd_status.update({"reachable": True, "length": len(osdd)})
        log("[osdd] EUMETSAT OpenSearch description fetched")
    except Exception as exc:
        osdd_status.update({"error": str(exc)})
        log(f"[warn] EUMETSAT OSDD fetch failed: {exc}")

    credential_status = credential_probe()
    sample_search_status = sample_product_search(plan, timeout=args.timeout, download_sample=args.download_sample)
    if sample_search_status.get("reachable"):
        log("[search] SARAH-3 sample product search fetched")
    else:
        log(f"[warn] SARAH-3 sample product search failed: {sample_search_status.get('error')}")
    report = {
        "metadata_status": metadata_status,
        "dates_status": dates_status,
        "osdd_status": osdd_status,
        "sample_search_status": sample_search_status,
        "credential_status": credential_status,
        "station_count": int(len(tahmo)),
        "recommendation": "Use SARAH-3 as a satellite solar/cloud-albedo residual source if authenticated download is available.",
    }
    (RUN_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RUN_DIR / "sarah3_access_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_next_steps(metadata_status, credential_status)
    log(f"[done] reachable={metadata_status.get('reachable')} eumdac={credential_status.get('eumdac_installed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
