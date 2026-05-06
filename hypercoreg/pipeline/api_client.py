"""Sentinel-2 CDSE API helpers.

This module is the migration target for query/ranking/download runtime logic
now owned by the modular pipeline runtime.
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from time import perf_counter, sleep
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from hypercoreg.pipeline import auth
from hypercoreg.pipeline.time_utils import _to_utc_datetime
from hypercoreg.utils import CDSEAuthenticationError, SentinelNotFoundError

logger = logging.getLogger("COREG_PROCESSING")

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
HTTP_TIMEOUT_S = 30
HTTP_TIMEOUT_CONNECT_READ = (30, 180)
HTTP_QUERY_RETRY_ATTEMPTS = 3
HTTP_DOWNLOAD_RETRY_ATTEMPTS = 3


def _emit_progress(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    stage: str,
    scene_idx: int = 1,
    scene_total: int = 1,
    status: str = "running",
    **extra: Any,
) -> None:
    if progress_callback is None:
        return
    payload = {
        "stage": str(stage),
        "scene_idx": int(scene_idx),
        "scene_total": max(1, int(scene_total)),
        "status": str(status),
    }
    if extra:
        payload.update(extra)
    try:
        progress_callback(payload)
    except Exception as exc:
        logger.debug("Progress callback failed: %s", exc)


def _get_attr(attributes, name, default=100.0):
    for attr in attributes:
        if attr.get("Name") == name:
            try:
                return float(attr.get("Value"))
            except Exception:
                return default
    return default


def _bbox_to_wkt(bbox):
    minx, miny, maxx, maxy = bbox
    return f"POLYGON(({minx} {miny},{minx} {maxy},{maxx} {maxy},{maxx} {miny},{minx} {miny}))"


def _query_s2(session, center_time, bbox, days_window=30, max_cloud=20):
    center_time = _to_utc_datetime(center_time)
    logger.info("Querying Sentinel-2...")
    logger.info("  Search window: +/-%s days from %s", days_window, center_time.strftime("%Y-%m-%d"))
    logger.info("  Max cloud cover: %s%%, Bounding box: %s", max_cloud, bbox)

    start = (center_time - timedelta(days=days_window)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end = (center_time + timedelta(days=days_window)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    wkt = _bbox_to_wkt(bbox)

    filter_expr = (
        "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
        f"Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' and a/OData.CSC.DoubleAttribute/Value le {max_cloud}) and "
        f"ContentDate/Start gt {start} and ContentDate/Start lt {end} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
    )

    params = {"$filter": filter_expr, "$expand": "Attributes"}
    attempts = max(1, int(HTTP_QUERY_RETRY_ATTEMPTS))
    response = None
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            response = session.get(CATALOGUE_URL, params=params, timeout=HTTP_TIMEOUT_S)
            if response.status_code == 200:
                break
            if auth._is_retryable_http_status(response.status_code) and attempt < attempts:
                delay_s = auth._retry_delay_seconds(attempt)
                logger.warning(
                    auth._fmt_issue(
                        "CDSE_QUERY",
                        f"Retryable HTTP {response.status_code} on attempt {attempt}/{attempts}; "
                        f"retrying in {delay_s:.1f}s.",
                    )
                )
                sleep(delay_s)
                continue
            logger.error(auth._fmt_issue("CDSE_QUERY", f"Query failed: {response.status_code} - {response.text[:500]}"))
            response.raise_for_status()
        except requests.RequestException as exc:
            if isinstance(exc, requests.HTTPError):
                status = exc.response.status_code if getattr(exc, "response", None) is not None else None
                if status in (400, 401, 403):
                    raise
            last_error = exc
            if attempt < attempts:
                delay_s = auth._retry_delay_seconds(attempt)
                logger.warning(
                    auth._fmt_issue(
                        "CDSE_QUERY",
                        f"Request failed on attempt {attempt}/{attempts}: {exc}; retrying in {delay_s:.1f}s.",
                    )
                )
                sleep(delay_s)
                continue
            raise RuntimeError(auth._fmt_issue("CDSE_QUERY", f"Request failed after retries: {exc}")) from exc

    if response is None:
        raise RuntimeError(auth._fmt_issue("CDSE_QUERY", f"No response returned: {last_error}"))

    items = response.json().get("value", [])
    logger.info("  Found %s Sentinel-2 products", len(items))
    return items


def _query_s2_with_retry(
    session,
    center_time,
    bbox,
    days_window=30,
    max_cloud=20,
    allow_gui_prompt=False,
    prompt_userpass_fn=None,
):
    try:
        return _query_s2(session, center_time, bbox, days_window, max_cloud), session
    except requests.HTTPError as exc:
        status = exc.response.status_code if getattr(exc, "response", None) is not None else None
        if status in (400, 401, 403):
            try:
                if auth._force_refresh_cdse_session(session):
                    logger.warning(
                        auth._fmt_issue(
                            "CDSE_QUERY",
                            f"Sentinel-2 query failed with HTTP {status}. "
                            "Refreshing the existing CDSE token and retrying.",
                        )
                    )
                    try:
                        return _query_s2(session, center_time, bbox, days_window, max_cloud), session
                    except requests.HTTPError as retry_exc:
                        retry_status = (
                            retry_exc.response.status_code
                            if getattr(retry_exc, "response", None) is not None
                            else None
                        )
                        if retry_status not in (400, 401, 403):
                            raise
                        logger.warning(
                            auth._fmt_issue(
                                "CDSE_QUERY",
                                f"Refreshed CDSE session still failed with HTTP {retry_status}.",
                            )
                        )
            except Exception as refresh_exc:
                logger.warning(
                    auth._fmt_issue(
                        "CDSE_QUERY",
                        f"CDSE token refresh unavailable: {refresh_exc}",
                    )
                )
            logger.warning(
                auth._fmt_issue(
                    "CDSE_QUERY",
                    f"Sentinel-2 query failed with HTTP {status}. "
                    "Prompting for CDSE username/password and retrying.",
                )
            )
            public_session = auth._create_public_session_with_retry(
                allow_gui_prompt=allow_gui_prompt,
                prompt_userpass_fn=prompt_userpass_fn,
            )
            try:
                return (
                    _query_s2(public_session, center_time, bbox, days_window, max_cloud),
                    public_session,
                )
            except requests.HTTPError as retry_exc:
                retry_status = (
                    retry_exc.response.status_code
                    if getattr(retry_exc, "response", None) is not None
                    else None
                )
                if retry_status in (400, 401, 403):
                    raise CDSEAuthenticationError(
                        auth._fmt_issue(
                            "CDSE_QUERY",
                            f"Sentinel-2 query authentication failed after retry (HTTP {retry_status}).",
                        )
                    ) from retry_exc
                raise
        raise


def _rank_s2_candidates(items, center_time, bbox, min_overlap=0.5):
    from shapely.geometry import Polygon, shape
    import shapely.wkt as shapely_wkt

    center_time = _to_utc_datetime(center_time)
    logger.info("Ranking S2 candidates (min overlap: %.1f%%)...", min_overlap * 100.0)

    minx, miny, maxx, maxy = bbox
    bbox_poly = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
    scored_candidates = []

    for item in items:
        geom = item["GeoFootprint"]
        geom_poly = shapely_wkt.loads(geom) if isinstance(geom, str) else shape(geom)
        inter = bbox_poly.intersection(geom_poly)
        overlap = inter.area / bbox_poly.area if bbox_poly.area > 0 else 0.0

        dt = datetime.fromisoformat(item["ContentDate"]["Start"].replace("Z", "+00:00")).astimezone(timezone.utc)
        cloud = _get_attr(item["Attributes"], "cloudCover", 100.0)

        if overlap < min_overlap:
            logger.debug("  %s: overlap=%.1f%% < %.1f%% - REJECTED", item["Name"], overlap * 100.0, min_overlap * 100.0)
            continue

        logger.info("  %s: overlap=%.1f%%, cloud=%.1f%%", item["Name"], overlap * 100.0, cloud)
        tdiff_h = abs((dt - center_time).total_seconds()) / 3600.0
        score = cloud + 0.5 * tdiff_h - 20.0 * overlap
        scored_candidates.append({"item": item, "score": score})

    if not scored_candidates:
        logger.error(auth._fmt_issue("CDSE_QUERY", "No suitable Sentinel-2 products found."))
        raise SentinelNotFoundError("No valid Sentinel-2 match found for this scene")

    scored_candidates.sort(key=lambda x: x["score"])
    logger.info("Found %s valid candidates. Top: %s", len(scored_candidates), scored_candidates[0]["item"]["Name"])
    return [x["item"] for x in scored_candidates]


def _download_s2_product(
    session,
    product,
    out_dir,
    allow_gui_prompt=False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    scene_idx: int = 1,
    scene_total: int = 1,
    prompt_userpass_fn=None,
):
    pid = product["Id"]
    download_urls = [
        f"https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value",
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({pid})/$value",
        f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({pid})/$value",
    ]
    zip_path = os.path.join(out_dir, f"{pid}.zip")
    os.makedirs(out_dir, exist_ok=True)

    def _is_valid_zip_quick(path: str) -> bool:
        if not os.path.exists(path):
            return False
        if not zipfile.is_zipfile(path):
            return False
        try:
            with zipfile.ZipFile(path) as zf:
                return len(zf.infolist()) > 0
        except Exception:
            return False

    if os.path.exists(zip_path):
        if _is_valid_zip_quick(zip_path):
            logger.info("S2 already downloaded (valid zip file).")
            return zip_path, session
        try:
            os.remove(zip_path)
        except Exception:
            pass

    logger.info("Downloading S2...")
    _emit_progress(progress_callback, "Downloading Sentinel-2 reference", scene_idx=scene_idx, scene_total=scene_total)

    def _attempt_download(sess: requests.Session) -> Tuple[bool, list[str], bool]:
        attempt_errors: list[str] = []
        auth_error = False
        audience_error = False
        download_attempts = max(1, int(HTTP_DOWNLOAD_RETRY_ATTEMPTS))

        for url in download_urls:
            for attempt in range(1, download_attempts + 1):
                try:
                    logger.info("  Trying endpoint: %s (attempt %s/%s)", url, attempt, download_attempts)
                    _emit_progress(
                        progress_callback,
                        "Downloading Sentinel-2 reference",
                        scene_idx=scene_idx,
                        scene_total=scene_total,
                        download_endpoint=url,
                        download_percent=0.0,
                        download_mb=0.0,
                    )
                    with sess.get(url, stream=True, timeout=HTTP_TIMEOUT_CONNECT_READ) as response:
                        if response.status_code >= 400:
                            msg = response.text[:500]
                            if response.status_code in (401, 403):
                                auth_error = True
                            if "DAT-ZIP-609" in msg or "Token audience not allowed" in msg:
                                audience_error = True
                            if auth._is_retryable_http_status(response.status_code) and attempt < download_attempts:
                                delay_s = auth._retry_delay_seconds(attempt)
                                logger.warning(
                                    auth._fmt_issue(
                                        "S2_DOWNLOAD",
                                        f"Retryable HTTP {response.status_code} for {url} "
                                        f"(attempt {attempt}/{download_attempts}); retrying in {delay_s:.1f}s.",
                                    )
                                )
                                sleep(delay_s)
                                continue
                            attempt_errors.append(f"{url} -> {response.status_code}: {msg}")
                            break

                        total_size = int(response.headers.get("Content-Length", "0") or "0")
                        bytes_written = 0
                        last_emit_t = perf_counter()
                        last_emit_bytes = 0
                        tmp_fd, tmp_path = tempfile.mkstemp(
                            prefix=f".{pid}.",
                            suffix=".part",
                            dir=out_dir,
                        )
                        try:
                            with os.fdopen(tmp_fd, "wb") as fh:
                                for chunk in response.iter_content(1024 * 1024):
                                    if not chunk:
                                        continue
                                    fh.write(chunk)
                                    bytes_written += len(chunk)
                                    now_t = perf_counter()
                                    if (now_t - last_emit_t) >= 1.0 or (bytes_written - last_emit_bytes) >= (25 * 1024 * 1024):
                                        event_kwargs = {
                                            "download_endpoint": url,
                                            "download_mb": round(bytes_written / (1024 * 1024), 2),
                                        }
                                        if total_size > 0:
                                            event_kwargs["download_total_mb"] = round(total_size / (1024 * 1024), 2)
                                            event_kwargs["download_percent"] = round((100.0 * bytes_written) / total_size, 2)
                                        _emit_progress(
                                            progress_callback,
                                            "Downloading Sentinel-2 reference",
                                            scene_idx=scene_idx,
                                            scene_total=scene_total,
                                            **event_kwargs,
                                        )
                                        last_emit_t = now_t
                                        last_emit_bytes = bytes_written
                            if _is_valid_zip_quick(tmp_path):
                                os.replace(tmp_path, zip_path)
                            else:
                                attempt_errors.append(f"{url} -> downloaded file is not a valid zip")
                                break
                        finally:
                            if os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass

                        if _is_valid_zip_quick(zip_path):
                            return True, attempt_errors, audience_error

                        try:
                            os.remove(zip_path)
                        except Exception:
                            pass
                        break
                except requests.RequestException as exc:
                    if attempt < download_attempts:
                        delay_s = auth._retry_delay_seconds(attempt)
                        logger.warning(
                            auth._fmt_issue(
                                "S2_DOWNLOAD",
                                f"Request error for {url} (attempt {attempt}/{download_attempts}): {exc}; "
                                f"retrying in {delay_s:.1f}s.",
                            )
                        )
                        sleep(delay_s)
                        continue
                    attempt_errors.append(f"{url} -> request error: {exc}")
                    break
                except Exception as exc:
                    attempt_errors.append(f"{url} -> request error: {exc}")
                    break

        return False, attempt_errors, auth_error or audience_error

    ok, errors, saw_auth_error = _attempt_download(session)
    if ok:
        return zip_path, session

    if saw_auth_error:
        try:
            if auth._force_refresh_cdse_session(session):
                logger.warning(
                    auth._fmt_issue(
                        "S2_DOWNLOAD",
                        "Sentinel-2 download authentication failed. "
                        "Refreshing the existing CDSE token and retrying.",
                    )
                )
                ok2, errors2, saw_auth_error_2 = _attempt_download(session)
                if ok2:
                    return zip_path, session
                errors.extend(errors2)
                saw_auth_error = saw_auth_error or saw_auth_error_2
        except Exception as refresh_exc:
            logger.warning(
                auth._fmt_issue(
                    "S2_DOWNLOAD",
                    f"CDSE token refresh unavailable: {refresh_exc}",
                )
            )

        fallback_session = None
        try:
            logger.warning(
                auth._fmt_issue(
                    "S2_DOWNLOAD",
                    "Sentinel-2 download authentication failed. Trying username/password token flow.",
                )
            )
            fallback_session = auth._create_public_session_with_retry(
                allow_gui_prompt=allow_gui_prompt,
                prompt_userpass_fn=prompt_userpass_fn,
            )
        except Exception as exc:
            logger.warning(
                auth._fmt_issue(
                    "S2_DOWNLOAD",
                    f"Username/password fallback unavailable: {exc}",
                )
            )

        if fallback_session is not None:
            ok2, errors2, saw_auth_error_2 = _attempt_download(fallback_session)
            if ok2:
                return zip_path, fallback_session
            errors.extend(errors2)
            saw_auth_error = saw_auth_error or saw_auth_error_2

    if saw_auth_error:
        raise CDSEAuthenticationError(
            auth._fmt_issue(
                "S2_DOWNLOAD",
                "CDSE authentication failed for product download. "
                "Please re-run and log in again with CDSE username/password. "
                f"Endpoint attempts: {' | '.join(errors[:3])}",
            )
        )

    if errors:
        raise RuntimeError(auth._fmt_issue("S2_DOWNLOAD", f"S2 download failed on all endpoints: {' | '.join(errors[:3])}"))

    return zip_path, session


__all__ = [
    "_bbox_to_wkt",
    "_download_s2_product",
    "_get_attr",
    "_query_s2",
    "_query_s2_with_retry",
    "_rank_s2_candidates",
]
