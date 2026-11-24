#!/usr/bin/env python3
"""Search NASA CMR and NOAA cloud storage for VIIRS satellite granules.

Query NASA CMR for geolocation granules over a bounding box and time range,
then find matching data granules on NOAA mirrors (AWS S3, Google Cloud, Azure).

Usage:
  python noaa_cmr_search.py --bbox -170 51 -130 72 \\
      --start 2026-10-01T00:00 --end 2026-10-02T00:00
  python noaa_cmr_search.py --bbox -170 51 -130 72 --start 2026-10-01 \\
      --end 2026-10-02 --provider gcs --sat n21 --download /path/to/dir

Requires: requests, s3fs (aws), gcsfs (gcs), or adlfs (azure) for cloud access.
    For refining passes (NASA granules are 6 minutes vs. NOAA's 85 seconds, so some NOAA granules will be outside bounds),
    requires h5py, shapely, antimeridian.
"""

from __future__ import annotations
import argparse
import datetime as dt
import re
from collections import defaultdict
import numpy as np
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from pprint import pformat

import requests

if TYPE_CHECKING:
    from shapely.geometry import Polygon

log = logging.getLogger("viirs_search")
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(handler)
    log.propagate = False  # Don't propagate to root logger

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"

# NASA I-band geolocation collections
CMR_SHORT_NAME = {
    "n20": {"iband": "VJ103IMG", "mband": "VJ103MOD", "dnb": "VJ103DNB"},
    "n21": {"iband": "VJ203IMG", "mband": "VJ203MOD", "dnb": "VJ203DNB"},
    "snpp": {"iband": "VNP03IMG", "mband": "VNP03MOD", "dnb": "VNP03DNB"},
}

# Map product types to their S3 folder paths and GEO product types
PRODUCT_PATHS = {
    # I-Bands
    "I1": ("VIIRS-I1-SDR", "VIIRS-IMG-GEO-TC"),
    "I2": ("VIIRS-I2-SDR", "VIIRS-IMG-GEO-TC"),
    "I3": ("VIIRS-I3-SDR", "VIIRS-IMG-GEO-TC"),
    "I4": ("VIIRS-I4-SDR", "VIIRS-IMG-GEO-TC"),
    "I5": ("VIIRS-I5-SDR", "VIIRS-IMG-GEO-TC"),
    # M-Bands
    "M1": ("VIIRS-M1-SDR", "VIIRS-MOD-GEO-TC"),
    "M2": ("VIIRS-M2-SDR", "VIIRS-MOD-GEO-TC"),
    "M3": ("VIIRS-M3-SDR", "VIIRS-MOD-GEO-TC"),
    "M4": ("VIIRS-M4-SDR", "VIIRS-MOD-GEO-TC"),
    "M5": ("VIIRS-M5-SDR", "VIIRS-MOD-GEO-TC"),
    "M6": ("VIIRS-M6-SDR", "VIIRS-MOD-GEO-TC"),
    "M7": ("VIIRS-M7-SDR", "VIIRS-MOD-GEO-TC"),
    "M8": ("VIIRS-M8-SDR", "VIIRS-MOD-GEO-TC"),
    "M9": ("VIIRS-M9-SDR", "VIIRS-MOD-GEO-TC"),
    "M10": ("VIIRS-M10-SDR", "VIIRS-MOD-GEO-TC"),
    "M11": ("VIIRS-M11-SDR", "VIIRS-MOD-GEO-TC"),
    "M12": ("VIIRS-M12-SDR", "VIIRS-MOD-GEO-TC"),
    "M13": ("VIIRS-M13-SDR", "VIIRS-MOD-GEO-TC"),
    "M14": ("VIIRS-M14-SDR", "VIIRS-MOD-GEO-TC"),
    "M15": ("VIIRS-M15-SDR", "VIIRS-MOD-GEO-TC"),
    "M16": ("VIIRS-M16-SDR", "VIIRS-MOD-GEO-TC"),
    # NCC Band
    "DNB": ("VIIRS-DNB-SDR", "VIIRS-DNB-GEO"),
}

GEO_PRODUCTS = {
    "VIIRS-IMG-GEO-TC": "iband",  # NOAA I-band terrain-corrected geolocation (GITCO)
    "VIIRS-MOD-GEO-TC": "mband",
    "VIIRS-DNB-GEO": "dnb",
}

# provider -> satellite -> bucket/container
BUCKETS = {
    "aws": {
        "n20": "noaa-nesdis-n20-pds",
        "n21": "noaa-nesdis-n21-pds",
        "snpp": "noaa-nesdis-snpp-pds",
    },
    "gcs": {
        "n20": "gcp-noaa-nesdis-n20",
        "n21": "gcp-noaa-nesdis-n21",
        "snpp": "gcp-noaa-nesdis-snpp",
    },
    "azure": {"n20": "noaa-20", "n21": "noaa-21", "snpp": "snpp"},
}
SCHEME = {"aws": "s3://", "gcs": "gs://", "azure": "abfs://"}

G_RING_LOCS = {
    "iband": "/Data_Products/VIIRS-IMG-GEO-TC/VIIRS-IMG-GEO-TC_Gran_0",
    "mband": "/Data_Products/VIIRS-MOD-GEO-TC/VIIRS-MOD-GEO-TC_Gran_0",
    "dnb": "/Data_Products/VIIRS-DNB-GEO/VIIRS-DNB-GEO_Gran_0",
}


def parse_dt(s: str) -> dt.datetime:
    """Parse ISO format datetime string, handling 'Z' timezone indicator.

    Args:
        s: ISO format datetime string (e.g., '2026-10-01T00:00:00Z')

    Returns:
        datetime object with timezone info removed
    """
    d = dt.datetime.fromisoformat(s.replace("Z", ""))
    return d.replace(tzinfo=None)


def get_fs(provider: str):
    """Get a filesystem object for the specified cloud provider.

    Args:
        provider: Cloud provider name ('aws', 'gcs', or 'azure')

    Returns:
        Filesystem object (S3FileSystem, GCSFileSystem, or AzureBlobFileSystem)

    Raises:
        ValueError: If provider is not supported
    """
    if provider == "aws":
        import s3fs

        log.debug(f"Initializing anonymous S3 filesystem for AWS")
        return s3fs.S3FileSystem(anon=True)
    if provider == "gcs":
        import gcsfs

        log.debug(f"Initializing anonymous GCS filesystem for Google Cloud")
        return gcsfs.GCSFileSystem(token="anon")
    if provider == "azure":
        import adlfs

        log.debug(f"Initializing anonymous Azure filesystem")
        return adlfs.AzureBlobFileSystem(account_name="jpss", anon=True)
    raise ValueError(
        f"Unsupported provider: {provider}. Must be 'aws', 'gcs', or 'azure'"
    )


def split_bbox(
    w: float, s: float, e: float, n: float
) -> list[tuple[float, float, float, float]]:
    """Split a bounding box that may cross the antimeridian into safe sub-boxes.

    CMR queries are safest when they don't cross the antimeridian (180/-180 longitude).
    If the input bbox crosses it, split it into two boxes on each side.

    Args:
        w: West longitude (-180 to 180)
        s: South latitude (-90 to 90)
        e: East longitude (-180 to 180)
        n: North latitude (-90 to 90)

    Returns:
        List of (w, s, e, n) tuples; one box if no crossing, two if crossing antimeridian
    """
    if w <= e:
        return [(w, s, e, n)]
    return [(w, s, 180.0, n), (-180.0, s, e, n)]


def cmr_windows(
    bbox: tuple[float, float, float, float],
    start: dt.datetime,
    end: dt.datetime,
    short_name: str,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Query NASA CMR for geolocation granules intersecting a bbox and time range.

    Fetches all VIIRS geolocation granules from NASA's Common Metadata Repository
    that intersect the given bounding box within the specified time window.

    Args:
        bbox: Bounding box as (west, south, east, north) in degrees
        start: Start datetime (UTC)
        end: End datetime (UTC)
        short_name: CMR collection short name (e.g., 'VJ103IMG' for NOAA-20 I-band)

    Returns:
        Sorted list of unique (start, end) datetime tuples for NASA granules

    Raises:
        requests.HTTPError: If CMR API request fails
    """
    wins = []
    for w, s, e, n in split_bbox(*bbox):
        log.debug(
            f"Querying CMR for {short_name} in bbox ({w}, {s}, {e}, {n}) from {start} to {end}"
        )
        params = {
            "short_name": short_name,
            "bounding_box": f"{w},{s},{e},{n}",
            "temporal": f"{start:%Y-%m-%dT%H:%M:%SZ},{end:%Y-%m-%dT%H:%M:%SZ}",
            "page_size": 2000,
        }
        headers = {}
        page_count = 0
        while True:
            page_count += 1
            log.debug(f"Fetching CMR page {page_count}")
            r = requests.get(CMR, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            entries = r.json()["feed"]["entry"]
            log.debug(f"CMR page {page_count} returned {len(entries)} granules")
            for g in entries:
                wins.append(
                    (parse_dt(g["time_start"][:19]), parse_dt(g["time_end"][:19]))
                )
            token = r.headers.get("CMR-Search-After")
            if not token:
                break
            headers = {"CMR-Search-After": token}

    unique_wins = sorted(set(wins))
    log.debug(f"CMR query returned {len(unique_wins)} unique granule windows")
    return unique_wins


FNAME = re.compile(r"_d(\d{8})_t(\d{6})(\d)_e(\d{6})(\d)_")


def noaa_times(key: str) -> tuple[dt.datetime, dt.datetime] | None:
    """Extract start and end times from a NOAA VIIRS granule filename.

    NOAA granule filenames encode timestamps in a specific format:
    _d<YYYYMMDD>_t<HHMMSS><tenth_of_second>_e<HHMMSS><tenth_of_second>_

    Args:
        key: NOAA granule filename or S3 key

    Returns:
        Tuple of (start_time, end_time) as datetime objects, or None if filename doesn't match pattern
    """
    m = FNAME.search(key)
    if not m:
        return None
    day, t0, f0, t1, f1 = m.groups()
    s = dt.datetime.strptime(day + t0, "%Y%m%d%H%M%S") + dt.timedelta(
        seconds=int(f0) / 10
    )
    e = dt.datetime.strptime(day + t1, "%Y%m%d%H%M%S") + dt.timedelta(
        seconds=int(f1) / 10
    )
    if e < s:  # granule crosses midnight
        e += dt.timedelta(days=1)
    return s, e


def days_covered(
    windows: list[tuple[dt.datetime, dt.datetime]], pad: dt.timedelta
) -> list[dt.date]:
    """Calculate all calendar days that need to be checked for data.

    Given time windows (from CMR granules), expand them by the padding amount
    and collect all unique calendar days that fall within the expanded windows.

    Args:
        windows: List of (start, end) datetime tuples
        pad: Timedelta padding to expand each window

    Returns:
        Sorted list of unique dates
    """
    days = set()
    for ws, we in windows:
        d = (ws - pad).date()
        while d <= (we + pad).date():
            days.add(d)
            d += dt.timedelta(days=1)
    return sorted(days)


def find_noaa_keys(
    fs,
    bucket: str,
    windows: list[tuple[dt.datetime, dt.datetime]],
    pad: dt.timedelta,
    product: str,
    sort_into_windows: bool = False,
) -> list[str] | dict:
    """Find NOAA VIIRS granule keys on cloud storage that overlap with CMR windows.

    Lists NOAA cloud storage by day, filters granules by filename timestamps,
    and keeps those whose times overlap a NASA CMR window (with padding tolerance).

    Args:
        fs: Filesystem object (S3, GCS, or Azure)
        bucket: Cloud storage bucket path prefix (e.g., 'noaa-nesdis-n20-pds')
        windows: List of (start, end) datetime tuples from CMR
        pad: Timedelta tolerance for time matching
        product: Product folder name (e.g., 'VIIRS-IMG-GEO-TC')
        sort_into_windows: If True, return dict mapping windows to granule lists;
                          if False, return flat sorted list

    Returns:
        Dict mapping windows to granule lists (if sort_into_windows=True),
        or sorted list of granule keys (if sort_into_windows=False)
    """
    if sort_into_windows:
        windows = sorted(windows)
    grouped = {w: [] for w in windows} if sort_into_windows else None
    keys = []
    days = days_covered(windows, pad)
    log.debug(f"Searching {len(days)} days of data for product {product}")

    for day in days:
        folder = f"{bucket}/{product}/{day:%Y/%m/%d}"
        try:
            listing = fs.ls(folder, detail=False)
        except FileNotFoundError:
            continue
        log.debug(f"Found {len(listing)} files on {day}")

        for k in listing:
            t = noaa_times(k)
            if not t:
                continue
            if sort_into_windows:
                for w in windows:
                    w_start, w_end = w
                    if t[0] < w_end + pad and t[1] > w_start - pad:
                        grouped[w].append(k)
            elif any(
                t[0] < w_end + pad and t[1] > w_start - pad
                for w_start, w_end in windows
            ):
                keys.append(k)

    result = (
        grouped
        if sort_into_windows
        else sorted(set(keys), key=lambda k: noaa_times(k)[0])
    )
    log.debug(
        f"Found {len(result) if isinstance(result, dict) else len(result)} granules for product {product}"
    )
    return result


def find_noaa_keys_from_geo_keys(
    fs, bucket: str, product: str, geo_keys: list[str]
) -> list[str]:
    """Find data product granules matching the times of geolocation granules.

    Given a list of geolocation (GEO) granule keys, extract their time tokens
    and find corresponding data product granules on the same day that share those tokens.

    Args:
        fs: Filesystem object (S3, GCS, or Azure)
        bucket: Cloud storage bucket path prefix
        product: Product folder name (e.g., 'VIIRS-I4-SDR')
        geo_keys: List of geolocation granule keys

    Returns:
        Sorted list of data product granule keys
    """
    # group wanted (t, e) tokens by day folder
    wanted = defaultdict(set)
    for geo_key in geo_keys:
        _sat, _prod, y, m, d, filename = geo_key.split("/")
        _, _, _, t, e, *_ = filename.split("_")
        wanted[f"{bucket}/{product}/{y}/{m}/{d}"].add(f"_{t}_{e}_")

    keys = set()
    log.debug(
        f"Searching {len(wanted)} day folders for {product} granules matching geolocation times"
    )
    for folder, tokens in wanted.items():
        try:
            listing = fs.ls(folder, detail=False)
        except FileNotFoundError:
            continue
        for k in listing:
            if k.endswith(".h5") and any(tok in k for tok in tokens):
                keys.add(k)

    log.debug(f"Found {len(keys)} {product} granules matching geolocation times")
    return sorted(keys, key=lambda k: noaa_times(k)[0])


def sort_into_passes(keys: list[str]) -> dict[str, list[str]]:
    """Group sequential granules into satellite passes based on time gaps.

    Consecutive granules with time gaps < 120 seconds belong to the same pass.
    Each new pass is identified by its first granule key.

    Args:
        keys: Sorted list of granule keys

    Returns:
        Dictionary mapping first granule key of each pass to list of granule keys in that pass
    """
    passes = {}
    current_group = None
    prev_key = None
    prev_start_time = None

    for key in keys:
        start_time = noaa_times(key)[0]
        if prev_key is None or start_time - prev_start_time >= dt.timedelta(
            seconds=120
        ):
            current_group = key
            passes[current_group] = []
        passes[current_group].append(key)
        prev_start_time = start_time
        prev_key = key

    log.debug(f"Grouped {len(keys)} granules into {len(passes)} passes")
    return passes


def lon_winding(lon: np.ndarray) -> float:
    """Calculate the net winding number of a longitude array around the globe.

    The winding number indicates how many times the path winds around the Earth
    in longitude. Used to detect pole-enclosing regions.

    Args:
        lon: 1D numpy array of longitude values in degrees

    Returns:
        Winding number (approximately -1, 0, or +1)
    """
    d = np.diff(lon)
    d = (d + 180) % 360 - 180
    return d.sum() / 360


def ring_to_geometry(lon: np.ndarray, lat: np.ndarray) -> Polygon:
    """Convert geographic ring coordinates to a valid Shapely Polygon.

    Handles edge cases like antimeridian crossing, pole enclosure, and coordinates
    outside the valid [-90, 90] / [-180, 180] ranges.

    Args:
        lon: 1D numpy array of longitude values in degrees
        lat: 1D numpy array of latitude values in degrees

    Returns:
        Valid Shapely Polygon geometry

    Raises:
        ValueError: If fewer than 4 valid points remain after filtering, or if
                   the resulting polygon is invalid
    """
    from shapely.geometry import box, Polygon
    import antimeridian
    from shapely.affinity import translate
    from shapely.validation import explain_validity
    from shapely.ops import unary_union

    ok = (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180)
    lat, lon = lat[ok].astype("float64"), lon[ok].astype("float64")
    if lat.size < 4:
        raise ValueError(
            f"ring has {lat.size} valid points after filtering; check (lon, lat) order"
        )
    w = round(lon_winding(lon))
    if w == 0:
        # unwrap lons so the ring is continuous, then check true orientation
        d = (np.diff(lon) + 180) % 360 - 180
        lon_u = lon[0] + np.concatenate([[0], np.cumsum(d)])
        poly = Polygon(zip(lon_u, lat))
        main = poly.intersection(box(-180, -90, 180, 90))
        west = translate(poly.intersection(box(-540, -90, -180, 90)), xoff=360)
        east = translate(poly.intersection(box(180, -90, 540, 90)), xoff=-360)
        fixed = unary_union([main, west, east])
    else:  # pole-enclosing
        want = 1 if lat.mean() > 0 else -1
        if w != want:
            lon, lat = lon[::-1], lat[::-1]
        poly = Polygon(zip(lon, lat))
        fixed = antimeridian.fix_polygon(poly, fix_winding=False)

    if not fixed.is_valid:
        raise ValueError(explain_validity(fixed))
    return fixed


def granule_ring(fs, geo_key: str, g_ring_loc: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract G-Ring geolocation coordinates from an HDF5 geolocation file.

    Reads the G-Ring bounding polygon from VIIRS geolocation (GEO) files stored
    in cloud storage.

    Args:
        fs: Filesystem object (S3, GCS, or Azure)
        geo_key: Path to the HDF5 geolocation file on cloud storage
        g_ring_loc: HDF5 internal path to the granule containing G-Ring attributes

    Returns:
        Tuple of (lon, lat) numpy arrays forming a closed ring
    """
    import h5py

    with fs.open(
        geo_key,
        "rb",
        block_size=256 * 1024,
        cache_type="blockcache",
        cache_options={"maxblocks": 32},
    ) as f:
        with h5py.File(f, "r") as h5:
            gran = h5[g_ring_loc]
            lat = np.asarray(gran.attrs["G-Ring_Latitude"]).ravel()
            lon = np.asarray(gran.attrs["G-Ring_Longitude"]).ravel()
    return np.append(lon, lon[0]), np.append(lat, lat[0])


def geo_file_intersects(fs, geo_key: str, bbox: Polygon, g_ring_loc: str) -> bool:
    """Check if a geolocation granule's bounding polygon intersects a bbox.

    Extracts the G-Ring coordinates from the HDF5 file and tests geometric intersection.
    If the G-Ring is invalid, the granule is kept (conservative approach).

    Args:
        fs: Filesystem object (S3, GCS, or Azure)
        geo_key: Path to the HDF5 geolocation file
        bbox: Shapely Polygon representing the search area
        g_ring_loc: HDF5 internal path to granule with G-Ring attributes

    Returns:
        True if granule intersects bbox (or if G-Ring is invalid), False otherwise
    """
    try:
        poly = ring_to_geometry(*granule_ring(fs, geo_key, g_ring_loc))
    except ValueError as err:
        log.warning(f"Invalid G-Ring geometry for {geo_key}: {err}; keeping granule")
        return True
    return poly.intersects(bbox)


def refine_pass(fs, geo_keys: list[str], bbox, g_ring_loc: str) -> list[str]:
    """Trim leading and trailing granules from a pass that don't intersect bbox.

    Within a satellite pass, some leading/trailing granules may fall outside
    the search area. This function removes them while keeping contiguous interior granules.

    Args:
        fs: Filesystem object (S3, GCS, or Azure)
        geo_keys: Ordered list of geolocation granule keys in a pass
        bbox: Shapely Polygon representing the search area
        g_ring_loc: HDF5 internal path to granule with G-Ring attributes

    Returns:
        Trimmed list of granule keys
    """
    start, end = 0, len(geo_keys)

    log.debug(f"Refining pass with {len(geo_keys)} granules")

    while start < end and not geo_file_intersects(
        fs, geo_keys[start], bbox, g_ring_loc
    ):
        log.debug(f"Removing leading granule: {geo_keys[start]}")
        start += 1

    while end > start and not geo_file_intersects(
        fs, geo_keys[end - 1], bbox, g_ring_loc
    ):
        log.debug(f"Removing trailing granule: {geo_keys[end - 1]}")
        end -= 1

    removed = len(geo_keys) - (end - start)
    log.debug(
        f"Refined pass: {len(geo_keys)} → {end - start} granules (removed {removed})"
    )

    return geo_keys[start:end]


def bbox_polygon(w: float, s: float, e: float, n: float) -> Polygon:
    """Convert a bounding box to a Shapely Polygon, handling antimeridian crossing.

    Args:
        w: West longitude (-180 to 180)
        s: South latitude (-90 to 90)
        e: East longitude (-180 to 180)
        n: North latitude (-90 to 90)

    Returns:
        Shapely Polygon representing the bbox
    """
    from shapely.geometry import box, Polygon
    from shapely.ops import unary_union

    if w > e:  # crosses the antimeridian
        return unary_union([box(w, s, 180, n), box(-180, s, e, n)])
    return box(w, s, e, n)


def download_pass(fs, pass_info: dict, dest_root: str) -> Path:
    """Download all granules in a pass to local disk.

    Downloads geolocation and product granules, using .part temporary files
    to support resumable downloads.

    Args:
        fs: Filesystem object (S3, GCS, or Azure)
        pass_info: Dictionary with 'folder_name', 'geo_items', and 'products' keys
        dest_root: Root directory for downloads

    Returns:
        Path to the downloaded pass folder
    """
    folder = Path(dest_root) / pass_info["folder_name"]
    folder.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading pass to {folder}")

    remote = list(pass_info["geo_items"])
    for keys in pass_info.get("products", {}).values():
        remote.extend(keys)

    todo = [(k, folder / Path(k).name) for k in remote]
    todo = [(k, p) for k, p in todo if not p.exists()]

    if not todo:
        log.info(f"All files already downloaded for pass {pass_info['folder_name']}")
        return folder

    log.info(f"Downloading {len(todo)} files for pass {pass_info['folder_name']}")
    parts = [str(p) + ".part" for _, p in todo]
    fs.get([k for k, _ in todo], parts)

    for (_, final), part in zip(todo, parts):
        os.replace(part, final)

    log.info(f"Download complete for pass {pass_info['folder_name']}")
    return folder


def noaa_cmr_search(
    bbox: tuple[float, float, float, float],
    start: str | dt.datetime,
    end: str | dt.datetime,
    products: list[str],
    satellites: list[str],
    provider: str = "gcs",
    pad_seconds: int = 60,
    refine: bool = True,
    download_dir: str | None = None,
) -> dict:
    """Query NASA CMR and NOAA mirrors for VIIRS granules.

    Main programmatic entry point for VIIRS granule search. Can be called directly
    from other Python modules or via the command-line interface.

    Args:
        bbox: Bounding box as (west, south, east, north) in degrees.
              W > E means crossing antimeridian.
        start: Start datetime (UTC). Can be string (ISO format) or datetime object.
        end: End datetime (UTC). Can be string (ISO format) or datetime object.
        products: List of product names (e.g., ['I4', 'I5', 'M1']). Must be from PRODUCT_PATHS keys.
        satellites: List of satellite codes (e.g., ['n20', 'n21', 'snpp']).
        provider: Cloud provider ('gcs', 'aws', or 'azure'). Defaults to 'gcs'.
        pad_seconds: Tolerance in seconds when matching NASA to NOAA granules. Defaults to 60.
        refine: Whether to trim passes to bbox boundaries. Defaults to True.
        download_dir: If provided, download granules to this directory. If None, only list results.

    Returns:
        Dictionary organized as {satellite: {geo_type: [pass_info_dicts]}} where each
        pass_info_dict contains folder_name, min_time, max_time, geo_items, and optional products.

    Raises:
        ValueError: If products are not in PRODUCT_PATHS or if end <= start.
    """
    # Parse datetime inputs
    if isinstance(start, str):
        start = parse_dt(start)
    if isinstance(end, str):
        end = parse_dt(end)

    if end <= start:
        raise ValueError(f"End time ({end}) must be after start time ({start})")

    log.info(f"Starting search for VIIRS granules in bbox {bbox} from {start} to {end}")
    log.debug(
        f"Options: provider={provider}, satellites={satellites}, products={products}, refine={refine}"
    )

    # Validate products
    if invalid_products := [
        p for p in products if p.upper() not in PRODUCT_PATHS.keys()
    ]:
        raise ValueError(
            f"Invalid products {invalid_products}. Must be from {list(PRODUCT_PATHS.keys())}"
        )

    # Build product mapping
    product_tuples = [PRODUCT_PATHS[p.upper()] for p in products]
    geo_products = set([p[1] for p in product_tuples])
    product_types = defaultdict(list)
    for p in product_tuples:
        product_types[GEO_PRODUCTS[p[1]]].append(p[0])

    fs = get_fs(provider)
    pad = dt.timedelta(seconds=pad_seconds)

    sat_groups = {}
    for sat in satellites:
        log.info(f"Processing satellite {sat}")

        geo_groups = {}
        for geo_product in geo_products:
            geo_product_type = GEO_PRODUCTS[geo_product]
            log.info(
                f"  Querying NASA CMR for {sat} {geo_product_type} geolocation product ({start} to {end})"
            )

            short_name = CMR_SHORT_NAME[sat].get(geo_product_type)
            if not short_name:
                log.warning(f"No CMR short_name for {sat} {geo_product_type}, skipping")
                continue
            windows = cmr_windows(bbox, start, end, short_name)
            log.info(f"  CMR returned {len(windows)} granule windows intersecting bbox")
            if not windows:
                log.info(f"  No granules found for {sat} {geo_product_type}, skipping")
                continue

            log.info(
                f"  Searching {provider} mirror for matching NOAA {geo_product_type} granules"
            )
            geo_keys = find_noaa_keys(
                fs, BUCKETS[provider][sat], windows, pad, geo_product
            )
            log.info(
                f"  Found {len(geo_keys)} NOAA {geo_product_type} granules on {provider}"
            )

            passes = sort_into_passes(geo_keys)
            log.info(f"  Organized into {len(passes)} satellite passes")

            refined_pass_info = []
            for index, (k, v) in enumerate(passes.items(), start=1):
                if refine:
                    log.info(
                        f"    {index}/{len(passes)}: Refining geometry to clip out excess granules..."
                    )
                    v = refine_pass(
                        fs, v, bbox_polygon(*bbox), G_RING_LOCS[geo_product_type]
                    )
                if len(v) > 0:
                    min_time = min([noaa_times(i)[0] for i in v])
                    max_time = max([noaa_times(i)[1] for i in v])
                    pass_dict = {
                        "folder_name": f"{sat.upper()}.{min_time.strftime('%Y%m%d.%H%M%S')}",
                        "min_time": min_time,
                        "max_time": max_time,
                        "geo_items": v,
                    }
                    refined_pass_info.append(pass_dict)

            log.info(
                f"  {len(refined_pass_info)} passes intersect bbox after refinement"
            )

            # Find data product granules
            for pass_info in refined_pass_info:
                for product in product_types[geo_product_type]:
                    log.debug(
                        f"  Searching for {product} granules matching {pass_info['folder_name']}"
                    )
                    products_found = find_noaa_keys_from_geo_keys(
                        fs, BUCKETS[provider][sat], product, pass_info["geo_items"]
                    )
                    log.info(
                        f"    {product}: {len(products_found)} granules for {pass_info['folder_name']}"
                    )
                    if products_found:
                        pass_info.setdefault("products", {})[product] = products_found

                # Download if requested
                if download_dir:
                    download_pass(fs, pass_info, download_dir)

            geo_groups[GEO_PRODUCTS[geo_product]] = refined_pass_info
        sat_groups[sat] = geo_groups

    total_passes = sum(len(gg) for g in sat_groups.values() for gg in g.values())
    log.info(
        f"Search complete. Summary: {len(sat_groups)} satellites, {total_passes} total passes"
    )

    return sat_groups


def main() -> None:
    """CLI entry point: parse arguments and call noaa_cmr_search()."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--products",
        nargs="+",
        type=str.upper,
        choices=list(PRODUCT_PATHS),
        default=["I4"],
        metavar="PRODUCT",
        help=f"One or more of: {', '.join(PRODUCT_PATHS)} (default: %(default)s)",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("W", "S", "E", "N"),
        help="west south east north, degrees (W > E means crossing 180)",
    )
    parser.add_argument("--start", required=True, help="UTC, e.g. 2026-10-01T00:00")
    parser.add_argument("--end", required=True, help="UTC, e.g. 2026-10-02T00:00")
    parser.add_argument("--provider", choices=BUCKETS, default="gcs")
    parser.add_argument(
        "--sat", nargs="+", choices=list(CMR_SHORT_NAME), default=["n20"]
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=60,
        help="seconds of slack when matching NASA to NOAA granules (default 60)",
    )
    parser.add_argument(
        "--format",
        choices=["url", "key"],
        default="url",
        help="url = scheme://bucket/key (default), key = bucket/key",
    )
    parser.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Skip trimming passes to the bbox",
    )
    parser.add_argument(
        "--download",
        metavar="DIR",
        help="Download passes into DIR/<pass folder>/; omit to just list",
    )
    args = parser.parse_args()

    # Call the main search function
    results = noaa_cmr_search(
        bbox=tuple(args.bbox),
        start=args.start,
        end=args.end,
        products=args.products,
        satellites=args.sat,
        provider=args.provider,
        pad_seconds=args.pad,
        refine=args.refine,
        download_dir=args.download,
    )

    # Print results summary
    log.info("Results:\n%s", pformat(results, width=100, sort_dicts=False))


if __name__ == "__main__":
    main()
