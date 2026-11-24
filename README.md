Search NASA CMR and NOAA cloud storage for VIIRS satellite granules.

Query NASA CMR for geolocation granules over a bounding box and time range,
then find matching data granules on NOAA mirrors (AWS S3, Google Cloud, Azure).

## Quick Install

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```
python noaa_cmr_search.py --help
usage: noaa_cmr_search.py [-h] [--products PRODUCT [PRODUCT ...]] --bbox W S E N --start START --end END [--provider {aws,gcs,azure}] [--sat {n20,n21,snpp} [{n20,n21,snpp} ...]]
                          [--pad PAD] [--format {url,key}] [--no-refine] [--download DIR]

Search NASA CMR and NOAA cloud storage for VIIRS satellite granules.

Query NASA CMR for geolocation granules over a bounding box and time range,
then find matching data granules on NOAA mirrors (AWS S3, Google Cloud, Azure).

Usage:
  python noaa_cmr_search.py --bbox -170 51 -130 72 \
      --start 2026-10-01T00:00 --end 2026-10-02T00:00
  python noaa_cmr_search.py --bbox -170 51 -130 72 --start 2026-10-01 \
      --end 2026-10-02 --provider gcs --sat n21 --download /path/to/dir

Requires: requests, s3fs (aws), gcsfs (gcs), or adlfs (azure) for cloud access.
    For refining passes (NASA granules are 6 minutes vs. NOAA's 85 seconds, so some NOAA granules will be outside bounds),
    requires h5py, shapely, antimeridian.

options:
  -h, --help            show this help message and exit
  --products PRODUCT [PRODUCT ...]
                        One or more of: I1, I2, I3, I4, I5, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, DNB (default: ['I4'])
  --bbox W S E N        west south east north, degrees (W > E means crossing 180)
  --start START         UTC, e.g. 2026-10-01T00:00
  --end END             UTC, e.g. 2026-10-02T00:00
  --provider {aws,gcs,azure}
  --sat {n20,n21,snpp} [{n20,n21,snpp} ...]
  --pad PAD             seconds of slack when matching NASA to NOAA granules (default 60)
  --format {url,key}    url = scheme://bucket/key (default), key = bucket/key
  --no-refine           Skip trimming passes to the bbox
  --download DIR        Download passes into DIR/<pass folder>/; omit to just list
```