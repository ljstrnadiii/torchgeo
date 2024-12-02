# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utilities for building geopandas-based tile index(es)."""

import datetime as dt
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from shapely import box
from shapely.affinity import translate
from shapely.geometry import Polygon


def _sample_dates(
    start_date: datetime, end_date: datetime, num_samples: int, rng: np.random.Generator
) -> list[datetime]:
    delta = (end_date - start_date).days
    days = rng.choice(list(range(0, delta)), replace=False, size=num_samples)
    return [start_date + timedelta(days=int(day)) for day in days]


def build_fake_sparse_rasters(
    output_dir: str,
    lat_lons: list[tuple[float, float]],
    crs: CRS,
    num_dates: int = 5,
    start_date: datetime = datetime(2023, 1, 1),
    end_date: datetime = datetime(2023, 1, 15),
    resolution_meters: float = 10,
    height: int = 100,
    width: int = 100,
    nbands: int = 1,
    rng: np.random.Generator = np.random.default_rng(),
) -> list[str]:
    """Generate fake sparse rasters for testing purposes.

    Args:
        output_dir: Directory to save the generated rasters.
        lat_lons: List of (lat, lon) tuples to generate rasters for.
        crs: Coordinate reference system for the generated rasters.
        num_dates: Number of dates to generate rasters for.
        start_date: Start date for the generated rasters.
        end_date: End date for the generated rasters.
        resolution_meters: Resolution of the generated rasters in meters.
        height: Height of the generated rasters.
        width: Width of the generated rasters.
        nbands: Number of bands in the generated rasters.
        rng: Random number generator to use for sampling dates.

    Returns:
        List of paths to the generated rasters.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    if crs.is_geographic:
        resolution_x = resolution_y = resolution_meters / 111320
    else:
        resolution_x = resolution_y = resolution_meters

    paths = []
    for origin_idx, (lon, lat) in enumerate(lat_lons):
        x, y = transformer.transform(lon, lat)
        for timestamp in _sample_dates(start_date, end_date, num_dates, rng=rng):
            file_name = f"{origin_idx}_{timestamp.strftime('%Y%m%d')}.tif"
            file_path = os.path.join(output_dir, file_name)
            transform = from_origin(
                west=x, north=y, xsize=resolution_x, ysize=resolution_y
            )
            # Generate random data, but multiply by crs to make it more interesting
            data = rng.random((nbands, height, width)).astype('float32') * crs.to_epsg()
            with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=nbands,
                dtype=data.dtype,
                crs=crs.to_wkt(),
                transform=transform,
            ) as dst:
                for i in range(1, nbands + 1):
                    dst.write(data[i - 1], i)
            paths.append(file_path)
    return paths


@dataclass
class _RasterSpec:
    crs: str
    geometry: Polygon
    location: str
    datetime: dt.datetime | None
    collection: str


def build_raster_index(
    tifs: list[str], date_regex: str, collection: str
) -> gpd.GeoDataFrame:
    """Given tifs and a regex pattern to extract dates, build a tile index.

    Note: This can be used in various ways to manage spatiotemporal tiled data
    1. Since there will be a location col, this is GTI ready
    2. We can groupby (crs, datetime), and aggregate tifs to vrt with rasterio
    3. We can open vrt with rasterio directly or pass to rioxarray.

    Args:
        tifs: A list of tif paths to use for a tile index.
        date_regex: Regex pattern to extract dates from file names.
        collection: Name of the collection e.g. 'Landsat', 'Sentinel', etc.

    Returns:
        A GeoDataFrame with columns for crs, geometry, location, datetime, and
        collection.
    """
    pattern = re.compile(date_regex, re.VERBOSE)

    specs = []
    for tif in tifs:
        with rasterio.open(tif) as dset:
            match = pattern.search(tif)
            t = None
            if match is not None:
                t = dt.datetime.strptime(match.group(1), '%Y%m%d')
            crs = dset.crs.to_string()
            bounds = list(dset.bounds)
            if crs != 'EPSG:4326':
                transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                bounds = transformer.transform_bounds(*bounds)
            specs.append(
                _RasterSpec(
                    crs=crs,
                    geometry=box(*bounds, ccw=True),
                    location=tif,
                    datetime=t,
                    collection=collection,
                )
            )
    return gpd.GeoDataFrame(data=specs, geometry='geometry', crs='EPSG:4326')


def jitter_geometries(
    gdf: gpd.GeoDataFrame, max_offset: float = 0.0001
) -> gpd.GeoDataFrame:
    """Helper to visualize points if tiles perfectly overlap.

    Args:
        gdf: GeoDataFrame with geometries to jitter.
        max_offset: Maximum offset for jittering.

    Returns:
        GeoDataFrame with jittered geometries.
    """
    """Helper to visualize points if tiles perfectly overlap."""
    jittered_geometries = []
    for geom in gdf.geometry:
        if geom.is_empty:
            jittered_geometries.append(geom)
        else:
            dx = np.random.uniform(-max_offset, max_offset)
            dy = np.random.uniform(-max_offset, max_offset)
            jittered_geometries.append(translate(geom, xoff=dx, yoff=dy))

    gdf = gdf.copy()
    gdf['geometry'] = jittered_geometries
    return gdf
