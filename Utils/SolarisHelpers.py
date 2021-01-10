from shapely.wkt import loads
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
import pandas as pd
import geopandas as gpd
import pyproj
from fiona._err import CPLE_OpenFailedError
from fiona.errors import DriverError
from warnings import warn
from distutils.version import LooseVersion
import skimage
import numpy as np
import rasterio
from rasterio import features
from affine import Affine


def footprint_mask(df, out_file=None, reference_im=None, geom_col='geometry',
                   do_transform=None, affine_obj=None, shape=(900, 900),
                   out_type='int', burn_value=255, burn_field=None):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to ``None``, in which case the function
        attempts to infer whether or not a transformation is required based on
        the presence or absence of a CRS in `df`. If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    out_type : 'float' or 'int'
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided.
    burn_field : str, optional
        Name of a column in `df` that provides values for `burn_value` for each
        independent object. If provided, `burn_value` is ignored.

    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`.

    """
    # start with required checks and pre-population of values
    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')
    df = _check_df_load(df)

    if len(df) == 0 and not out_file:
        return np.zeros(shape=shape, dtype='uint8')

    if do_transform is None:
        # determine whether or not transform should be done
        do_transform = _check_do_transform(df, reference_im, affine_obj)

    df[geom_col] = df[geom_col].apply(_check_geom)  # load in geoms if wkt
    if not do_transform:
        affine_obj = Affine(1, 0, 0, 0, 1, 0)  # identity transform

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
        shape = reference_im.shape
        if do_transform:
            affine_obj = reference_im.transform

    # extract geometries and pair them with burn values
    if burn_field:
        if out_type == 'int':
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('uint8')))
        else:
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('float32')))
    else:
        feature_list = list(zip(df[geom_col], [burn_value] * len(df)))

    if len(df) > 0:
        output_arr = features.rasterize(shapes=feature_list, out_shape=shape,
                                        transform=affine_obj)
    else:
        output_arr = np.zeros(shape=shape, dtype='uint8')
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        if out_type == 'int':
            meta.update(dtype='uint8')
            meta.update(nodata=0)
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr


def _check_do_transform(df, reference_im, affine_obj):
    """Check whether or not a transformation should be performed."""
    try:
        crs = getattr(df, 'crs')
    except AttributeError:
        return False  # if it doesn't have a CRS attribute

    if not crs:
        return False  # return False for do_transform if crs is falsey
    elif crs and (reference_im is not None or affine_obj is not None):
        # if the input has a CRS and another obj was provided for xforming
        return True


def _check_df_load(df):
    """Check if `df` is already loaded in, if not, load from file."""
    if isinstance(df, str):
        if df.lower().endswith('json'):
            return _check_gdf_load(df)
        else:
            return pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise ValueError(f"{df} is not an accepted DataFrame format.")


def _check_geom(geom):
    """Check if a geometry is loaded in.

    Returns the geometry if it's a shapely geometry object. If it's a wkt
    string or a list of coordinates, convert to a shapely geometry.
    """
    if isinstance(geom, BaseGeometry):
        return geom
    elif isinstance(geom, str):  # assume it's a wkt
        return loads(geom)
    elif isinstance(geom, list) and len(geom) == 2:  # coordinates
        return Point(geom)


def _check_crs(input_crs, return_rasterio=False):
    """Convert CRS to the ``pyproj.CRS`` object passed by ``solaris``."""
    if not isinstance(input_crs, pyproj.CRS) and input_crs is not None:
        out_crs = pyproj.CRS(input_crs)
    else:
        out_crs = input_crs

    if return_rasterio:
        if LooseVersion(rasterio.__gdal_version__) >= LooseVersion("3.0.0"):
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt())
        else:
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt("WKT1_GDAL"))

    return out_crs

def _check_skimage_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return skimage.io.imread(im)
    elif isinstance(im, np.ndarray):
        return im
    else:
        raise ValueError(
            "{} is not an accepted image format for scikit-image.".format(im))

def _check_rasterio_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return rasterio.open(im)
    elif isinstance(im, rasterio.DatasetReader):
        return im
    else:
        raise ValueError(
            "{} is not an accepted image format for rasterio.".format(im))

def _check_gdf_load(gdf):
    """Check if `gdf` is already loaded in, if not, load from geojson."""
    if isinstance(gdf, str):
        # as of geopandas 0.6.2, using the OGR CSV driver requires some add'nal
        # kwargs to create a valid geodataframe with a geometry column. see
        # https://github.com/geopandas/geopandas/issues/1234
        if gdf.lower().endswith('csv'):
            return gpd.read_file(gdf, GEOM_POSSIBLE_NAMES="geometry",
                                 KEEP_GEOM_COLUMNS="NO")
        try:
            return gpd.read_file(gdf)
        except (DriverError, CPLE_OpenFailedError):
            warn(f"GeoDataFrame couldn't be loaded: either {gdf} isn't a valid"
                 " path or it isn't a valid vector file. Returning an empty"
                 " GeoDataFrame.")
            return gpd.GeoDataFrame()
    elif isinstance(gdf, gpd.GeoDataFrame):
        return gdf
    else:
        raise ValueError(f"{gdf} is not an accepted GeoDataFrame format.")
