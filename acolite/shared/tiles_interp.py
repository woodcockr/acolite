## def tiles_interp
## interpolates tiled dataset to full scene extent
## written by Quinten Vanhellemont, RBINS for the PONDER project
## 2017-12-11
## modifications: 2020-11-17 (QV) added target mask for limiting interpolation extent and processing time
##                2020-11-18 (QV) added dtype to convert from griddata float64, by default float32
##                                this improves peak memory use when several datasets are kept in memory
##                2021-02-11 (QV) added smooth keyword,  default to nearest
##                2024-04-18 (QV) new version using interpn, added option to use RGI
import numpy as np
import pyinterp
import pyinterp.backends.xarray
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, interpn
from scipy.ndimage import uniform_filter

import acolite as ac

def tiles_interp(*args, interpolator='interpn', **kwargs):
    """
    Wrapper function to select interpolation method.

    Parameters:
        interpolator (str or callable): 'interpn', 'pyinterp', or a custom function.
        *args, **kwargs: Arguments passed to the interpolation function.

    Returns:
        Interpolated result from the selected function.
    """
    if callable(interpolator):
        return interpolator(*args, **kwargs)
    elif interpolator == 'interpn':
        return tiles_interpn(*args, **kwargs)
    elif interpolator == 'pyinterp':
        return tiles_pyinterp(*args, **kwargs)
    else:
        raise ValueError(f"Unknown interpolator: {interpolator}")

def tiles_interpn(data, xnew, ynew, smooth = False, kern_size=2, method='nearest', mask = None,
                 target_mask = None, target_mask_full = False, fill_nan = True, dtype = 'float32', use_rgi = False):

    if mask is not None: data[mask] = np.nan

    ## fill nans with closest value
    if fill_nan:
        cur_data = ac.shared.fillnan(data)
    else:
        cur_data = data.copy() # *1.0

    dim = cur_data.shape
    if smooth: cur_data = uniform_filter(cur_data, size = kern_size)

    ## grid positions
    x = np.arange(0., dim[1], 1)
    y = np.arange(0., dim[0], 1)

    ## set up interpolator
    if use_rgi: ## use RGI
        rgi = RegularGridInterpolator((y,x), cur_data, bounds_error = False, fill_value = np.nan, method = method)
        if target_mask is None:
            znew = rgi((ynew[:,None], xnew[None, :]))
        else: ## limit to target mask
            vd = np.where(target_mask)
            if target_mask_full:
                znew = np.zeros((len(ynew), len(xnew))).astype(dtype)+np.nan
                znew[vd] = rgi((ynew[vd[0]], xnew[vd[1]]))
            else:
                znew = rgi((ynew[vd[0]], xnew[vd[1]]))
    else:
        if target_mask is None:
            znew = interpn((y,x), cur_data, (ynew[:,None], xnew[None, :]), method = method, bounds_error = False)
        else: ## limit to target mask
            vd = np.where(target_mask)
            if target_mask_full:
                znew = np.zeros((len(ynew), len(xnew))).astype(dtype)+np.nan
                znew[vd] = interpn((y,x), cur_data, (ynew[vd[0]], xnew[vd[1]]), method = method, bounds_error = False)
            else:
                znew = interpn((y,x), cur_data, (ynew[vd[0]], xnew[vd[1]]), method = method, bounds_error = False)

    ## to convert data type - scipy always returns float64
    if dtype is not None: znew = znew.astype(np.dtype(dtype))
    return(znew)

def tiles_pyinterp(data, xnew, ynew, smooth=False, kern_size=2, method='nearest', mask=None,
                 target_mask=None, target_mask_full=False, fill_nan=True, dtype='float32', use_rgi=False):
    if mask is not None:
        data[mask] = np.nan

    # Fill nans with closest value
    if fill_nan:
        cur_data = ac.shared.fillnan(data)
    else:
        cur_data = data.copy()

    if smooth:
        cur_data = uniform_filter(cur_data, size=kern_size)

    dim = cur_data.shape
    x = np.arange(0., dim[1], 1)
    y = np.arange(0., dim[0], 1)

    # Interpolation method mapping
    if method not in ['nearest', 'linear']:
        raise ValueError("Method must be either 'nearest' or 'linear'.")
    if method == 'nearest':
        interpolator = 'nearest'
    elif method == 'linear':
        interpolator = 'bilinear'
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    # Prepare query points
    if target_mask is not None:
        vd = np.where(target_mask)
        xi = xnew[vd[1]]
        yi = ynew[vd[0]]
    else:
        xi, yi = np.meshgrid(xnew, ynew)
        xi = xi.ravel()
        yi = yi.ravel()

    # Set up xarray DataArray
    da = xr.DataArray(cur_data, coords=[y, x], dims=["y", "x"])
    ## set up interpolator
    grid = pyinterp.backends.xarray.Grid2D(da, geodetic=False)

    # Interpolate
    znew = grid.bivariate(coords={
        'x': xi,
        'y': yi
        }, interpolator=interpolator)

    # Reshape output
    if target_mask is not None and target_mask_full:
        out = np.full((len(ynew), len(xnew)), np.nan, dtype=dtype)
        out[vd] = znew
        znew = out
    elif target_mask is None:
        znew = znew.reshape((len(ynew), len(xnew)))

    if dtype is not None:
        znew = znew.astype(np.dtype(dtype))
    return znew