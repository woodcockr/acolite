## def tiles_pyinterp
## interpolates tiled dataset to full scene extent using pyinterp
import numpy as np
import xarray as xr
import pyinterp
import pyinterp.backends.xarray
from scipy.ndimage import uniform_filter
import acolite as ac

def tiles_interp(data, xnew, ynew, smooth=False, kern_size=2, method='nearest', mask=None,
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
        raise NotImplementedError("Nearest neighbor interpolation is not yet implemented in pyinterp.")

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
        })

    # if target_mask is None:
    #     znew = znew.reshape(mx.shape)
    # else:
    #     # WIP implement target mask handling
    #     raise NotImplementedError("Target mask is not yet implemented for pyinterp interpolation")
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