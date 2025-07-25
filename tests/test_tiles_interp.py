import os
import time

import numpy as np
import pytest
from utils import acolite_fixtures_path

from acolite.shared.tiles_interp import tiles_interp


def test_tiles_interp_regression():
    test_dir = acolite_fixtures_path
    for name in ["aot_550"]:
        path = os.path.join(test_dir, "tiles_interp", name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test data directory {path} does not exist.")
        arrays = {}
        for key in ["data", "xnew", "ynew", "znew"]:
            arrays[key] = np.load(os.path.join(path, f"{key}.npy"))

        data = arrays["data"]
        xnew = arrays["xnew"]
        ynew = arrays["ynew"]
        original_znew = arrays["znew"]

        # create a random data array of the same shape as the original data
        for i in range(5):
            random_data = np.random.rand(*data.shape)

            # Interpolate without smoothing or masking
            start_time = time.time()
            znew_pyinterp = tiles_interp(
                random_data,
                xnew,
                ynew,
                smooth=True,
                kern_size=3,
                method="linear",
                mask=None,
                target_mask=None,
                target_mask_full=False,
                fill_nan=True,
                dtype="float32",
                use_rgi=False,
                interpolator="pyinterp",
                num_threads=8
            )
            elapsed_time = time.time() - start_time
            print(f"random_data: pyinterp execution time: {elapsed_time:.4f} seconds")
            start_time = time.time()
            znew_interpn = tiles_interp(
                random_data,
                xnew,
                ynew,
                smooth=True,
                kern_size=3,
                method="linear",
                mask=None,
                target_mask=None,
                target_mask_full=False,
                fill_nan=True,
                dtype="float32",
                use_rgi=False,
                interpolator="interpn",
                num_threads=0
            )
            elapsed_time = time.time() - start_time
            print(f"random_data: interpn execution time: {elapsed_time:.4f} seconds")

            assert znew_pyinterp.shape == znew_interpn.shape, (
                f"{name}: Expected shape {znew_interpn.shape}, got {znew_pyinterp.shape}"
            )
            # Check if both methods produce the same output
            assert np.allclose(
                znew_pyinterp, znew_interpn, rtol=1e-5, atol=1e-8, equal_nan=True
            ), f"{name}: Pyinterp and legacy interpolation results do not match."
        # Check against the original znew
        start_time = time.time()
        znew_pyinterp = tiles_interp(
            data,
            xnew,
            ynew,
            smooth=True,
            kern_size=3,
            method="linear",
            mask=None,
            target_mask=None,
            target_mask_full=False,
            fill_nan=True,
            dtype="float32",
            use_rgi=False,
            interpolator="pyinterp",
        )
        elapsed_time = time.time() - start_time
        print(f"{name}: pyinterp execution time: {elapsed_time:.4f} seconds")
        assert np.allclose(
            znew_pyinterp, original_znew, rtol=1e-5, atol=1e-7, equal_nan=True
        ), f"{name}: Pyinterp and original_znew interpolation results do not match."
