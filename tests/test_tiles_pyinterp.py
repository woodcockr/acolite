import numpy as np
import os
import pytest
from acolite.shared.tiles_pyinterp import tiles_interp as tiles_pyinterp
from acolite.shared.tiles_interp import tiles_interp as tiles_interp_legacy
import time

def test_tiles_interp_regression():
    test_dir = os.path.dirname(__file__)
    for name in ["aot_550"]: # WIP Fixture for romix is broken , "romix"]:
        path = os.path.join(test_dir, "data/tiles_interp", name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test data directory {path} does not exist.")
        arrays = {}
        for key in ["data", "xnew", "ynew", "znew"]:
            arrays[key] = np.load(os.path.join(path, f"{key}.npy"))

        data = arrays["data"]
        xnew = arrays["xnew"]
        ynew = arrays["ynew"]
        original_znew = arrays["znew"]

        #create a random data array of the same shape as the original data
        for i in range(5):
            random_data = np.random.rand(*data.shape)

            # Interpolate without smoothing or masking
            start_time = time.time()
            znew_pyinterp = tiles_pyinterp(
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
            )
            elapsed_time = time.time() - start_time
            print(f"random_data: tiles_pyinterp execution time: {elapsed_time:.4f} seconds")
            start_time = time.time()
            znew_legacy = tiles_interp_legacy(
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
            )
            elapsed_time = time.time() - start_time
            print(f"random_data: tiles_interp_legacy execution time: {elapsed_time:.4f} seconds")

            assert znew_pyinterp.shape == znew_legacy.shape, f"{name}: Expected shape {znew_legacy.shape}, got {znew_pyinterp.shape}"
            # Check if both methods produce the same output
            assert np.allclose(znew_pyinterp, znew_legacy, rtol=1e-5, atol=1e-8, equal_nan=True), f"{name}: Pyinterp and legacy interpolation results do not match."
        # Check against the original znew
        start_time = time.time()
        znew_pyinterp = tiles_pyinterp(
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
        )
        elapsed_time = time.time() - start_time
        print(f"{name}: tiles_pyinterp execution time: {elapsed_time:.4f} seconds")
        assert np.allclose(znew_pyinterp, original_znew, rtol=1e-5, atol=1e-7, equal_nan=True), f"{name}: Pyinterp and original_znew interpolation results do not match."
