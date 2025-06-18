import cProfile
import json
import os
import pstats
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from memory_profiler import profile
from utils import acolite_fixtures_path, arrays_almost_equal


@pytest.mark.parametrize(
    "outputs",
    [
        pytest.param(
            {
                "key": "landsat9",
                "input_path": "/data/acolite/test/acolite-mp/landsat9_workflow_output/",
            },
            id="landsat9"
        ),
        pytest.param(
            {
                "key": "sentinel2a",
                "input_path": "/data/acolite/test/acolite-mp/sentinel2a_workflow_output/",
            },
            id="sentinel2a"
        )
    ]
)
def test_workflow_outputs(outputs):
    # Path to the directories containing the GeoTIFF files
    fixture_path = Path(acolite_fixtures_path) / "workflow-output" / f"{outputs['key']}_workflow_output"

    # Get the input path from the test input
    input_path = Path(outputs["input_path"])
    # Check if the directories contain the same files
    assert input_path.exists(), f"Input path {input_path} does not exist."
    assert fixture_path.exists(), f"Fixture path {fixture_path} does not exist."
    input_tifs = list(input_path.glob("*.tif"))
    fixture_tifs = list(fixture_path.glob("*.tif"))
    assert len(input_tifs) == len(fixture_tifs), "Input and fixture directories do not contain the same number of tif files."
    # Check if the tif files in the input path match those in the fixture path
    input_tif_names = [f.name for f in input_tifs]
    fixture_tif_names = [f.name for f in fixture_tifs]
    assert set(input_tif_names) == set(fixture_tif_names), "Input and fixture directories do not contain the same tif files."
    # Compare each tif file in the input path with the corresponding tif file in the fixture path
    for input_tif in input_tifs:
        fixture_tif = fixture_path / input_tif.name
        # Compare the two tif files pixel-by-pixel
        assert geotiffs_are_close(input_tif, fixture_tif), f"TIF files {input_tif} are not close enough."


    # result = geotiffs_are_close(fixture_path / "file1.tif", fixture_path / "file2.tif")

def geotiffs_are_close(file1, file2, atol=1e-5, rtol=1e-8):
    """
    Compare two GeoTIFF files pixel-by-pixel.
    Returns True if all corresponding pixels are close within the given tolerances.
    """
    with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
        if src1.count != src2.count or src1.width != src2.width or src1.height != src2.height:
            return False
        arrays_close = True
        for b in range(1, src1.count + 1):
            arr1 = src1.read(b)
            arr2 = src2.read(b)
            if arr1.dtype == 'uint8' and arr2.dtype == 'uint8':
                    # For l2_flags, we can check if the arrays are almost equal with a count of differences
                    fraction = 0.0001  # Allow up to 0.01% differences
                    num_diff = np.count_nonzero(arr1 != arr2)
                    are_almost_equal = num_diff / arr1.size <= fraction
                    if not are_almost_equal:
                        arrays_close = False
                        print(f"Arrays differ in band {b} of {file1.name}.")
                        print_differences(arr1, arr2)
            elif arr1.dtype == 'float32' and arr2.dtype == 'float32':
                if not np.allclose(arr1, arr2, rtol=1e-5, atol=1e-8, equal_nan=True):
                    arrays_close = False
                    print(f"Arrays differ in band {b} of {file1.name}.")

    return arrays_close

def print_differences(arr1: np.ndarray, arr2: np.ndarray) -> None:
    """
    Compare two integer numpy arrays and print the values of elements that differ.

    Args:
        arr1: First integer numpy array.
        arr2: Second integer numpy array.
    """
    if arr1.shape != arr2.shape:
        print("Arrays have different shapes.")
        return

    diff_indices = np.where(arr1 != arr2)
    for idx in zip(*diff_indices):
        print(f"Difference at index {idx}: arr1={arr1[idx]}, arr2={arr2[idx]}")