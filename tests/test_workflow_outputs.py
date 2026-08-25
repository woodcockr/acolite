import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pytest
import rasterio
from utils import acolite_fixtures_path


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
@pytest.mark.skip(reason="This test is temporarily disabled.")
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
    tests_passed = True
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(geotiffs_are_close, input_tif, fixture_path / input_tif.name): input_tif for input_tif in input_tifs}
        for future in as_completed(futures):
            input_tif = futures[future]
            try:
                result = future.result()
                tests_passed = result and tests_passed
                if not result:
                    logging.warning(f"TIF files {input_tif.name} are not close enough.")
            except Exception as e:
                pytest.fail(f"Error comparing {input_tif}: {e}")
    assert tests_passed, "TIF files are not close enough."

def apply_scales_and_offsets(src):
    """
    Apply scales and offsets to a rasterio dataset.
    """
    scale = src.scales[0] if src.scales else 1.0
    offset = src.offsets[0] if src.offsets else 0.0
    data = src.read(1).astype('float32')
    data[data == src.nodata] = np.nan
    data = data * scale + offset
    return data

def geotiffs_are_close(file1, file2):
    """
    Compare two GeoTIFF files pixel-by-pixel.
    Returns True if all corresponding pixels are close within the given tolerances.
    """
    arrays_close = False
    with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
        if src1.count != src2.count or src1.width != src2.width or src1.height != src2.height:
            return False

        if src1.dtypes[0] == 'uint8' and src2.dtypes[0] == 'uint8': # l2_flags
                arr1 = src1.read(1)
                arr2 = src2.read(1)
                # For l2_flags, we can check if the arrays are almost equal with a count of differences
                fraction = 0.0001  # Allow up to 0.01% differences
                num_diff = np.count_nonzero(arr1 != arr2)
                are_almost_equal = num_diff / arr1.size <= fraction
                if are_almost_equal:
                    arrays_close = True
                else:
                    print_differences(arr1, arr2)

        elif src1.dtypes[0] == 'uint16' and src2.dtypes[0] == 'uint16':
            data1 = apply_scales_and_offsets(src1)
            data2 = apply_scales_and_offsets(src2)
            if np.allclose(data1, data2, rtol=1e-5, atol=1.1e-4, equal_nan=True):  # 1.1e-4 atol because of the round() call when converting to uint16
                arrays_close = True

        elif src1.dtypes[0] == 'float32' and src2.dtypes[0] == 'float32':
            arr1 = src1.read(1)
            arr2 = src2.read(1)
            if np.allclose(arr1, arr2, rtol=1e-5, atol=1e-8, equal_nan=True):
                arrays_close = True

        else:
            logging.warning(f"Unsupported data type in band {file1.name}: {src1.dtype[0]} and {file2.name}: {src2.dtype[0]}.")

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