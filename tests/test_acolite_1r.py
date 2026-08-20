# Test the Acolite landsat l1r module
import json
import os
import tempfile
import time

import pytest
import xarray as xr
from utils import acolite_fixtures_path

import acolite as ac


@pytest.mark.parametrize(
    "test_input",
    [
        pytest.param(
            {
                "key": "landsat",
                "input_path": "landsat9_original/ls9_usgs_original",
                "l1r_filename": "landsat9_original/L9_OLI_2025_04_29_23_36_04_088080_L1R_original.nc",
                "csiro_settings": "landsat9_original/csiro_settings"
            },
            id="landsat"
        ),
        pytest.param(
            {
                "key": "sentinel2",
                "input_path": "s2_original/S2A_MSIL1C_20250415T000311_N0511_R030_T56JMP_20250415T030026.SAFE",
                "l1r_filename": "s2_original/S2A_MSI_2025_04_15_00_04_50_T56JMP_L1R.nc",
                "csiro_settings": "s2_original/csiro_settings"
            },
            id="sentinel2"
        ),
        pytest.param(
            {
                "key": "sentinel3",
                "input_path": "s3_original/S3A_OL_1_EFR____20260104T235356_20260104T235656_20260106T003158_0179_134_301_3600_PS1_O_NT_004.SEN3",
                "l1r_filename": "s3_original/S3A_OLCI_2026_01_04_23_53_55_FR_L1R.nc",
                "csiro_settings": "s3_original/csiro_settings"
            },
            id="sentinel3"
        )
    ]
)


# @profile
def test_acolite_l1r(test_input):
    """
    Test the Acolite Landsat L1R module.
    """
    # Load the input file and original dataset filename from fixtures
    key = test_input["key"]
    input_filename = test_input["input_path"]
    original_dataset_filename = test_input["l1r_filename"]
    csiro_settings = test_input["csiro_settings"]
    # Check if the input file exists
    test_dir = acolite_fixtures_path
    input_filename = os.path.join(test_dir, input_filename)
    original_dataset_filename = os.path.join(test_dir, original_dataset_filename)
    csiro_settings = os.path.join(test_dir, csiro_settings + ".json")
    assert os.path.exists(input_filename), f"Input file {input_filename} does not exist."
    assert os.path.exists(original_dataset_filename), f"Original dataset file {original_dataset_filename} does not exist."
    assert os.path.exists(csiro_settings), f"CSIRO settings file {csiro_settings} does not exist."

    with open(csiro_settings, "r") as f:
        settings = json.load(f)

    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run the Acolite L1R module
        # profiler = cProfile.Profile()
        # profiler.enable()
        start_time = time.time()
        match key:
            case "landsat":
                result, _ = ac.landsat.l1_convert(input_filename, output=f'{temp_dir}', settings=settings)
            case "sentinel2":
                result, _ = ac.sentinel2.l1_convert(input_filename, output=f'{temp_dir}', settings=settings)
            case "sentinel3":
                result, _ = ac.sentinel3.l1_convert(input_filename, output=f'{temp_dir}', settings=settings)
            case _:
                raise ValueError(f"Unknown key {key} in fixtures.")
        elapsed_time = time.time() - start_time
        print(f"Execution time for {key}: {elapsed_time:.4f} seconds")
        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.dump_stats(f'{temp_dir}/profiler_stats_file.dat')
        # stats.strip_dirs()
        # stats.print_stats(5).sort_stats('tottime')

        # Check if the result is as expected
        assert result is not None, "Acolite L1R module did not return a result."
        assert os.path.exists(result[0]), f"Output file {result[0]} does not exist."

        result_dataset = xr.open_dataset(result[0])
        original_dataset = xr.open_dataset(original_dataset_filename)

        # Remove attribute that differ between runs
        result_noatts = result_dataset.drop_attrs(deep=True)
        original_noatts = original_dataset.drop_attrs(deep=True)

        # Check if the datasets are equal
        # ! May need to replace this with allclose for numerical precision issues in the event of library version changes per other tests
        for k in original_noatts.data_vars:
            print(f"{k}, result: {result_noatts[k].equals(original_noatts[k])}")

        assert result_dataset.equals(original_dataset), "Output dataset does not equal the original dataset."