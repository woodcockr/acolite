# Test the Acolite L2W module
import cProfile
import json
import os
import pstats
import tempfile
import time

import numpy as np
import pytest
import xarray as xr
from memory_profiler import profile
from utils import acolite_fixtures_path, arrays_almost_equal

import acolite as ac


# For the l2w tests the L2R output from acolite-mp is used and the resulting l2w compared with the original acolite l2w output.
@pytest.mark.parametrize(
    "test_input",
    [
        pytest.param(
            {
                "key": "landsat",
                "gem": "landsat9_acolite_mp/L9_OLI_2025_04_29_23_36_04_088080_L2R_acolite_mp.nc",
                "original_dataset_filename": "landsat9_original/L9_OLI_2025_04_29_23_36_04_088080_L2W_original.nc",
                "settings": "landsat9_original/csiro_settings"
            },
            id="landsat"
        ),
        pytest.param(
            {
                "key": "sentinel2",
                "gem": "sentinel2_acolite_mp/S2A_MSI_2025_04_15_00_04_50_T56JMP_L2R_acolite_mp.nc",
                "original_dataset_filename": "s2_original/S2A_MSI_2025_04_15_00_04_50_T56JMP_L2W.nc",
                "settings": "s2_original/csiro_settings"
            },
            id="sentinel2"
        )
    ]
)


# @profile
def test_acolite_l2w(test_input):
    """
    Test the Acolite L2W module.
    """
    gem = test_input["gem"]
    original_dataset_filename = test_input["original_dataset_filename"]
    settings_file = test_input["settings"]
    # Check if the original dataset file exists
    test_dir = acolite_fixtures_path
    gem = os.path.join(test_dir, gem)
    original_dataset_filename = os.path.join(test_dir, original_dataset_filename)
    settings_file = os.path.join(test_dir, settings_file + ".json")
    assert os.path.exists(gem), f"Test Gem file {gem} does not exist."
    assert os.path.exists(original_dataset_filename), f"Original dataset file {original_dataset_filename} does not exist."
    assert os.path.exists(settings_file), f"Settings file {settings_file} does not exist."

    # Load the settings from the JSON file
    with open(settings_file, "r") as f:
        csiro_settings_ls9 = json.load(f)
    # Check if the settings are loaded correctly
    assert csiro_settings_ls9 is not None, "Settings could not be loaded from the JSON file."

    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run the Acolite L2W module
        # profiler = cProfile.Profile()
        # profiler.enable()
        start_time = time.time()

        result = ac.acolite.acolite_l2w(gem, target_file=f'{temp_dir}/l2w_output.nc', settings=csiro_settings_ls9)

        elapsed_time = time.time() - start_time
        print(f"execution time: {elapsed_time:.4f} seconds")

        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.dump_stats(f'{temp_dir}/profiler_stats_file.dat')
        # stats.strip_dirs()
        # stats.print_stats(5).sort_stats('tottime')

        # Check if the result is as expected
        assert result is not None, "Acolite L2W module did not return a result."
        assert os.path.exists(result), f"Output file {result} does not exist."

        result_dataset = xr.open_dataset(result)
        original_dataset = xr.open_dataset(original_dataset_filename)

        # Remove attribute that differ between runs
        result_noatts = result_dataset.drop_attrs(deep=True)
        original_noatts = original_dataset.drop_attrs(deep=True)

        # Check if the datasets are equal
        for k in original_noatts.data_vars:
            if k not in ['l2_flags']: # Skip l2_flags as the array comparison is not valid
                assert arrays_almost_equal(result_noatts[k], original_noatts[k]), f"Arrays differ for variable {k}!"
            else:
                # For l2_flags, we can check if the arrays are almost equal with a count of differences
                fraction = 0.0001  # Allow up to 0.01% differences
                num_diff = np.count_nonzero(result_noatts[k] != original_noatts[k])
                are_almost_equal = num_diff / result_noatts[k].size <= fraction
                assert are_almost_equal, f"Arrays differ for variable {k}!"

        # WIP This may need to be replaced with allclose in future
        # assert result_dataset.equals(original_dataset), "Output dataset does not equal the original dataset."
