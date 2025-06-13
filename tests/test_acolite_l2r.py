# Test the Acolite L2R module
import cProfile
import json
import os
import tempfile
import time

import numpy as np
import pytest
import xarray as xr
from utils import arrays_almost_equal

# from memory_profiler import profile
import acolite as ac


@pytest.mark.parametrize(
    "test_input",
    [
        pytest.param(
            {
                "key": "landsat",
                "gem": "landsat9_original/L9_OLI_2025_04_29_23_36_04_088080_L1R_original.nc",
                "original_dataset_filename": "landsat9_original/L9_OLI_2025_04_29_23_36_04_088080_L2R_original.nc",
                "settings": "landsat9_original/csiro_settings"
            },
            id="landsat"
        ),
        pytest.param(
            {
                "key": "sentinel2",
                "gem": "s2_original/S2A_MSI_2025_04_15_00_04_50_T56JMP_L1R.nc",
                "original_dataset_filename": "s2_original/S2A_MSI_2025_04_15_00_04_50_T56JMP_L2R.nc",
                "settings": "s2_original/csiro_settings"
            },
            id="sentinel2"
        )
    ]
)

# @profile
def test_acolite_l2r(test_input):
    """
    Test the Acolite L2R module.
    """
    # Check if the gem file exists

    gem = test_input['gem']
    original_dataset_filename = test_input['original_dataset_filename']
    settings_file = test_input['settings']
    # Check if the original dataset file exists
    test_dir = os.path.dirname(__file__)
    gem = os.path.join(test_dir, "data", gem)
    original_dataset_filename = os.path.join(test_dir, "data", original_dataset_filename)
    settings_file = os.path.join(test_dir, "data", settings_file + ".json")
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
        # Run the Acolite L2R module
        # profiler = cProfile.Profile()
        # profiler.enable()
        start_time = time.time()

        result = ac.acolite.acolite_l2r(gem, output=temp_dir, settings=csiro_settings_ls9, return_gem=False) # return_gem=False for test - in full run is csiro_settings_ls9['l2r_return_gem'])

        elapsed_time = time.time() - start_time
        print(f"execution time: {elapsed_time:.4f} seconds")

        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.dump_stats(f'{temp_dir}/profiler_stats_file.dat')
        # stats.strip_dirs()
        # stats.print_stats(5).sort_stats('tottime')

        # Check if the result is as expected
        assert result is not None, "Acolite L2R module did not return a result."
        assert os.path.exists(result[0]), f"Output file {result[0]} does not exist."

        result_dataset = xr.open_dataset(result[0])
        original_dataset = xr.open_dataset(original_dataset_filename)

        # Remove attribute that differ between runs
        result_noatts = result_dataset.drop_attrs(deep=True)
        original_noatts = original_dataset.drop_attrs(deep=True)

        # Check if the datasets are equal
        for k in original_noatts.data_vars:
            assert arrays_almost_equal(result_noatts[k], original_noatts[k]), f"Arrays differ for variable {k}!"
