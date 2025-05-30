# Test the Acolite L2R module
import cProfile
import os
import pstats
import tempfile

import pytest
import xarray as xr
from memory_profiler import profile

import acolite as ac
import json

@pytest.fixture()
def gem():
    test_dir = os.path.dirname(__file__)
    path = os.path.join(test_dir, "data", "L9_OLI_2025_04_29_23_36_04_088080_L1R.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test Gem file {path} does not exist.")
    return str(path)

@pytest.fixture()
def original_dataset_filename():
    test_dir = os.path.dirname(__file__)
    path = os.path.join(test_dir, "data", "L9_OLI_2025_04_29_23_36_04_088080_L2R_main_branch_glint.nc")
    # path = os.path.join(test_dir, "data", "L9_OLI_2025_04_29_23_36_04_088080_L2R_original.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Original dataset file {path} does not exist.")
    return str(path)

@pytest.fixture()
def csiro_settings_ls9():
    test_dir = os.path.dirname(__file__)
    path = os.path.join(test_dir, "data", "csiro_ls9_settings.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSIRO LS9 settings file {path} does not exist.")
    with open(path, "r") as f:
        settings = json.load(f)
    return settings

@profile
def test_acolite_l2r_runs(gem, original_dataset_filename, csiro_settings_ls9):
    """
    Test the Acolite L2R module.
    """
    # Check if the gem file exists
    assert os.path.exists(gem), f"Test Gem file {gem} does not exist."

    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run the Acolite L2R module
        profiler = cProfile.Profile()
        profiler.enable()

        result = ac.acolite.acolite_l2r(gem, output=temp_dir, settings=csiro_settings_ls9)
        # settings = {"dsf_write_aot_550": True}
        # result = ac.acolite.acolite_l2r(gem, output=temp_dir, settings=settings) # Use default settings only

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.dump_stats(f'{temp_dir}/profiler_stats_file.dat')
        stats.strip_dirs()
        stats.print_stats(5).sort_stats('tottime')

        # Check if the result is as expected
        assert result is not None, "Acolite L2R module did not return a result."
        assert os.path.exists(result[0]), f"Output file {result[0]} does not exist."

        result_dataset = xr.open_dataset(result[0])
        original_dataset = xr.open_dataset(original_dataset_filename)

        # Remove attribute that differ between runs
        result_noatts = result_dataset.drop_attrs(deep=True)
        original_noatts = original_dataset.drop_attrs(deep=True)

        # Check if the datasets are equal
        for k in result_noatts.data_vars:
            print(f"{k}, result: {result_noatts[k].equals(original_noatts[k])}")

        assert result_dataset.equals(original_dataset), "Output dataset does not match the original dataset."
        # assert result_dataset.identical(original_dataset), "Output dataset does not match the original dataset."
