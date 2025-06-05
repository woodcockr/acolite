# Test the Acolite landsat l1r module
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
def input_filename():
    test_dir = os.path.dirname(__file__)
    path = os.path.join(test_dir, "data", "ls9_usgs_original")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input dataset {path} does not exist.")
    return str(path)

@pytest.fixture()
def original_dataset_filename():
    test_dir = os.path.dirname(__file__)
    path = os.path.join(test_dir, "data", "L9_OLI_2025_04_29_23_36_04_088080_L1R_original.nc")
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
def test_acolite_landsat_l1r_runs(input_filename, original_dataset_filename, csiro_settings_ls9):
    """
    Test the Acolite Landsat L1R module.
    """
    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run the Acolite L1R module
        profiler = cProfile.Profile()
        profiler.enable()

        result, _ = ac.landsat.l1_convert(input_filename, output=f'{temp_dir}', settings=csiro_settings_ls9)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.dump_stats(f'{temp_dir}/profiler_stats_file.dat')
        stats.strip_dirs()
        stats.print_stats(5).sort_stats('tottime')

        # Check if the result is as expected
        assert result is not None, "Acolite Landsat L1R module did not return a result."
        assert os.path.exists(result[0]), f"Output file {result[0]} does not exist."

        result_dataset = xr.open_dataset(result[0])
        original_dataset = xr.open_dataset(original_dataset_filename)

        # Remove attribute that differ between runs
        result_noatts = result_dataset.drop_attrs(deep=True)
        original_noatts = original_dataset.drop_attrs(deep=True)

        # Check if the datasets are equal
        for k in original_noatts.data_vars:
            print(f"{k}, result: {result_noatts[k].equals(original_noatts[k])}")

        assert result_dataset.equals(original_dataset), "Output dataset does not equal the original dataset."
