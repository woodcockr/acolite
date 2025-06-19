# Test performance of the Acolite L2R module
import json
import os
import tempfile
import time

import pytest
from utils import arrays_almost_equal, acolite_fixtures_path

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
@pytest.mark.skip(reason="This test is temporarily disabled.")
# @profile
def test_acolite_l2r(test_input):
    """
    Test the Acolite L2R module.
    """
    key = test_input["key"]
    # Check if the gem file exists
    gem = test_input['gem']
    original_dataset_filename = test_input['original_dataset_filename']
    settings_file = test_input['settings']
    # Check if the original dataset file exists
    test_dir = acolite_fixtures_path
    assert os.path.exists(test_dir), f"Fixtures repository {test_dir} does not exist."
    gem = os.path.join(test_dir,  gem)
    original_dataset_filename = os.path.join(test_dir,  original_dataset_filename)
    settings_file = os.path.join(test_dir,  settings_file + ".json")
    assert os.path.exists(gem), f"Test Gem file {gem} does not exist."
    assert os.path.exists(original_dataset_filename), f"Original dataset file {original_dataset_filename} does not exist."
    assert os.path.exists(settings_file), f"Settings file {settings_file} does not exist."

    # Load the settings from the JSON file
    with open(settings_file, "r") as f:
        csiro_settings_ls9 = json.load(f)
    # Check if the settings are loaded correctly
    assert csiro_settings_ls9 is not None, "Settings could not be loaded from the JSON file."

    # Check on performance depending on max_workers setting
    metrics = []
    # Loop through different max_workers settings
    # This is to test the performance of the acolite-mp L1R module with different max_workers settings
    for workers in range(1, 16):
        print(f"Using max_workers: {workers}")
        csiro_settings_ls9["acolite-mp_acolite_l2r_max_workers"] = workers
        avg_elapsed_time = 0.0
        for run in range(1, 4):  # Run each test multiple times for averaging
            print(f"Run {run} for {key} with max_workers={workers}")
            # Create a temporary directory for the output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run the Acolite L2R module
                # profiler = cProfile.Profile()
                # profiler.enable()
                start_time = time.time()

                _ = ac.acolite.acolite_l2r(gem, output=temp_dir, settings=csiro_settings_ls9, return_gem=False) # return_gem=False for test - in full run is csiro_settings_ls9['l2r_return_gem'])

                elapsed_time = time.time() - start_time
                print(f"execution time: {elapsed_time:.4f} seconds")

                avg_elapsed_time += elapsed_time
        # Average the elapsed time over the number of runs
        avg_elapsed_time /= 3
        metrics.append((workers, avg_elapsed_time))

    # Sort metrics by elapsed time
    metrics.sort(key=lambda x: x[1])
    for workers, elapsed_time in metrics:
        print(f"Metrics for {key} with max_workers={workers}: {elapsed_time:.4f} seconds")

    # save the metrics to a file
    metrics_filename = os.path.join(os.path.dirname(__file__), f"{key}_l2r_metrics.csv")
    with open(metrics_filename, "w") as f:
        f.write("max_workers,elapsed_time\n")
        for workers, elapsed_time in metrics:
            f.write(f"{workers},{elapsed_time:.4f}\n")