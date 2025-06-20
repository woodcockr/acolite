# Performance Test the Acolite landsat l1r module
import json
import os
import tempfile
import time

import pytest
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
        )
    ]
)
@pytest.mark.skip(reason="This test is temporarily disabled.")
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

    # Check on performance depending on max_workers setting
    metrics = []
    # Loop through different max_workers settings
    # This is to test the performance of the acolite-mp L1R module with different max_workers settings
    for workers in range(1, 16):
        print(f"Using max_workers: {workers}")
        settings["acolite-mp_l1_convert_max_workers"] = workers
        avg_elapsed_time = 0.0
        for run in range(1, 4):  # Run each test multiple times for averaging
            print(f"Run {run} for {key} with max_workers={workers}")
            # Create a temporary directory for the output
            with tempfile.TemporaryDirectory() as temp_dir:
                start_time = time.time()

                match key:
                    case "landsat":
                        result, _ = ac.landsat.l1_convert(input_filename, output=f'{temp_dir}', settings=settings)
                    case "sentinel2":
                        result, _ = ac.sentinel2.l1_convert(input_filename, output=f'{temp_dir}', settings=settings)
                    case _:
                        raise ValueError(f"Unknown key {key} in fixtures.")
                elapsed_time = time.time() - start_time
                print(f"Execution time for {key}: {elapsed_time:.4f} seconds")

                avg_elapsed_time += elapsed_time
        # Average the elapsed time over the number of runs
        avg_elapsed_time /= 3
        metrics.append((workers, avg_elapsed_time))

    # Sort metrics by elapsed time
    metrics.sort(key=lambda x: x[1])
    for workers, elapsed_time in metrics:
        print(f"Metrics for {key} with max_workers={workers}: {elapsed_time:.4f} seconds")

    # save the metrics to a file
    metrics_filename = os.path.join(os.path.dirname(__file__), f"{key}_l1r_metrics.csv")
    with open(metrics_filename, "w") as f:
        f.write("max_workers,elapsed_time\n")
        for workers, elapsed_time in metrics:
            f.write(f"{workers},{elapsed_time:.4f}\n")
