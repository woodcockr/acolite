# Project Guidelines

## Scope And Priority
- This file is the single workspace-wide instruction source for agents in this repository.
- Keep changes narrowly scoped to the user request. Avoid broad refactors across many sensor modules unless explicitly requested.

## Architecture
- ACOLITE-MP is a performance-focused variant of ACOLITE that preserves the staged processing flow.
- Main entrypoint is `launch_acolite.py` for CLI/GUI launch and LUT retrieval.
- Core pipeline orchestration is in `acolite/acolite/acolite_run.py`, with major stages in:
  - `acolite/acolite/acolite_l1r.py`
  - `acolite/acolite/acolite_l2r.py`
  - `acolite/acolite/acolite_l2w.py`
- Sensor-specific logic is organized by mission directories under `acolite/` (for example `landsat/`, `sentinel2/`, `sentinel3/`). Preserve existing per-sensor patterns when adding or changing behavior.
- Global config and settings are initialized in `acolite/__init__.py`, including `ac.settings['defaults']`, `ac.settings['run']`, and `ac.settings['user']`.

## Build And Test
- Preferred local environment in this branch is `uv` with Python 3.12 from `pyproject.toml`:
  - `uv sync --extra test`
- Alternative environment setup is conda via `environment.yml`:
  - `conda env create -f environment.yml`
  - `conda activate acolite`
- Run tests with:
  - `pytest`
  - or targeted tests such as `pytest tests/test_acolite_l2r.py -q`
- Performance tests in `tests/test_perf_*.py` are intentionally skipped by default and should only be enabled when explicitly requested.

## Runtime Commands
- CLI processing:
  - `python launch_acolite.py --cli --settings <settings_file>`
- LUT prefetch (network required):
  - `python launch_acolite.py --retrieve_luts --sensor L8,S2A`

## Project Conventions
- Respect existing numerical tolerance expectations in regression tests (especially `tests/test_workflow_outputs.py`).
- Keep dtype/scale handling behavior stable for GeoTIFF and NetCDF outputs unless a task explicitly asks for numerical behavior changes.
- Follow existing style in touched files; do not normalize legacy headers/comments across unrelated modules.
- Use minimal edits and avoid moving modules or renaming public functions unless necessary.

## External Dependencies And Pitfalls
- GDAL compatibility matters for mission data support (notably Sentinel-2 JPEG2000 handling).
- Some workflows require external resources/credentials:
  - EarthData credentials for ancillary/DEM data
  - LUT downloads from `acolite/acolite_luts` on first use
  - Optional libRadtran for TACT workflows (`external/libRadtran-2.0.5` or system install)
- Many tests rely on external fixture data paths (`tests/utils.py`). Validate fixture availability before assuming failures are code regressions.

## Documentation Links
- Project overview and baseline install/dependency notes: `README.md`
- Branch-specific goals, performance, and numerical tolerance context: `README-ACOLITE-MP.md`
- Processing defaults and tunable settings: `config/defaults.txt` and `config/defaults/`