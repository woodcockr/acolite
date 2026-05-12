# ACOLITE-MP

The `acolite-mp` branch is a refactored version of the standard `acolite` code intended to provide higher performance with a core-memory ratio to support more cost effective processing of large collections on Cloud resources. Cloud resources typically have 2, 4, or 8 GiB per core ratios, and many-core processers. The current implementation of `acolite-mp` makes minimal changes to data structure and IO and is thus limited to per-band parallelism. Additionally, the existing multi-stage processing of `acolite` has been preserved. To maintain the memory budget per-band parallelism operates with different parallelism counts with each stage, with higher spatial resolution (e.g., Sentinel 2) having less parallelism due to the memory constraint.

Additionally, `acolite-mp` implements a only a subset of `acolite` options - specially those that match my organisations particular use-case - (LS8,9, S2 - DSG+GC). Additionally some IO has been disabled (the current `gem` object causes issues with parallelism). the `numpy` based interpolation routine has also been replaced with the multithreaded `pyinterp` which is substantially faster and uses less memory.

**The code is at best Beta and needs further verification**.

Performance - For a full workflow including download of source dataa (variability in download speed expected), running on a PC with 1 Gbps download speed and 24 cores i9 13900K processor:

|  |  Acolite-mp  |  Acolite  |
|--|--------------|-----------|
| Sentinel2 | 197s | 791s |
| Landsat | 99s | 312s |


`acolite-mp`  also uses less memory per band but given the parallelism this isn't obvious as multiple bands are being processed at the same time.

Output result variation for COGS uint64 packed values (our workflow replaces the COGS export from `acolite`) is < 1.1e-04 per pixel for uint64 packed values (which are np.round(4) during COG creation), < 1e-08 for float32 values, 0.01% for bit flags. All with in tolerances given the use of float32 single precision floating point in the original and mp codes. The variation is mostly caused by the use of the new multi-threaded interpolator along with different (newer) versions of some libraries. `acolite` original running in the old and new python environments shows a similar variation in rounding errors from time to time to give you an idea of where in the numerical noise these levels are.

For the regression testing acolite `original` and `acolite-mp` were both processing in the same python and compute environment.

A more complete version of this development may be undertaken given the significant cost savings to be obtained on very large collections. Further performance improvements are possible but require more extensive changes to the file handling. There is also a fair amount of GIL contention which limits threading being caused by some structural choices in the implementation which could be removed.

## Developer environment (internal)

The main development workflow for this branch is a VS Code Dev Container.

### Dev container wiring

- Service: `acolite-mp-dev`
- Compose files: `docker-compose.yaml` plus `.devcontainer/docker-compose.extend.yaml`
- Workspace in container: `/home/vscode/acolite-mp`
- Remote user: `vscode`

The dev image is currently based on a local image in `docker/Dockerfile`:

- `local/easi-workflows-acolite:test`

If that image is missing, the container will not build or start.

### Required mounts and regression data

Current internal defaults:

- Fixture repository bind mount:
	- Host: `~/dev/aquawatch/acolite_workflow/acolite-mp-fixtures`
	- Container: `/home/vscode/acolite-mp-fixtures`
- Workflow output mount:
	- Host: `~/dev/devcontainers/data`
	- Container: `/data`
- Optional AWS credentials:
	- Host: `~/.aws`
	- Container: `/home/vscode/.aws`

Regression tests currently expect:

- Fixtures at `/home/vscode/acolite-mp-fixtures/20250600`
- Workflow outputs under `/data/acolite/test/acolite-mp`

If your local paths differ, update either:

- Dev container mount configuration in `.devcontainer/devcontainer.json`, and/or
- The fixture path constant in `tests/utils.py`

### Quick start

1. Open this repository in VS Code.
2. Run `Dev Containers: Reopen in Container`.
3. Verify mounts exist in the container:
	 - `/home/vscode/acolite-mp-fixtures/20250600`
	 - `/data/acolite/test/acolite-mp`
4. Install dependencies (if needed):
	 - `uv sync --extra test`
5. Run the default regression suite:
	 - `pytest`

Useful targeted commands:

- Single test module: `pytest tests/test_acolite_l2r.py -q`
- Network-gated LUT prefetch test: `ACOLITE_TEST_NETWORK=1 pytest tests/test_acolite_luts_prefetch.py`

Performance tests are intentionally disabled by default in `tests/test_perf_*.py`.

### Troubleshooting

- Container build fails on base image:
	- Ensure `local/easi-workflows-acolite:test` exists locally.
- `Fixture path ... does not exist` failures:
	- Confirm fixture mount source exists and is mounted to `/home/vscode/acolite-mp-fixtures`.
- `Input path /data/acolite/test/acolite-mp/... does not exist`:
	- Confirm the `/data` bind mount contains regression workflow outputs.
- Network test skipped unexpectedly:
	- Set `ACOLITE_TEST_NETWORK=1` when running network tests.

For a fuller environment reference, see `docs/plans/development-environment.md`.


