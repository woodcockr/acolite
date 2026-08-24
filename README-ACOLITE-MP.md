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

### Prerequisite: base dev image

The dev container builds `docker/Dockerfile`, which is based on a local image that is **not** built by anything in this repo:

- `local/easi-workflows-acolite:test`

Build/obtain it from the internal image repo before opening the dev container:

```
git clone https://github.com/csiro/csa-easi-workflows-images.git
cd csa-easi-workflows-images
make easi-workflows-acolite
```

If that image is missing, the container will not build or start.

### Dev container wiring

- Service: `acolite-mp-dev`
- Compose files: `docker-compose.yaml` plus `.devcontainer/docker-compose.extend.yaml`
- Workspace in container: `/home/vscode/acolite-mp`
- Remote user: `vscode`

### Required mounts and regression data

- Repo checkout (workspace):
  - Host: repository root
  - Container: `/home/vscode/acolite-mp`
- Fixtures directory (sibling of the repo checkout, see below):
  - Host: `<repo-parent>/acolite-mp-fixtures`
  - Container: `/home/vscode/acolite-mp-fixtures`
- Workflow output directory, configurable via the repo's tracked `.env` (`ACOLITE_DATA_DIR`, default `../acolite-mp-data`):
  - Host: `$ACOLITE_DATA_DIR`
  - Container: `/data`
- Optional AWS credentials:
  - Host: `~/.aws`
  - Container: `/home/vscode/.aws`

These mounts are resolved relative to your checkout (`${localWorkspaceFolder}` in `.devcontainer/devcontainer.json`, `.env` for `docker-compose.yaml`), so they work regardless of where you cloned the repo — no per-developer path edits needed.

### Fixtures: version and download

The regression test fixture set is versioned by the top-level [`FIXTURES_VERSION`](FIXTURES_VERSION) file (currently `20260820`), which must match an object in S3:

```
s3://adias-prod-dc-data-projects/csa-disr/acolite-mp/acolite-mp-fixtures_<FIXTURES_VERSION>.zip
```

Download and unpack fixtures into the sibling directory `../acolite-mp-fixtures/<FIXTURES_VERSION>` with:

```
AWS_PROFILE=<your-sso-profile> scripts/download_fixtures.sh
```

- `AWS_PROFILE` is required and developer-specific (an SSO profile with access to the fixtures bucket); it is never hardcoded in this repo.
- The script is idempotent — it skips downloading if the versioned fixtures directory already exists. Pass `--force` to re-download.
- Override the fixtures location with `ACOLITE_FIXTURES_DIR` if your checkout layout differs from the sibling-directory convention.
- When the source checkout is updated, bump `FIXTURES_VERSION` and re-run the script to fetch the matching fixture set.

`tests/utils.py` resolves `acolite_fixtures_path` from `FIXTURES_VERSION` and the same sibling/override convention, so it stays in sync with the download script automatically.

### Quick start

1. Build/obtain the base image (see Prerequisite above).
2. Adjust `ACOLITE_DATA_DIR` in the repo's `.env` file if the default (`../acolite-mp-data`) doesn't suit your setup.
3. Open this repository in VS Code and run `Dev Containers: Reopen in Container`.
4. Download fixtures (from inside or outside the container, using a host path accessible to both):
	 - `AWS_PROFILE=<your-sso-profile> scripts/download_fixtures.sh`
5. Install dependencies (if needed):
	 - `uv sync --extra test`
6. Run the default regression suite:
	 - `pytest`

Useful targeted commands:

- Single test module: `pytest tests/test_acolite_l2r.py -q`
- Network-gated LUT prefetch test: `ACOLITE_TEST_NETWORK=1 pytest tests/test_acolite_luts_prefetch.py`

Performance tests are intentionally disabled by default in `tests/test_perf_*.py`.

### Troubleshooting

- Container build fails on base image:
	- Ensure `local/easi-workflows-acolite:test` exists locally (see Prerequisite above).
- `Fixture path ... does not exist` failures:
	- Run `scripts/download_fixtures.sh` with `AWS_PROFILE` set, and confirm `FIXTURES_VERSION` matches the fixture set you expect.
- AWS auth errors during fixture download:
	- Confirm `AWS_PROFILE` is set to a valid, logged-in SSO profile (`aws sso login --profile <profile>`).
- `/data` mount is empty or wrong location:
	- Confirm `ACOLITE_DATA_DIR` in `.env` points to a directory that exists on the host.
- `Input path /data/acolite/test/acolite-mp/... does not exist`:
	- Confirm the `/data` bind mount contains regression workflow outputs.
- Network test skipped unexpectedly:
	- Set `ACOLITE_TEST_NETWORK=1` when running network tests.
- Container `acolite` import resolving to `/opt/acolite` instead of the workspace:
	- The base image sets `PYTHONPATH=/opt/acolite`. `pytest.ini` sets `pythonpath = .` so pytest always prioritises the workspace package; if you run acolite outside pytest, unset/override `PYTHONPATH` explicitly.


