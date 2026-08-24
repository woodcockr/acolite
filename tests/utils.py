import os
from pathlib import Path

import numpy as np

# Fixtures live in a versioned sibling directory of this repo checkout by default
# (see scripts/download_fixtures.sh); override with ACOLITE_FIXTURES_DIR if needed.
_repo_root = Path(__file__).resolve().parents[1]
_fixtures_version = (_repo_root / "FIXTURES_VERSION").read_text().strip()
_fixtures_base_dir = Path(os.environ.get("ACOLITE_FIXTURES_DIR", _repo_root.parent / "acolite-mp-fixtures"))
acolite_fixtures_path = str(_fixtures_base_dir / _fixtures_version)

def arrays_almost_equal(da1, da2, rtol=1e-5, atol=1e-6):
    return (
        da1.shape == da2.shape and
        da1.dims == da2.dims and
        np.allclose(da1.values, da2.values, rtol=rtol, atol=atol, equal_nan=True)
    )