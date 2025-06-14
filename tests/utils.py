import numpy as np

# Set this variable to the location of the acolite-mp-fixtures repository on your system
acolite_fixtures_path = "/home/vscode/acolite-mp-fixtures/20250600"

def arrays_almost_equal(da1, da2, rtol=1e-5, atol=1e-6):
    return (
        da1.shape == da2.shape and
        da1.dims == da2.dims and
        np.allclose(da1.values, da2.values, rtol=rtol, atol=atol, equal_nan=True)
    )