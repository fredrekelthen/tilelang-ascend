"""
Pytest configuration and shared fixtures for language API tests.

Fixtures defined here are automatically available to all test files
in testing/python/language/ and its subdirectories — no import needed.

Markers (l0/l1/l2/compile_time + project low_priority/ci_skip) are
registered in pyproject.toml [tool.pytest.ini_options].
"""

import pytest
import torch
import tilelang


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def disable_tilelang_cache():
    """Disable tilelang JIT cache for the entire test session."""
    tilelang.disable_cache()


@pytest.fixture(autouse=True)
def random_seed():
    """Set deterministic random seed for reproducibility."""
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Marker registration (moved to pyproject.toml [tool.pytest.ini_options])
# ---------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    """Parametrize 'dtype' from the test class's op spec.

    Each base test class declares a _dtype_source attribute (e.g.
    "supported_dtypes" or "boundary_dtypes") that points to a list on
    the BinaryOpSpec.  This hook reads that list and feeds it to pytest.
    """
    if "dtype" not in metafunc.fixturenames or metafunc.cls is None:
        return
    op = getattr(metafunc.cls, "op", None)
    if op is None:
        return
    source = getattr(metafunc.cls, "_dtype_source", "supported_dtypes")
    dtypes = getattr(op, source, None)
    if dtypes:
        metafunc.parametrize("dtype", dtypes)
