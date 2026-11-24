"""Test configuration for the simulation package."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # numpy on Apple's Accelerate raises spurious floating-point warnings in matmul on finite inputs
    # (see i2rt/simulation/__init__.py); pytest would otherwise re-enable and print every one.
    config.addinivalue_line("filterwarnings", "ignore:.*encountered in matmul:RuntimeWarning")
