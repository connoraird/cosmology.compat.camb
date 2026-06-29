from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import array_api_strict
import jax
import pytest

if TYPE_CHECKING:
    from types import ModuleType

xp_available_backends: dict[str, ModuleType] = {
    "numpy": np,
    "array_api_strict": array_api_strict,
    "jax.numpy": jax.numpy,
}

@pytest.fixture(params=xp_available_backends.values(), scope="session")
def xp(request: pytest.FixtureRequest) -> ModuleType:
    """
    Fixture for array backend.

    Access array library functions using `xp.` in tests.

    """
    return request.param