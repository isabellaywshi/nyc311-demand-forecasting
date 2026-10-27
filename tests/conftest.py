import pytest

from nyc311.config import load_categories


@pytest.fixture
def cats():
    return load_categories()
