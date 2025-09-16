import os

import pytest

from gpuhunt.providers.hotaisle import HotAisleProvider


@pytest.fixture
def provider():
    api_key = os.environ.get("HOTAISLE_API_KEY")
    team_handle = os.environ.get("HOTAISLE_TEAM_HANDLE")
    return HotAisleProvider(api_key=api_key, team_handle=team_handle)


@pytest.fixture
def offers(provider):
    """Fixture that provides the list of offers from HotAisle."""
    return provider.get()


def test_positive_prices(offers):
    """Test that all offers have positive prices."""
    assert all(offer.price > 0 for offer in offers)
