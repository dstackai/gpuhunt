import os

import pytest

from gpuhunt.providers.hotaisle import HotAisleProvider


@pytest.fixture
def provider():
    return HotAisleProvider(
        api_key=os.environ["HOTAISLE_API_KEY"],
        team_handle=os.environ["HOTAISLE_TEAM_HANDLE"],
    )


@pytest.fixture
def offers(provider):
    """Fixture that provides the list of offers from HotAisle."""
    return provider.get()


def test_positive_prices(offers):
    """Test that all offers have positive prices."""
    assert all(offer.price > 0 for offer in offers)
