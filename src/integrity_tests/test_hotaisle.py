import os

import pytest

from gpuhunt.providers.hotaisle import HotAisleProvider


@pytest.fixture
def provider():
    api_key = os.environ.get("HOTAISLE_API_KEY")
    team_name = os.environ.get("HOTAISLE_TEAM_NAME")
    return HotAisleProvider(api_key=api_key, team_name=team_name)


@pytest.fixture
def offers(provider):
    """Fixture that provides the list of offers from HotAisle."""
    return provider.get()


def test_positive_prices(offers):
    """Test that all offers have positive prices."""
    assert all(offer.price > 0 for offer in offers)


def test_prices_converted_from_cents(offers):
    """Test that prices are properly converted from cents to dollars."""
    for offer in offers:
        # Prices should be in reasonable dollar range for GPU instances
        # If HotAisle changed API to return dollars instead of cents,
        # our /100 conversion would make prices way too low (e.g., $1.99 -> $0.0199)
        assert (
            offer.price >= 1.0
        ), f"Price {offer.price} too low - HotAisle may have changed from cents to dollars"
