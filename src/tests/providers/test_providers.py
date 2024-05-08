import importlib
import inspect
import pkgutil

import pytest

import gpuhunt.providers
from gpuhunt._internal.catalog import OFFLINE_PROVIDERS, ONLINE_PROVIDERS


@pytest.fixture()
def providers():
    """List of all provider classes"""
    members = []
    for module_info in pkgutil.walk_packages(gpuhunt.providers.__path__):
        module = importlib.import_module(
            f".{module_info.name}",
            package="gpuhunt.providers",
        )
        for _, member in inspect.getmembers(module):
            if not inspect.isclass(member):
                continue
            if not issubclass(member, gpuhunt.providers.AbstractProvider):
                continue
            if member.__name__ == "AbstractProvider":
                continue
            if member.NAME == "nebius":  # The provider has been temporarily disabled
                continue
            members.append(member)
    return members


def test_catalog_providers_is_unique():
    CATALOG_PROVIDERS = OFFLINE_PROVIDERS + ONLINE_PROVIDERS
    assert len(set(CATALOG_PROVIDERS)) == len(CATALOG_PROVIDERS)


def test_all_providers_have_a_names(providers):
    names = [p.NAME for p in providers]
    assert gpuhunt.providers.AbstractProvider.NAME not in names
    assert len(set(names)) == len(names)


def test_catalog_providers(providers):
    CATALOG_PROVIDERS = OFFLINE_PROVIDERS + ONLINE_PROVIDERS
    names = [p.NAME for p in providers]
    assert set(CATALOG_PROVIDERS) == set(names)
    assert len(CATALOG_PROVIDERS) == len(names)
