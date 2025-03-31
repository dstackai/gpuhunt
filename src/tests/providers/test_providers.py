import importlib
import inspect
import pkgutil
import sys

import pytest

import gpuhunt.providers
from gpuhunt._internal.catalog import OFFLINE_PROVIDERS, ONLINE_PROVIDERS


@pytest.fixture()
def providers():
    """List of all provider classes"""
    members = []
    for module_info in pkgutil.walk_packages(gpuhunt.providers.__path__):
        if sys.version_info < (3, 10) and module_info.name == "nebius":
            continue
        module = importlib.import_module(
            f".{module_info.name}",
            package="gpuhunt.providers",
        )
        for _, member in inspect.getmembers(module):
            if not inspect.isclass(member):
                continue
            if member.__name__.islower():
                continue  # skip builtins to avoid CPython bug #89489 in `issubclass` below
            if not issubclass(member, gpuhunt.providers.AbstractProvider):
                continue
            if member.__name__ == "AbstractProvider":
                continue
            members.append(member)
    assert members
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
    if sys.version_info < (3, 10):
        CATALOG_PROVIDERS = [p for p in CATALOG_PROVIDERS if p != "nebius"]
    names = [p.NAME for p in providers]
    assert set(CATALOG_PROVIDERS) == set(names)
    assert len(CATALOG_PROVIDERS) == len(names)
