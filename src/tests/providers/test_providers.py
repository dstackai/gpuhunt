import importlib
import inspect
import pkgutil

import pytest

import gpuhunt.providers
import gpuhunt.providers.base
from gpuhunt._internal.catalog import OFFLINE_PROVIDERS, ONLINE_PROVIDERS
from gpuhunt._internal.default import ONLINE_PROVIDER_MODULES
from gpuhunt.providers.base import OfflineProvider, OnlineProvider


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
            if member.__name__.islower():
                continue  # skip builtins to avoid CPython bug #89489 in `issubclass` below
            if not issubclass(member, gpuhunt.providers.base.AbstractProvider):
                continue
            if inspect.isabstract(member):
                continue  # skip AbstractProvider, OnlineProvider, OfflineProvider
            members.append(member)
    assert members
    return members


def test_catalog_providers_is_unique():
    CATALOG_PROVIDERS = OFFLINE_PROVIDERS + ONLINE_PROVIDERS
    assert len(set(CATALOG_PROVIDERS)) == len(CATALOG_PROVIDERS)


def test_all_providers_have_a_names(providers):
    names = [p.NAME for p in providers]
    assert gpuhunt.providers.base.AbstractProvider.NAME not in names
    assert len(set(names)) == len(names)


def test_catalog_providers(providers):
    CATALOG_PROVIDERS = OFFLINE_PROVIDERS + ONLINE_PROVIDERS
    names = [p.NAME for p in providers]
    assert set(CATALOG_PROVIDERS) == set(names)
    assert len(CATALOG_PROVIDERS) == len(names)


def test_online_providers_subclass_online_provider(providers):
    online = [p for p in providers if p.NAME in ONLINE_PROVIDERS]
    assert online
    for provider in online:
        assert issubclass(provider, OnlineProvider), provider


def test_offline_providers_subclass_offline_provider(providers):
    offline = [p for p in providers if p.NAME in OFFLINE_PROVIDERS]
    assert offline
    for provider in offline:
        assert issubclass(provider, OfflineProvider), provider


def test_default_catalog_loads_every_online_provider(providers):
    classes_by_name = {p.__name__: p for p in providers}
    names = {classes_by_name[class_name].NAME for _, class_name in ONLINE_PROVIDER_MODULES}
    assert names == set(ONLINE_PROVIDERS)
