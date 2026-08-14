import pytest

from gpuhunt._internal.catalog import Catalog
from gpuhunt._internal.default import default_catalog

CREDS_ENV_VARS = [
    "CRUSOE_ACCESS_KEY",
    "CRUSOE_SECRET_KEY",
    "CRUSOE_PROJECT_ID",
    "DIGITAL_OCEAN_API_KEY",
    "HOTAISLE_API_KEY",
    "HOTAISLE_TEAM_HANDLE",
    "JL_API_KEY",
    "SEEWEB_API_TOKEN",
]


@pytest.fixture
def offline_catalog(monkeypatch):
    monkeypatch.setattr(Catalog, "load", lambda self, version=None: None)
    for var in CREDS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    default_catalog.cache_clear()
    yield
    default_catalog.cache_clear()


class TestDefaultCatalog:
    def test_skips_providers_with_missing_creds(self, offline_catalog) -> None:
        catalog = default_catalog()
        assert sorted(p.NAME for p in catalog.providers) == ["vastai", "vultr"]

    def test_loads_providers_with_creds(self, offline_catalog, monkeypatch) -> None:
        monkeypatch.setenv("HOTAISLE_API_KEY", "key")
        monkeypatch.setenv("HOTAISLE_TEAM_HANDLE", "team")
        catalog = default_catalog()
        assert "hotaisle" in [p.NAME for p in catalog.providers]
