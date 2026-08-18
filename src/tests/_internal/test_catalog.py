import io
import urllib.error
import urllib.request
import zipfile
from unittest.mock import Mock

import pytest

import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import AcceleratorVendor, Catalog, CatalogItem, GPUHuntError
from gpuhunt._internal.storage import CATALOG_V2_FIELDS
from gpuhunt.providers.vastai import VastAIProvider
from gpuhunt.providers.vultr import VultrProvider


class TestQuery:
    def test_query_merge(self):
        catalog = Catalog(balance_resources=False, auto_reload=False)

        vultr = VultrProvider()
        vultr.get = Mock(return_value=[catalog_item(price=1), catalog_item(price=3)])
        catalog.add_provider(vultr)

        vastai = VastAIProvider()
        vastai.get = Mock(
            return_value=[
                catalog_item(provider="vastai", price=2),
                catalog_item(provider="vastai", price=1),
            ]
        )
        catalog.add_provider(vastai)

        assert catalog.query(provider=["vultr", "vastai"]) == [
            catalog_item(provider="vultr", price=1),
            catalog_item(provider="vastai", price=2),
            catalog_item(provider="vastai", price=1),
            catalog_item(provider="vultr", price=3),
        ]

    def test_no_providers_some_not_loaded(self):
        catalog = Catalog(balance_resources=False, auto_reload=False)

        vultr = VultrProvider()
        vultr.get = Mock(return_value=[catalog_item(price=1)])
        catalog.add_provider(vultr)

        internal_catalog.OFFLINE_PROVIDERS = []
        assert catalog.query() == [
            catalog_item(provider="vultr", price=1),
        ]

    def test_provider_filter(self):
        catalog = Catalog(balance_resources=False, auto_reload=False)
        catalog.add_provider(vultr := VultrProvider())
        catalog.add_provider(vastai := VastAIProvider())

        vultr_offers = [catalog_item(price=1)]
        vastai_offers = [
            catalog_item(provider="vastai", price=2),
            catalog_item(provider="vastai", price=3),
        ]

        vultr.get = Mock(return_value=vultr_offers)
        vastai.get = Mock(return_value=vastai_offers)

        assert len(catalog.query(provider="vultr")) == 1
        assert len(catalog.query(provider="Vultr")) == 1
        assert len(catalog.query(provider="vastai")) == 2
        assert len(catalog.query(provider="VastAI")) == 2
        assert len(catalog.query(provider=["vultr", "VastAI"])) == 3

    def test_gpu_name_filter(self):
        catalog = Catalog(balance_resources=False, auto_reload=False)
        catalog.add_provider(vultr := VultrProvider())

        vultr.get = Mock(
            return_value=[
                catalog_item(gpu_name="A10"),
                catalog_item(gpu_name="A100"),
                catalog_item(gpu_name="a100"),
            ]
        )

        assert len(catalog.query(gpu_name="V100")) == 0
        assert len(catalog.query(gpu_name="A10")) == 1
        assert len(catalog.query(gpu_name="a10")) == 1
        assert len(catalog.query(gpu_name="A100")) == 2
        assert len(catalog.query(gpu_name="a100")) == 2
        assert len(catalog.query(gpu_name=["a10", "A100"])) == 3


def catalog_item(
    provider: str = "vultr", price: float = 1, gpu_name: str | None = "gpu"
) -> CatalogItem:
    return CatalogItem(
        provider=provider,
        instance_name="instance",
        cpu=1,
        memory=1,
        gpu_vendor=AcceleratorVendor.NVIDIA,
        gpu_count=1,
        gpu_name=gpu_name,
        gpu_memory=1,
        location="location",
        price=price,
        spot=False,
        disk_size=None,
    )


class TestLoad:
    def test_load(self, bucket):
        catalog = Catalog(auto_reload=False)
        catalog.load()

        assert prices(catalog, "aws") == [1.0]
        assert prices(catalog, "gcp") == [1.0]

    def test_load_version(self, bucket):
        catalog = Catalog(auto_reload=False)
        catalog.load(version="2")

        assert prices(catalog, "aws") == [2.0]
        assert prices(catalog, "gcp") == [2.0]

    def test_keeps_loaded_catalog_if_provider_fails(self, bucket):
        catalog = Catalog(auto_reload=False)
        catalog.load()
        bucket.versions["aws"] = "2"
        bucket.versions["gcp"] = "2"
        bucket.failing_providers.add("aws")
        catalog.load()

        assert prices(catalog, "aws") == [1.0]
        assert prices(catalog, "gcp") == [2.0]

    def test_raises_if_all_providers_fail(self, bucket):
        catalog = Catalog(auto_reload=False)
        bucket.failing_providers.update(bucket.versions)

        with pytest.raises(GPUHuntError):
            catalog.load()
        assert catalog.loaded_at is None

    def test_raises_if_version_is_not_available(self, bucket):
        catalog = Catalog(auto_reload=False)
        catalog.load()
        bucket.failing_providers.add("aws")

        with pytest.raises(GPUHuntError):
            catalog.load(version="2")
        assert prices(catalog, "gcp") == [1.0]

    def test_skips_up_to_date_providers(self, bucket):
        catalog = Catalog(auto_reload=False)
        catalog.load()
        bucket.versions["gcp"] = "2"
        bucket.downloads.clear()
        catalog.load()

        assert bucket.downloads == [("gcp", "2")]


class FakeBucket:
    """Serves the v3 catalogs of the `aws` and `gcp` providers. The offer price in a
    catalog is equal to the catalog version, so that tests can tell versions apart."""

    def __init__(self) -> None:
        self.versions = {"aws": "1", "gcp": "1"}
        self.failing_providers: set[str] = set()
        self.downloads: list[tuple[str, str]] = []

    def urlopen(self, url: str) -> io.BytesIO:
        provider, path = url.removeprefix(f"{internal_catalog.CATALOG_URL}/").split("/", 1)
        if provider in self.failing_providers:
            raise urllib.error.URLError(f"{provider} is unavailable")
        if path == "version":
            return io.BytesIO(self.versions[provider].encode())
        version = path.removesuffix("/catalog.zip")
        self.downloads.append((provider, version))
        return io.BytesIO(catalog_zip(provider, price=float(version)))


@pytest.fixture
def bucket(monkeypatch: pytest.MonkeyPatch) -> FakeBucket:
    bucket = FakeBucket()
    monkeypatch.setattr(internal_catalog, "OFFLINE_PROVIDERS", list(bucket.versions))
    monkeypatch.setattr(urllib.request, "urlopen", bucket.urlopen)
    return bucket


def catalog_zip(provider: str, price: float) -> bytes:
    row = {field: "" for field in CATALOG_V2_FIELDS}
    row.update(
        instance_name="instance",
        location="location",
        price=str(price),
        cpu="1",
        memory="1",
        gpu_count="0",
        spot="False",
        cpu_arch="x86",
        provider_data="{}",
    )
    csv = ",".join(CATALOG_V2_FIELDS) + "\n" + ",".join(row[f] for f in CATALOG_V2_FIELDS) + "\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        zip_file.writestr(f"{provider}.csv", csv)
    return buffer.getvalue()


def prices(catalog: Catalog, provider: str) -> list[float]:
    return [item.price for item in catalog.catalog[provider].items]
