from typing import Union
from unittest.mock import Mock

import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog, CatalogItem, RawCatalogItem
from gpuhunt.providers.tensordock import TensorDockProvider
from gpuhunt.providers.vastai import VastAIProvider


class TestQuery:
    def test_query_merge(self):
        catalog = Catalog(balance_resources=False, auto_reload=False)

        tensordock = TensorDockProvider()
        tensordock.get = Mock(return_value=[catalog_item(price=1), catalog_item(price=3)])
        catalog.add_provider(tensordock)

        vastai = VastAIProvider()
        vastai.get = Mock(return_value=[catalog_item(price=2), catalog_item(price=1)])
        catalog.add_provider(vastai)

        assert catalog.query(provider=["tensordock", "vastai"]) == [
            catalog_item(provider="tensordock", price=1),
            catalog_item(provider="vastai", price=2),
            catalog_item(provider="vastai", price=1),
            catalog_item(provider="tensordock", price=3),
        ]

    def test_no_providers_some_not_loaded(self):
        catalog = Catalog(balance_resources=False, auto_reload=False)

        tensordock = TensorDockProvider()
        tensordock.get = Mock(return_value=[catalog_item(price=1)])
        catalog.add_provider(tensordock)

        internal_catalog.OFFLINE_PROVIDERS = []
        assert catalog.query() == [
            catalog_item(provider="tensordock", price=1),
        ]

    def test_provider_filter(self):
        catalog = Catalog(balance_resources=False, auto_reload=False)
        catalog.add_provider(tensordock := TensorDockProvider())
        catalog.add_provider(vastai := VastAIProvider())

        tensordock_offers = [catalog_item(price=1)]
        vastai_offers = [catalog_item(price=2), catalog_item(price=3)]

        tensordock.get = Mock(return_value=tensordock_offers)
        vastai.get = Mock(return_value=vastai_offers)

        assert len(catalog.query(provider="tensordock")) == 1
        assert len(catalog.query(provider="Tensordock")) == 1
        assert len(catalog.query(provider="vastai")) == 2
        assert len(catalog.query(provider="VastAI")) == 2
        assert len(catalog.query(provider=["tensordock", "VastAI"])) == 3

    def test_gpu_name_filter(self):
        catalog = Catalog(balance_resources=False, auto_reload=False)
        catalog.add_provider(tensordock := TensorDockProvider())

        tensordock.get = Mock(
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


def catalog_item(**kwargs) -> Union[CatalogItem, RawCatalogItem]:
    values = dict(
        instance_name="instance",
        cpu=1,
        memory=1,
        gpu_vendor="nvidia",
        gpu_count=1,
        gpu_name="gpu",
        gpu_memory=1,
        location="location",
        price=1,
        spot=False,
        disk_size=None,
    )
    values.update(kwargs)
    if "provider" in values:
        return CatalogItem(**values)
    return RawCatalogItem(**values)
