import dataclasses
import logging
from typing import List

import pytest

import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog, CatalogItem, RawCatalogItem
from gpuhunt.providers.datacrunch import (
    DataCrunchProvider,
    InstanceType,
    generate_instances,
    gpu_name,
)


@pytest.fixture
def raw_instance_types() -> List[dict]:
    # datacrunch.instance_types.get()
    item = {
        "best_for": ["Gargantuan ML models", "Multi-GPU training", "FP64 HPC", "NVLINK"],
        "cpu": {"description": "30 CPU", "number_of_cores": 30},
        "deploy_warning": "H100: Use Nvidia driver 535 or higher for best performance",
        "description": "Dedicated Hardware Instance",
        "gpu": {"description": "1x H100 SXM5 80GB", "number_of_gpus": 1},
        "gpu_memory": {"description": "80GB GPU RAM", "size_in_gigabytes": 80},
        "id": "c01dd00d-0000-480b-ae4e-d429115d055b",
        "instance_type": "1H100.80S.30V",
        "memory": {"description": "120GB RAM", "size_in_gigabytes": 120},
        "model": "H100 80GB",
        "name": "H100 SXM5 80GB",
        "p2p": "",
        "price_per_hour": "3.95",
        "spot_price": "1.70",
        "storage": {"description": "dynamic"},
    }
    return [item]


@pytest.fixture
def availabilities() -> List[dict]:
    # datacrunch.instances.get_availabilities(is_spot=True)
    data = [
        {
            "location_code": "FIN-01",
            "availabilities": [
                "1A100.22V",
                "1RTX6000ADA.10V",
                "8RTX6000ADA.80V",
                "2A6000.20V",
                "1V100.6V",
                "2V100.10V",
                "4V100.20V",
                "8V100.48V",
                "CPU.4V.16G",
                "CPU.8V.32G",
                "CPU.16V.64G",
                "CPU.32V.128G",
                "CPU.64V.256G",
                "CPU.96V.384G",
            ],
        },
        {
            "location_code": "ICE-01",
            "availabilities": [
                "CPU.4V.16G",
                "CPU.8V.32G",
                "CPU.16V.64G",
                "CPU.32V.128G",
                "CPU.64V.256G",
                "CPU.96V.384G",
                "CPU.120V.480G",
            ],
        },
    ]
    return data


@pytest.fixture
def locations():
    # datacrunch.locations.get()
    return [
        {"code": "FIN-01", "name": "Finland 1", "country_code": "FI"},
        {"code": "ICE-01", "name": "Iceland 1", "country_code": "IS"},
    ]


@pytest.fixture
def instance_types(raw_instance_types):
    item = raw_instance_types.pop()
    instance = InstanceType(
        id=item["id"],
        instance_type=item["instance_type"],
        price_per_hour=item["price_per_hour"],
        spot_price_per_hour=item["spot_price"],
        description=item["description"],
        cpu=item["cpu"],
        gpu=item["gpu"],
        memory=item["memory"],
        gpu_memory=item["gpu_memory"],
        storage=item["storage"],
    )
    return instance


def list_available_instances(instance_types, locations):
    spots = (True, False)
    locations = [loc["loc"] for loc in locations]
    instances = [instance_types]
    list_instances = generate_instances(spots, locations, instances)

    assert len(list_instances) == 4
    assert [i.price for i in list_instances if i.spot] == [1, 70] * 2
    assert [i.price for i in list_instances if not i.spot] == [3.95] * 2


def test_gpu_name(caplog):
    assert gpu_name("1x H100 SXM5 80GB") == "H100"
    assert gpu_name("") is None
    assert gpu_name(None) is None

    with caplog.at_level(logging.WARNING):
        gpu_name("1x H200 SXM5 80GB")
    assert "There is no '1x H200 SXM5 80GB' in gpu_map" in caplog.text


def transform(raw_catalog_items: List[RawCatalogItem]) -> List[CatalogItem]:
    items = []
    for raw in raw_catalog_items:
        item = CatalogItem(provider="datacrunch", **dataclasses.asdict(raw))
        items.append(item)
    return items


def test_available_query(mocker, instance_types):
    catalog = Catalog(fill_missing=False, auto_reload=False)

    availabilities = [{"location_code": "FIN-01", "availabilities": ["1H100.80S.30V"]}]

    mocker.patch("datacrunch.DataCrunchClient.__init__", return_value=None)
    datacrunch = DataCrunchProvider("EXAMPLE", "EXAMPLE")
    datacrunch._get_instance_types = mocker.Mock(return_value=[instance_types])
    datacrunch._get_locations = mocker.Mock(return_value=[{"code": "FIN-01"}])

    internal_catalog.ONLINE_PROVIDERS = ["datacrunch"]
    internal_catalog.OFFLINE_PROVIDERS = []

    catalog.add_provider(datacrunch)
    query_result = catalog.query(provider=["datacrunch"])

    assert len(query_result) == 2

    expected_spot = CatalogItem(
        instance_name="1H100.80S.30V",
        location="FIN-01",
        price=1.7,
        cpu=30,
        memory=120.0,
        gpu_count=1,
        gpu_name="H100",
        gpu_memory=80.0,
        spot=True,
        provider="datacrunch",
    )
    expected_non_spot = CatalogItem(
        instance_name="1H100.80S.30V",
        location="FIN-01",
        price=3.95,
        cpu=30,
        memory=120.0,
        gpu_count=1,
        gpu_name="H100",
        gpu_memory=80.0,
        spot=False,
        provider="datacrunch",
    )
    assert [r for r in query_result if r.spot] == [expected_spot]
    assert [r for r in query_result if not r.spot] == [expected_non_spot]
