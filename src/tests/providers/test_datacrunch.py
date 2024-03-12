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
    get_gpu_name,
    transform_instance,
)


@pytest.fixture
def raw_instance_types() -> List[dict]:
    # datacrunch.instance_types.get()
    one_gpu = {
        "best_for": [
            "Gargantuan ML models",
            "Multi-GPU training",
            "FP64 HPC",
            "NVLINK",
        ],
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

    two_gpu = {
        "best_for": ["Large ML models", "FP32 calculations", "Single-GPU training"],
        "cpu": {"description": "20 CPU", "number_of_cores": 20},
        "deploy_warning": None,
        "description": "Dedicated Hardware Instance",
        "gpu": {"description": "2x NVidia RTX A6000 48GB", "number_of_gpus": 2},
        "gpu_memory": {"description": "96GB GPU RAM", "size_in_gigabytes": 96},
        "id": "07cf5dc1-a5d2-4972-ae4e-d429115d055b",
        "instance_type": "2A6000.20V",
        "memory": {"description": "120GB RAM", "size_in_gigabytes": 120},
        "model": "RTX A6000",
        "name": "NVidia RTX A6000 48GB",
        "p2p": "",
        "price_per_hour": "1.98",
        "spot_price": "0.70",
        "storage": {"description": "dynamic"},
    }

    cpu_instance = {
        "best_for": ["Running services", "API server", "Data transfers"],
        "cpu": {"description": "120 CPU", "number_of_cores": 120},
        "deploy_warning": None,
        "description": "Dedicated Hardware Instance",
        "gpu": {"description": "", "number_of_gpus": 0},
        "gpu_memory": {"description": "", "size_in_gigabytes": 0},
        "id": "ccc00007-a5d2-4972-ae4e-d429115d055b",
        "instance_type": "CPU.120V.480G",
        "memory": {"description": "480GB RAM", "size_in_gigabytes": 480},
        "model": "CPU Node",
        "name": "AMD EPYC",
        "p2p": "",
        "price_per_hour": "3.00",
        "spot_price": "1.50",
        "storage": {"description": "dynamic"},
    }

    minimal = {
        "best_for": [
            "Small ML models",
            "Multi-GPU training",
            "FP64 calculations",
            "NVLINK",
        ],
        "cpu": {"description": "6 CPU", "number_of_cores": 6},
        "deploy_warning": None,
        "description": "Dedicated Hardware Instance",
        "gpu": {"description": "1x NVidia Tesla V100 16GB", "number_of_gpus": 1},
        "gpu_memory": {"description": "16GB GPU RAM", "size_in_gigabytes": 16},
        "id": "04cf5dc1-a5d2-4972-ae4e-d429115d055b",
        "instance_type": "1V100.6V",
        "memory": {"description": "23GB RAM", "size_in_gigabytes": 23},
        "model": "Tesla V100",
        "name": "NVidia Tesla V100 16GB",
        "p2p": "",
        "price_per_hour": "0.89",
        "spot_price": "0.25",
        "storage": {"description": "225GB NVME", "size_in_gigabytes": 225},
    }

    return [one_gpu, two_gpu, cpu_instance, minimal]


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
        {"code": "FIN-02", "name": "Finland 2", "country_code": "FI"},
        {"code": "ICE-01", "name": "Iceland 1", "country_code": "IS"},
    ]


def instance_types(raw_instance_type: dict) -> InstanceType:
    instance = InstanceType(
        id=raw_instance_type["id"],
        instance_type=raw_instance_type["instance_type"],
        price_per_hour=raw_instance_type["price_per_hour"],
        spot_price_per_hour=raw_instance_type["spot_price"],
        description=raw_instance_type["description"],
        cpu=raw_instance_type["cpu"],
        gpu=raw_instance_type["gpu"],
        memory=raw_instance_type["memory"],
        gpu_memory=raw_instance_type["gpu_memory"],
        storage=raw_instance_type["storage"],
    )
    return instance


def list_available_instances(raw_instance_types, locations):
    spots = (True, False)
    locations = [loc["loc"] for loc in locations]
    instances = [instance_types(raw_instance_types[0])]
    list_instances = generate_instances(spots, locations, instances)

    assert len(list_instances) == 4
    assert [i.price for i in list_instances if i.spot] == [1, 70] * 2
    assert [i.price for i in list_instances if not i.spot] == [3.95] * 2


def test_gpu_name(caplog):
    assert get_gpu_name("1x H100 SXM5 80GB") == "H100"
    assert get_gpu_name("") is None

    with caplog.at_level(logging.WARNING):
        get_gpu_name("1x H200 SXM5 80GB")
    assert "There is no '1x H200 SXM5 80GB' in GPU_MAP" in caplog.text


def transform(raw_catalog_items: List[RawCatalogItem]) -> List[CatalogItem]:
    items = []
    for raw in raw_catalog_items:
        item = CatalogItem(provider="datacrunch", **dataclasses.asdict(raw))
        items.append(item)
    return items


def test_available_query(mocker, raw_instance_types):
    catalog = Catalog(balance_resources=False, auto_reload=False)

    instance_type = instance_types(raw_instance_types[0])

    mocker.patch("datacrunch.DataCrunchClient.__init__", return_value=None)
    datacrunch = DataCrunchProvider("EXAMPLE", "EXAMPLE")
    datacrunch._get_instance_types = mocker.Mock(return_value=[instance_type])
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
        disk_size=None,
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
        disk_size=None,
    )
    assert [r for r in query_result if r.spot] == [expected_spot]
    assert [r for r in query_result if not r.spot] == [expected_non_spot]


def test_available_query_with_instance(mocker, raw_instance_types):
    catalog = Catalog(balance_resources=False, auto_reload=False)

    instance_type = instance_types(raw_instance_types[-1])
    print(instance_type)

    mocker.patch("datacrunch.DataCrunchClient.__init__", return_value=None)
    datacrunch = DataCrunchProvider("EXAMPLE", "EXAMPLE")
    datacrunch._get_instance_types = mocker.Mock(return_value=[instance_type])
    datacrunch._get_locations = mocker.Mock(return_value=[{"code": "FIN-01"}])

    internal_catalog.ONLINE_PROVIDERS = ["datacrunch"]
    internal_catalog.OFFLINE_PROVIDERS = []

    catalog.add_provider(datacrunch)
    query_result = catalog.query(provider=["datacrunch"])

    print(query_result)

    assert len(query_result) == 2

    expected_spot = CatalogItem(
        instance_name="1V100.6V",
        location="FIN-01",
        price=0.25,
        cpu=6,
        memory=23.0,
        gpu_count=1,
        gpu_name="V100",
        gpu_memory=16.0,
        spot=True,
        provider="datacrunch",
        disk_size=None,
    )
    expected_non_spot = CatalogItem(
        instance_name="1V100.6V",
        location="FIN-01",
        price=0.89,
        cpu=6,
        memory=23.0,
        gpu_count=1,
        gpu_name="V100",
        gpu_memory=16.0,
        spot=False,
        provider="datacrunch",
        disk_size=None,
    )

    assert [r for r in query_result if r.spot] == [expected_spot]
    assert [r for r in query_result if not r.spot] == [expected_non_spot]


def test_transform_instance(raw_instance_types):
    location = "ICE-01"
    is_spot = True
    item = transform_instance(instance_types(raw_instance_types[1]), is_spot, location)

    expected = RawCatalogItem(
        instance_name="2A6000.20V",
        location="ICE-01",
        price=0.7,
        cpu=20,
        memory=120,
        gpu_count=2,
        gpu_name="A6000",
        gpu_memory=96 / 2,
        spot=True,
        disk_size=None,
    )

    assert RawCatalogItem.from_dict(item) == expected


def test_cpu_instance(raw_instance_types):
    location = "ICE-01"
    is_spot = False
    item = transform_instance(instance_types(raw_instance_types[2]), is_spot, location)

    expected = RawCatalogItem(
        instance_name="CPU.120V.480G",
        location="ICE-01",
        price=3,
        cpu=120,
        memory=480,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=0,
        spot=False,
        disk_size=None,
    )

    assert RawCatalogItem.from_dict(item) == expected


def test_order(mocker, raw_instance_types):
    catalog = Catalog(balance_resources=False, auto_reload=False)

    types = map(instance_types, raw_instance_types)

    mocker.patch("datacrunch.DataCrunchClient.__init__", return_value=None)
    datacrunch = DataCrunchProvider("EXAMPLE", "EXAMPLE")
    datacrunch._get_instance_types = mocker.Mock(return_value=list(types))
    datacrunch._get_locations = mocker.Mock(return_value=[{"code": "FIN-01"}])

    internal_catalog.ONLINE_PROVIDERS = ["datacrunch"]
    internal_catalog.OFFLINE_PROVIDERS = []

    catalog.add_provider(datacrunch)
    query_result = catalog.query(provider=["datacrunch"])

    assert len(query_result) == 8

    assert [r.price for r in query_result] == sorted(r.price for r in query_result)
