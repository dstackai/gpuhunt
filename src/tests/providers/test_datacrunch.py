import logging
from typing import List

import pytest

from gpuhunt.providers.datacrunch import (
    InstanceType,
    _make_availability_list,
    _make_list_available_instances,
    gpu_name,
    make_instance_map,
)


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
def instance_type():
    # datacrunch.instance_types.get()[0]
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


def test_instance_map(instance_type):
    result = make_instance_map([instance_type])
    assert "location" not in result[instance_type.instance_type]
    assert "spot" not in result[instance_type.instance_type]
    assert result[instance_type.instance_type]["gpu_name"] == "H100"


def test_availability_list(availabilities):
    spot = True
    assert ("FIN-01", spot, "1A100.22V") in _make_availability_list(spot, availabilities)


def list_available_instances(availabilities, instance_type):
    spot = True
    instance_map = make_instance_map([instance_type])
    availability_list = _make_availability_list(spot=spot, availabilities=availabilities)

    result = _make_list_available_instances(availability_list, instance_map)
    assert len(result) == 1
    assert result[0]["spot"] == spot


def test_gpu_name(caplog):
    assert gpu_name("1x H100 SXM5 80GB") == "H100"
    assert gpu_name("") is None
    assert gpu_name(None) is None

    with caplog.at_level(logging.WARNING):
        gpu_name("1x H200 SXM5 80GB")
    assert "There is no '1x H200 SXM5 80GB' in gpu_map" in caplog.text
