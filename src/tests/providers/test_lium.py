import copy

import pytest
import requests

from gpuhunt._internal.constraints import find_accelerators
from gpuhunt._internal.models import AcceleratorVendor
from gpuhunt.providers.lium import NODES_URL, LiumProvider, get_dstack_gpu_name, get_location

# Recorded from https://lium.io/api/public/v1/nodes on 2026-09-10 (trimmed to four nodes).
RECORDED_FEED = {
    "generated_at": "2026-09-10T07:37:02.247559Z",
    "count": 4,
    "nodes": [
        {
            "id": "ebdfdbe2-875a-4caa-8eea-1ab5c2a763e1",
            "gpu_model": "RTX 5090",
            "gpu_count": 8,
            "available_gpu_count": 8,
            "price_per_gpu_hour": 0.5,
            "min_rentable_gpu_count": 8,
            "gpu_memory_gb": 32,
            "cpu_count": 384,
            "ram_gb": 251,
            "disk_gb": 935,
            "tier": "secure",
            "country": "Japan",
            "country_code": "JP",
            "region": "13",
            "city": "Chiyoda City",
            "rent_url": "https://lium.io/browse-pods/ebdfdbe2-875a-4caa-8eea-1ab5c2a763e1",
            "price_per_node_hour": 4.0,
        },
        {
            "id": "f092246d-5704-4c9e-b57f-7c0710a06deb",
            "gpu_model": "H200",
            "gpu_count": 8,
            "available_gpu_count": 5,
            "price_per_gpu_hour": 3.99,
            "min_rentable_gpu_count": 1,
            "gpu_memory_gb": 140,
            "cpu_count": 192,
            "ram_gb": 2015,
            "disk_gb": 3575,
            "tier": "secure",
            "country": "United States",
            "country_code": "US",
            "region": "MS",
            "city": "Jackson",
            "rent_url": "https://lium.io/browse-pods/f092246d-5704-4c9e-b57f-7c0710a06deb",
            "price_per_node_hour": 31.92,
        },
        {
            "id": "45af0308-a13c-497f-93ea-ab06144eee29",
            "gpu_model": "RTX 5090",
            "gpu_count": 1,
            "available_gpu_count": 1,
            "price_per_gpu_hour": 0.5,
            "min_rentable_gpu_count": 1,
            "gpu_memory_gb": 32,
            "cpu_count": 24,
            "ram_gb": 62,
            "disk_gb": 1832,
            "tier": "spot",
            "country": "United States",
            "country_code": "US",
            "region": "MN",
            "city": "Minneapolis",
            "rent_url": "https://lium.io/browse-pods/45af0308-a13c-497f-93ea-ab06144eee29",
            "price_per_node_hour": 0.5,
        },
        {
            "id": "6780e503-6861-4dca-bdec-ca9dcf37e8d0",
            "gpu_model": "A100-SXM4-80GB",
            "gpu_count": 8,
            "available_gpu_count": 8,
            "price_per_gpu_hour": 1.23,
            "min_rentable_gpu_count": 8,
            "gpu_memory_gb": 80,
            "cpu_count": 240,
            "ram_gb": 1772,
            "disk_gb": 19383,
            "tier": "secure",
            "country": "United States",
            "country_code": "US",
            "region": "VA",
            "city": "Ashburn",
            "rent_url": "https://lium.io/browse-pods/6780e503-6861-4dca-bdec-ca9dcf37e8d0",
            "price_per_node_hour": 9.84,
        },
    ],
}


@pytest.fixture
def feed() -> dict:
    return copy.deepcopy(RECORDED_FEED)


def make_node(**overrides) -> dict:
    node = copy.deepcopy(RECORDED_FEED["nodes"][2])
    node.update(overrides)
    return node


class TestGet:
    def test_one_offer_per_rentable_gpu_count(self, requests_mock, feed):
        requests_mock.get(NODES_URL, json=feed)

        offers = LiumProvider().get()

        # 8x RTX 5090 (min 8) -> 1, 8x H200 with 5 free (min 1) -> 5, 1x RTX 5090 -> 1,
        # 8x A100 (min 8) -> 1
        assert len(offers) == 8
        h200 = [o for o in offers if o.instance_name == "f092246d-5704-4c9e-b57f-7c0710a06deb"]
        assert [o.gpu_count for o in h200] == [1, 2, 3, 4, 5]
        assert all(o.gpu_name == "H200" for o in h200)

    def test_offers_sorted_by_price(self, requests_mock, feed):
        requests_mock.get(NODES_URL, json=feed)

        offers = LiumProvider().get()

        prices = [o.price for o in offers]
        assert prices == sorted(prices)

    def test_whole_node_offer(self, requests_mock, feed):
        requests_mock.get(NODES_URL, json=feed)

        offers = LiumProvider().get()

        offer = next(
            o for o in offers if o.instance_name == "ebdfdbe2-875a-4caa-8eea-1ab5c2a763e1"
        )
        assert offer.provider == "lium"
        assert offer.location == "jp-13"
        assert offer.price == 4.0  # 8 GPUs x $0.5 per GPU-hour
        assert offer.cpu == 384
        assert offer.memory == 251
        assert offer.disk_size == 935
        assert offer.gpu_vendor == AcceleratorVendor.NVIDIA
        assert offer.gpu_count == 8
        assert offer.gpu_name == "RTX5090"
        assert offer.gpu_memory == 32
        assert offer.spot is False
        assert offer.flags == []
        assert offer.provider_data == {
            "rent_url": "https://lium.io/browse-pods/ebdfdbe2-875a-4caa-8eea-1ab5c2a763e1",
        }

    def test_partial_rental_gets_proportional_resources(self, requests_mock, feed):
        requests_mock.get(NODES_URL, json=feed)

        offers = LiumProvider().get()

        h200 = {
            o.gpu_count: o
            for o in offers
            if o.instance_name == "f092246d-5704-4c9e-b57f-7c0710a06deb"
        }
        assert h200[1].price == 3.99
        assert h200[1].cpu == 24  # 192 / 8
        assert h200[1].memory == 251.88  # 2015 / 8
        assert h200[1].disk_size == 446.88  # 3575 / 8
        assert h200[5].price == 19.95
        assert h200[5].cpu == 120
        assert h200[5].memory == 1259.38
        # The feed reports 140 GB for a 141 GB card; the known size wins.
        assert all(o.gpu_memory == 141 for o in h200.values())

    def test_spot_tier(self, requests_mock, feed):
        requests_mock.get(NODES_URL, json=feed)

        offers = LiumProvider().get()

        spot = [o for o in offers if o.spot]
        assert [o.instance_name for o in spot] == ["45af0308-a13c-497f-93ea-ab06144eee29"]
        assert spot[0].location == "us-mn"

    def test_gpu_name_normalized(self, requests_mock, feed):
        requests_mock.get(NODES_URL, json=feed)

        offers = LiumProvider().get()

        a100 = next(o for o in offers if o.instance_name == "6780e503-6861-4dca-bdec-ca9dcf37e8d0")
        assert a100.gpu_name == "A100"
        assert a100.gpu_memory == 80

    def test_skips_node_with_missing_fields(self, requests_mock, caplog):
        requests_mock.get(
            NODES_URL,
            json={"nodes": [make_node(price_per_gpu_hour=None), make_node(gpu_model=None)]},
        )

        assert LiumProvider().get() == []
        assert "missing price_per_gpu_hour" in caplog.text
        assert "missing gpu_model" in caplog.text

    def test_skips_node_without_location(self, requests_mock, caplog):
        requests_mock.get(NODES_URL, json={"nodes": [make_node(country_code=None)]})

        assert LiumProvider().get() == []
        assert "no location" in caplog.text

    def test_location_without_region(self, requests_mock):
        requests_mock.get(NODES_URL, json={"nodes": [make_node(region=None)]})

        offers = LiumProvider().get()

        assert [o.location for o in offers] == ["us"]

    def test_skips_node_when_fewer_gpus_free_than_min_order(self, requests_mock):
        requests_mock.get(
            NODES_URL,
            json={
                "nodes": [make_node(gpu_count=8, available_gpu_count=3, min_rentable_gpu_count=4)]
            },
        )

        assert LiumProvider().get() == []

    def test_missing_min_rentable_defaults_to_one(self, requests_mock):
        requests_mock.get(
            NODES_URL,
            json={
                "nodes": [
                    make_node(gpu_count=2, available_gpu_count=2, min_rentable_gpu_count=None)
                ]
            },
        )

        offers = LiumProvider().get()

        assert [o.gpu_count for o in offers] == [1, 2]

    def test_missing_disk_size(self, requests_mock):
        requests_mock.get(NODES_URL, json={"nodes": [make_node(disk_gb=None)]})

        offers = LiumProvider().get()

        assert [o.disk_size for o in offers] == [None]

    def test_skips_unknown_gpu_model(self, requests_mock, caplog):
        requests_mock.get(
            NODES_URL, json={"nodes": [make_node(gpu_model="RTX 4090 D"), make_node()]}
        )

        offers = LiumProvider().get()

        assert [o.gpu_name for o in offers] == ["RTX5090"]
        assert "unknown gpu_model 'RTX 4090 D'" in caplog.text

    def test_skips_malformed_nodes(self, requests_mock, caplog):
        requests_mock.get(NODES_URL, json={"nodes": ["not a node", make_node()]})

        offers = LiumProvider().get()

        assert len(offers) == 1
        assert "malformed" in caplog.text

    def test_unexpected_response_is_rejected(self, requests_mock):
        requests_mock.get(NODES_URL, json={"nodes": {}})

        with pytest.raises(ValueError, match="Unexpected response"):
            LiumProvider().get()

    def test_http_error_is_propagated(self, requests_mock):
        requests_mock.get(NODES_URL, status_code=503)

        with pytest.raises(requests.HTTPError):
            LiumProvider().get()


def test_from_env_needs_no_credentials():
    assert isinstance(LiumProvider.from_env(), LiumProvider)


@pytest.mark.parametrize(
    ("lium_name", "expected"),
    [
        ("RTX 5090", "RTX5090"),
        ("RTX 4090", "RTX4090"),
        ("RTX 3090", "RTX3090"),
        ("RTX 4070 Ti SUPER", "RTX4070TiSUPER"),
        ("RTX PRO 6000 Blackwell Server Edition", "RTXPRO6000"),
        ("RTX PRO 6000 Blackwell Workstation Edition", "RTXPRO6000"),
        ("RTX PRO 4500 Blackwell", "RTXPRO4500"),
        ("RTX 6000 Ada Generation", "RTX6000Ada"),
        ("RTX 4000 Ada Generation", "RTX4000Ada"),
        ("RTX A6000", "A6000"),
        ("RTX A4000", "A4000"),
        ("Quadro RTX 6000", "RTX6000"),
        ("H100 80GB HBM3", "H100"),
        ("H100 PCIe", "H100"),
        ("H100 NVL", "H100NVL"),
        ("H200", "H200"),
        ("H200 NVL", "H200NVL"),
        ("B200", "B200"),
        ("B300 SXM6 AC", "B300"),
        ("A100-SXM4-80GB", "A100"),
        ("A100 80GB PCIe", "A100"),
        ("A10 Tensor Core GPU", "A10"),
        ("T4 Tensor Core GPU", "T4"),
        ("Tesla V100 Tensor Core GPU", "V100"),
        ("Tesla P100", "P100"),
        ("L40S", "L40S"),
        ("L4", "L4"),
    ],
)
def test_get_dstack_gpu_name(lium_name, expected):
    assert get_dstack_gpu_name(lium_name) == expected
    # Every name in the table must be one an offer can be published with
    assert find_accelerators(names=[expected], vendors=[AcceleratorVendor.NVIDIA])


def test_get_location():
    assert get_location("US", "CA") == "us-ca"
    assert get_location("JP", "13") == "jp-13"
    assert get_location("DE", None) == "de"
    assert get_location("DE", "") == "de"
    assert get_location(None, "CA") == ""
