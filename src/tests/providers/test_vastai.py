import pytest

import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog
from gpuhunt._internal.models import QueryFilter
from gpuhunt.providers.vastai import (
    VastAIProvider,
    bundles_url,
    compute_cap,
    get_dstack_gpu_name,
    get_vastai_gpu_names,
)


def make_offer(**overrides) -> dict:
    offer = {
        "id": 1,
        "cpu_cores": 32,
        "cpu_cores_effective": 4.0,
        "cpu_ram": 7859,
        "disk_space": 100.0,
        "storage_cost": 0.1,
        "dph_base": 0.2,
        "geolocation": "Bulgaria, BG",
        "gpu_name": "RTX 3090",
        "gpu_ram": 24576,
        "num_gpus": 1,
        "rentable": True,
        "rented": False,
    }
    offer.update(overrides)
    return offer


class TestGet:
    def test_keeps_fractional_memory(self, requests_mock):
        requests_mock.post(bundles_url, json={"offers": [make_offer()]})

        offers = VastAIProvider().get()

        assert [offer.memory for offer in offers] == [0.98]

    def test_skips_offers_with_less_than_one_cpu(self, requests_mock):
        requests_mock.post(bundles_url, json={"offers": [make_offer(cpu_cores_effective=0.5)]})

        assert VastAIProvider().get() == []

    def test_skips_offers_without_memory(self, requests_mock):
        requests_mock.post(bundles_url, json={"offers": [make_offer(cpu_ram=0)]})

        assert VastAIProvider().get() == []

    @pytest.mark.parametrize(
        ["min_cc", "max_cc", "expected"],
        [
            pytest.param((8, 0), None, {"gte": 800}, id="query-min-stricter"),
            pytest.param((5, 0), None, {"gte": 600}, id="extra-min-stricter"),
            pytest.param(None, (9, 0), {"gte": 600, "lte": 900}, id="query-max"),
        ],
    )
    def test_merges_extra_filters_keeping_stricter_bound(
        self,
        requests_mock,
        min_cc: tuple[int, int] | None,
        max_cc: tuple[int, int] | None,
        expected: dict,
    ):
        requests_mock.post(bundles_url, json={"offers": []})
        provider = VastAIProvider(extra_filters={"compute_cap": {"gte": 600}})

        provider.get(QueryFilter(min_compute_capability=min_cc, max_compute_capability=max_cc))

        assert requests_mock.last_request.json()["compute_cap"] == expected

    def test_filters_offers_by_compute_cap(self, requests_mock):
        requests_mock.post(
            bundles_url,
            json={
                "offers": [make_offer(id=1, compute_cap=750), make_offer(id=2, compute_cap=860)]
            },
        )

        offers = VastAIProvider().get(QueryFilter(min_compute_capability=(8, 0)))

        assert [offer.instance_name for offer in offers] == ["2"]

    def test_lists_b300_pc_as_b300(self, requests_mock):
        requests_mock.post(
            bundles_url,
            json={
                "offers": [
                    make_offer(id=1, gpu_name="B300", gpu_ram=275040),
                    make_offer(id=2, gpu_name="B300 PC", gpu_ram=275040),
                ]
            },
        )

        offers = VastAIProvider().get(QueryFilter(gpu_name=["B300"]))

        assert requests_mock.last_request.json()["gpu_name"] == {"in": ["B300", "B300 PC"]}
        assert [(o.instance_name, o.gpu_name, o.gpu_memory) for o in offers] == [
            ("1", "B300", 270.0),
            ("2", "B300", 270.0),
        ]


def test_make_filters_defaults_to_datacenter_only():
    filters = VastAIProvider(community_cloud=False).make_filters(QueryFilter())
    assert filters["datacenter"] == {"eq": True}
    assert "external" not in filters


def test_make_filters_does_not_constrain_scope_when_community_cloud_enabled():
    filters = VastAIProvider(community_cloud=True).make_filters(QueryFilter())
    assert "datacenter" not in filters
    assert "external" not in filters


@pytest.mark.parametrize(
    ["cc", "expected"],
    [
        pytest.param((7, 0), 700, id="7.0"),
        pytest.param((7, 5), 750, id="7.5"),
        pytest.param((12, 0), 1200, id="12.0"),
    ],
)
def test_compute_cap(cc: tuple[int, int], expected: int):
    assert compute_cap(cc) == expected


@pytest.mark.parametrize(
    ["vastai_name", "expected"],
    [
        ("RTX A5000", "A5000"),
        ("Tesla V100", "V100"),
        ("A100 SXM4", "A100"),
        ("H100 NVL", "H100NVL"),
        ("B300", "B300"),
        ("B300 PC", "B300"),
        ("RTX 2060S", "RTX2060SUPER"),
        ("RTX 2070S", "RTX2070SUPER"),
        ("RTX 4070S", "RTX4070SUPER"),
        ("RTX 4070S Ti", "RTX4070TiSUPER"),
        ("RTX 4080S", "RTX4080SUPER"),
        ("Q RTX 4000", "RTX4000"),
        ("Q RTX 5000", "RTX5000"),
        ("Q RTX 6000", "RTX6000"),
        ("Q RTX 8000", "RTX8000"),
    ],
)
def test_get_dstack_gpu_name(vastai_name: str, expected: str):
    assert get_dstack_gpu_name(vastai_name) == expected


def test_get_vastai_gpu_names_includes_b300_pc():
    assert get_vastai_gpu_names("B300") == ["B300", "B300 PC"]


class TestGpuMemory:
    @pytest.mark.parametrize(
        ("name", "memory_mib", "expected", "warns"),
        [
            ("A100 SXM4", 39936, 40, False),
            ("A100 SXM4", 80896, 80, False),
            ("RTX 4090", 49140, 49140 / 1024, True),
            ("unregistered", 17777, 17777 / 1024, True),
            ("RTX A2000", 6138, 6, False),
            ("RTX A2000", 12282, 12, False),
            ("RTX PRO 5000", 48935, 48, False),
            ("RTX PRO 5000", 73415, 72, False),
        ],
    )
    def test_reported_vram_matches_exact_filters(
        self, requests_mock, name, memory_mib, expected, warns, caplog, monkeypatch
    ):
        requests_mock.post(
            bundles_url, json={"offers": [make_offer(gpu_name=name, gpu_ram=memory_mib)]}
        )
        monkeypatch.setattr(internal_catalog, "ONLINE_PROVIDERS", ["vastai"])
        monkeypatch.setattr(internal_catalog, "OFFLINE_PROVIDERS", [])
        catalog = Catalog(auto_reload=False)
        catalog.add_provider(VastAIProvider())
        unfiltered = catalog.query(provider="vastai")
        assert unfiltered and unfiltered[0].gpu_memory == expected
        assert (
            catalog.query(provider="vastai", min_gpu_memory=expected, max_gpu_memory=expected)
            == unfiltered
        )
        assert requests_mock.last_request.json()["gpu_ram"] == pytest.approx(
            {"gte": expected * 0.93 * 1024, "lte": expected * 1.07 * 1024}
        )
        assert bool(caplog.records) == warns
        if warns:
            assert "provider=vastai, offer=1, location=bg-bulgaria" in caplog.text
            assert f"reported VRAM={memory_mib / 1024} GiB per GPU" in caplog.text
            assert "keeping reported VRAM" in caplog.text
