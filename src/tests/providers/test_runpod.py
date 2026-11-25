from urllib.parse import parse_qs, urlparse

import pytest
import requests

import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog
from gpuhunt._internal.models import AcceleratorVendor, QueryFilter
from gpuhunt.providers import runpod as runpod_module
from gpuhunt.providers.runpod import API_URL, RunpodProvider

GPUS_URL = f"{API_URL}/catalog/gpus"
CPUS_URL = f"{API_URL}/catalog/cpus"


def make_gpu(
    id: str = "NVIDIA H100 80GB HBM3",
    name: str = "H100 SXM",
    manufacturer: str = "NVIDIA",
    memory: int = 80,
    secure: bool = True,
    secure_price: float = 2.5,
    max_count: int = 8,
    min_vcpu: int | None = 16,
    min_ram: int | None = 128,
    data_centers: dict[str, str] | None = None,
) -> dict:
    if data_centers is None:
        data_centers = {"EU-RO-1": "HIGH", "US-KS-2": "LOW"}
    availability = "HIGH" if data_centers else "NONE"
    gpu = {
        "id": id,
        "name": name,
        "manufacturer": manufacturer,
        "memory": memory,
        "pool": None,
        "secure": secure,
        "community": False,
        "price": {"community": 0.5, "secure": secure_price},
        "maxCount": {"community": 0, "secure": max_count},
        "availability": availability,
    }
    if data_centers:
        gpu["minVcpuCount"] = min_vcpu
        gpu["minRam"] = min_ram
        gpu["dataCenters"] = [
            {"id": dc, "name": dc, "availability": av} for dc, av in data_centers.items()
        ]
    return gpu


def make_cpu(
    id: str = "cpu3c",
    group: str = "CPU3",
    min_vcpu: int = 2,
    max_vcpu: int = 32,
    ram_per_vcpu: int = 2,
    price_per_vcpu: float = 0.03,
    data_centers: dict[str, str] | None = None,
) -> dict:
    if data_centers is None:
        data_centers = {"EU-RO-1": "HIGH"}
    return {
        "id": id,
        "name": "Compute-Optimized",
        "group": group,
        "vcpu": {"min": min_vcpu, "max": max_vcpu},
        "ramGbPerVcpu": ram_per_vcpu,
        "price": {"securePerVcpu": price_per_vcpu, "serverlessPerVcpu": 0.036},
        "availability": "HIGH" if data_centers else "NONE",
        "dataCenters": [
            {"id": dc, "name": dc, "availability": av} for dc, av in data_centers.items()
        ],
    }


def params(request) -> dict[str, str]:
    # `request.qs` is lowercased by requests_mock
    return {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}


@pytest.fixture
def api(requests_mock):
    """
    Serves `gpus_by_count[count]` and `cpus_by_vcpu[vcpuCount]`. Unset keys get empty lists.
    """

    class Api:
        gpus_by_count: dict[int, list[dict]] = {}
        cpus_by_vcpu: dict[int, list[dict]] = {}
        failing_gpu_counts: set[int] = set()

        # Requests after the first one are concurrent, so they are sorted for stable asserts
        def gpu_requests(self) -> list[dict[str, str]]:
            return sorted(
                (params(r) for r in requests_mock.request_history if r.url.startswith(GPUS_URL)),
                key=lambda r: int(r["count"]),
            )

        def cpu_requests(self) -> list[dict[str, str]]:
            return sorted(
                (params(r) for r in requests_mock.request_history if r.url.startswith(CPUS_URL)),
                key=lambda r: int(r["vcpuCount"]),
            )

    api = Api()

    def gpus(request, context):
        count = int(params(request)["count"])
        if count in api.failing_gpu_counts:
            context.status_code = 400
            return {"error": "bad request"}
        return {"gpus": api.gpus_by_count.get(count, [])}

    def cpus(request, context):
        return {"cpus": api.cpus_by_vcpu.get(int(params(request)["vcpuCount"]), [])}

    requests_mock.get(GPUS_URL, json=gpus)
    requests_mock.get(CPUS_URL, json=cpus)
    return api


def get(query_filter: QueryFilter):
    return RunpodProvider(api_key="test").get(query_filter=query_filter)


class TestGpuItems:
    def test_makes_item_per_available_data_center(self, api):
        api.gpus_by_count = {
            1: [make_gpu(data_centers={"EU-RO-1": "HIGH", "US-KS-2": "NONE", "CA-MTL-1": "LOW"})]
        }
        items = get(QueryFilter(min_gpu_count=1, max_gpu_count=1))
        assert [(i.location, i.gpu_count, i.price) for i in items] == [
            ("EU-RO-1", 1, 2.5),
            ("CA-MTL-1", 1, 2.5),
        ]
        item = items[0]
        assert item.provider == "runpod"
        assert item.instance_name == "NVIDIA H100 80GB HBM3"
        assert item.gpu_vendor == AcceleratorVendor.NVIDIA
        assert item.gpu_name == "H100"
        assert item.gpu_memory == 80
        assert (item.cpu, item.memory) == (16, 128)
        assert item.disk_size is None
        assert item.spot is False
        assert item.flags == []

    def test_requests_per_gpu_resource_filters(self, api):
        api.gpus_by_count = {
            1: [make_gpu(min_vcpu=30, min_ram=120)],
            2: [make_gpu(min_vcpu=32, min_ram=128)],
            3: [make_gpu(min_vcpu=48, min_ram=128)],
        }
        items = get(QueryFilter(min_gpu_count=1, max_gpu_count=3, min_cpu=30, min_memory=120))
        assert [
            (r["count"], r["minVcpuCountPerGpu"], r["minRamPerGpu"]) for r in api.gpu_requests()
        ] == [("1", "30", "120"), ("2", "15", "60"), ("3", "10", "40")]
        for r in api.gpu_requests():
            assert r["include"] == "AVAILABILITY"
            assert r["product"] == "POD"
            assert r["cloud"] == "SECURE"
            assert r["minCudaVersion"] == RunpodProvider.MIN_CUDA_VERSION
        by_count = {i.gpu_count: (i.cpu, i.memory, i.price) for i in items}
        assert by_count == {1: (30, 120, 2.5), 2: (32, 128, 5.0), 3: (48, 128, 7.5)}

    def test_omits_unset_resource_filters(self, api):
        api.gpus_by_count = {1: [make_gpu(max_count=1)]}
        get(QueryFilter(min_gpu_count=1))
        [request] = api.gpu_requests()
        assert "minVcpuCountPerGpu" not in request
        assert "minRamPerGpu" not in request

    def test_requests_counts_up_to_max_count_of_matching_types(self, api):
        api.gpus_by_count = {
            1: [
                make_gpu(max_count=4),
                make_gpu(id="NVIDIA A40", name="A40", memory=48, max_count=10),
            ]
        }
        get(QueryFilter(gpu_name=["H100"]))
        assert [r["count"] for r in api.gpu_requests()] == ["1", "2", "3", "4"]

    def test_requests_counts_up_to_max_gpu_count(self, api):
        api.gpus_by_count = {2: [make_gpu(max_count=8)]}
        get(QueryFilter(min_gpu_count=2, max_gpu_count=3))
        assert [r["count"] for r in api.gpu_requests()] == ["2", "3"]

    def test_skips_counts_above_type_max_count(self, api):
        a40 = make_gpu(id="NVIDIA A40", name="A40", memory=48, max_count=2)
        api.gpus_by_count = {1: [make_gpu(max_count=4), a40], 3: [make_gpu(max_count=4), a40]}
        items = get(QueryFilter())
        assert {(i.gpu_name, i.gpu_count) for i in items} == {("H100", 1), ("A40", 1), ("H100", 3)}

    @pytest.mark.parametrize(
        "gpu",
        [
            pytest.param(make_gpu(secure=False), id="community-only"),
            pytest.param(make_gpu(max_count=0), id="no-secure-count"),
            pytest.param(make_gpu(secure_price=0), id="no-secure-price"),
            pytest.param(make_gpu(data_centers={}), id="no-availability"),
            pytest.param(make_gpu(data_centers={"EU-RO-1": "NONE"}), id="no-dc-availability"),
            pytest.param(make_gpu(min_vcpu=None), id="no-min-vcpu"),
            pytest.param(
                make_gpu(
                    id="NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
                    name="PRO 6000 MIG 24GB",
                    memory=24,
                ),
                id="mig",
            ),
            pytest.param(
                make_gpu(id="unknown", name="unknown", manufacturer="UNKNOWN"), id="unknown"
            ),
        ],
    )
    def test_skips_unusable_types(self, api, gpu):
        api.gpus_by_count = {1: [gpu]}
        assert get(QueryFilter(min_gpu_count=1, max_gpu_count=1)) == []

    def test_amd(self, api):
        api.gpus_by_count = {
            1: [
                make_gpu(
                    id="AMD Instinct MI300X OAM", name="MI300X", manufacturer="AMD", memory=192
                )
            ]
        }
        item = get(QueryFilter(min_gpu_count=1, max_gpu_count=1))[0]
        assert (item.gpu_vendor, item.gpu_name, item.gpu_memory) == (
            AcceleratorVendor.AMD,
            "MI300X",
            192,
        )

    def test_failed_count_does_not_fail_other_counts(self, api):
        api.gpus_by_count = {1: [make_gpu(max_count=3)], 3: [make_gpu(max_count=3)]}
        api.failing_gpu_counts = {2}
        items = get(QueryFilter(min_gpu_count=1))
        assert {i.gpu_count for i in items} == {1, 3}

    def test_raises_if_all_requests_fail(self, api):
        api.failing_gpu_counts = {1}
        with pytest.raises(requests.HTTPError):
            get(QueryFilter(min_gpu_count=1))


class TestCpuItems:
    def test_makes_item_per_vcpu_count_and_data_center(self, api):
        api.cpus_by_vcpu = {
            vcpu: [
                make_cpu(),
                make_cpu(
                    id="cpu5m",
                    group="CPU5",
                    ram_per_vcpu=8,
                    price_per_vcpu=0.065,
                    data_centers={"EU-RO-1": "LOW", "US-KS-2": "NONE"},
                ),
            ]
            for vcpu in (2, 4, 8, 16, 32)
        }
        api.cpus_by_vcpu[32][0]["dataCenters"][0]["availability"] = "NONE"
        items = get(QueryFilter(min_gpu_count=0, max_gpu_count=0))
        assert [r["vcpuCount"] for r in api.cpu_requests()] == ["2", "4", "8", "16", "32"]
        assert api.gpu_requests() == []
        for r in api.cpu_requests():
            assert r["include"] == "AVAILABILITY"
            assert r["product"] == "POD"
        assert sorted((i.instance_name, i.location) for i in items) == sorted(
            [(f"cpu3c-{v}-{v * 2}", "EU-RO-1") for v in (2, 4, 8, 16)]
            + [(f"cpu5m-{v}-{v * 8}", "EU-RO-1") for v in (2, 4, 8, 16, 32)]
        )
        item = next(i for i in items if i.instance_name == "cpu5m-16-128")
        assert (item.cpu, item.memory, item.disk_size) == (16, 128, 240.0)
        assert item.price == pytest.approx(0.065 * 16)
        assert (item.gpu_count, item.gpu_name, item.gpu_memory, item.gpu_vendor) == (
            0,
            None,
            None,
            None,
        )
        assert item.spot is False
        assert item.flags == []
        assert next(i for i in items if i.instance_name == "cpu3c-2-4").disk_size == 20.0

    def test_requests_vcpu_counts_within_filter(self, api):
        api.cpus_by_vcpu = {4: [make_cpu(max_vcpu=32)]}
        get(QueryFilter(min_gpu_count=0, min_cpu=3, max_cpu=20))
        assert [r["vcpuCount"] for r in api.cpu_requests()] == ["4", "8", "16"]

    def test_requests_vcpu_counts_up_to_flavor_max(self, api):
        api.cpus_by_vcpu = {2: [make_cpu(max_vcpu=8)]}
        get(QueryFilter(min_gpu_count=0, max_gpu_count=0))
        assert [r["vcpuCount"] for r in api.cpu_requests()] == ["2", "4", "8"]

    @pytest.mark.parametrize(
        "cpu",
        [
            pytest.param(make_cpu(group="CPU9"), id="unknown-disk-limit"),
            pytest.param(make_cpu(min_vcpu=4), id="below-min-vcpu"),
            pytest.param(make_cpu(price_per_vcpu=0), id="no-price"),
            pytest.param(make_cpu(data_centers={}), id="no-availability"),
        ],
    )
    def test_skips_unusable_flavors(self, api, cpu):
        api.cpus_by_vcpu = {2: [cpu]}
        assert get(QueryFilter(min_gpu_count=0, max_gpu_count=0, max_cpu=2)) == []

    @pytest.mark.parametrize(
        "query_filter",
        [
            QueryFilter(min_gpu_count=1),
            QueryFilter(gpu_name=["H100"]),
            QueryFilter(gpu_vendor=AcceleratorVendor.NVIDIA),
            QueryFilter(min_gpu_memory=24),
        ],
    )
    def test_not_requested_for_gpu_queries(self, api, query_filter):
        get(query_filter)
        assert api.cpu_requests() == []

    def test_requested_with_gpu_filters_if_zero_gpus_allowed(self, api):
        get(QueryFilter(min_gpu_count=0, gpu_name=["H100"]))
        assert api.cpu_requests()


@pytest.mark.parametrize(
    "query_filter",
    [QueryFilter(spot=True), QueryFilter(cpu_arch=runpod_module.CPUArchitecture.ARM)],
)
def test_no_requests_for_unsupported_queries(api, requests_mock, query_filter):
    assert get(query_filter) == []
    assert requests_mock.request_history == []


def test_catalog_query(api, monkeypatch):
    api.gpus_by_count = {1: [make_gpu(secure_price=3)]}
    api.cpus_by_vcpu = {2: [make_cpu(max_vcpu=2)]}
    monkeypatch.setattr(internal_catalog, "ONLINE_PROVIDERS", ["runpod"])
    monkeypatch.setattr(internal_catalog, "OFFLINE_PROVIDERS", [])
    catalog = Catalog(auto_reload=False)
    catalog.add_provider(RunpodProvider(api_key="test"))
    items = catalog.query(provider="runpod", max_price=1)
    assert [i.instance_name for i in items] == ["cpu3c-2-4"]
    assert api.gpu_requests()[0]["count"] == "1"


def test_sends_api_key(api, requests_mock):
    RunpodProvider(api_key="secret").get(QueryFilter(min_gpu_count=1, max_gpu_count=1))
    assert requests_mock.last_request.headers["Authorization"] == "Bearer secret"


def test_from_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    assert RunpodProvider.from_env().api_key == "secret"


@pytest.mark.parametrize(("n", "expected"), [(1, 1), (2, 2), (3, 4), (16, 16), (17, 32)])
def test_next_power_of_two(n, expected):
    assert runpod_module._next_power_of_two(n) == expected


def test_b300_mig_is_not_treated_as_full_gpu():
    assert runpod_module._get_nvidia_gpu_name("B300 1g.35gb") is None
