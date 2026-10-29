import pytest

from gpuhunt import CatalogItem, QueryFilter
from gpuhunt._internal.constraints import (
    correct_gpu_memory_gib,
    find_accelerators,
    get_compute_capability,
    get_gpu_vendor,
    matches,
)
from gpuhunt._internal.models import AcceleratorVendor, NvidiaGPUInfo


@pytest.fixture
def item() -> CatalogItem:
    return CatalogItem(
        instance_name="large",
        location="us-east-1",
        price=1.2,
        cpu=16,
        memory=64.0,
        gpu_vendor=AcceleratorVendor.NVIDIA,
        gpu_count=1,
        gpu_name="A100",
        gpu_memory=40.0,
        spot=False,
        provider="aws",
        disk_size=None,
    )


@pytest.fixture
def cpu_items() -> list[CatalogItem]:
    verda = CatalogItem(
        instance_name="CPU.120V.480G",
        location="ICE-01",
        price=3.0,
        cpu=120,
        memory=480.0,
        gpu_vendor=None,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=None,
        spot=False,
        provider="verda",
        disk_size=None,
    )
    aws = CatalogItem(
        instance_name="c5.12xlarge",
        location="us-east-2",
        price=2.04,
        cpu=48,
        memory=96.0,
        gpu_vendor=None,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=None,
        spot=False,
        provider="aws",
        disk_size=None,
    )
    return [verda, aws]


class TestMatches:
    def test_empty(self, item: CatalogItem):
        assert matches(item, QueryFilter())

    def test_cpu(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_cpu=16))
        assert matches(item, QueryFilter(max_cpu=16))
        assert not matches(item, QueryFilter(min_cpu=32))
        assert not matches(item, QueryFilter(max_cpu=8))

    def test_memory(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_memory=64.0))
        assert matches(item, QueryFilter(max_memory=64.0))
        assert not matches(item, QueryFilter(min_memory=128.0))
        assert not matches(item, QueryFilter(max_memory=32.0))

    def test_disk_size(self, item: CatalogItem):
        item.disk_size = 100.5
        assert matches(item, QueryFilter(min_disk_size=100.5))
        assert matches(item, QueryFilter(max_disk_size=100.5))
        assert not matches(item, QueryFilter(min_disk_size=100.6))
        assert not matches(item, QueryFilter(max_disk_size=100.4))

    def test_gpu_vendor_nvidia(self, item: CatalogItem):
        assert matches(item, QueryFilter(gpu_vendor=AcceleratorVendor.NVIDIA))
        assert not matches(item, QueryFilter(gpu_vendor=AcceleratorVendor.AMD))

    def test_gpu_vendor_amd(self, item: CatalogItem):
        item.gpu_vendor = AcceleratorVendor.AMD
        assert matches(item, QueryFilter(gpu_vendor=AcceleratorVendor.AMD))
        assert not matches(item, QueryFilter(gpu_vendor=AcceleratorVendor.NVIDIA))

    def test_gpu_count(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_gpu_count=1))
        assert matches(item, QueryFilter(max_gpu_count=1))
        assert not matches(item, QueryFilter(min_gpu_count=2))
        assert not matches(item, QueryFilter(max_gpu_count=0))

    def test_gpu_memory(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_gpu_memory=40.0))
        assert matches(item, QueryFilter(max_gpu_memory=40.0))
        assert not matches(item, QueryFilter(min_gpu_memory=80.0))
        assert not matches(item, QueryFilter(max_gpu_memory=20.0))

    def test_gpu_name(self, item: CatalogItem):
        assert matches(item, QueryFilter(gpu_name=["a100"]))
        assert matches(item, QueryFilter(gpu_name=["A100"]))
        assert not matches(item, QueryFilter(gpu_name=["A10"]))

    def test_gpu_name_with_filter_setattr(self, item: CatalogItem):
        q = QueryFilter()
        q.gpu_name = ["a100"]
        assert matches(item, q)
        q.gpu_name = ["A100"]
        assert matches(item, q)
        q.gpu_name = ["A10"]
        assert not matches(item, q)

    def test_total_gpu_memory(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_total_gpu_memory=40.0))
        assert matches(item, QueryFilter(max_total_gpu_memory=40.0))
        assert not matches(item, QueryFilter(min_total_gpu_memory=80.0))
        assert not matches(item, QueryFilter(max_total_gpu_memory=20.0))

    def test_price(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_price=1.2))
        assert matches(item, QueryFilter(max_price=1.2))
        assert not matches(item, QueryFilter(min_price=1.3))
        assert not matches(item, QueryFilter(max_price=1.1))

    def test_spot(self, item: CatalogItem):
        assert matches(item, QueryFilter(spot=False))
        assert not matches(item, QueryFilter(spot=True))

    def test_compute_capability(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_compute_capability=(8, 0)))
        assert matches(item, QueryFilter(max_compute_capability=(8, 0)))
        assert not matches(item, QueryFilter(min_compute_capability=(8, 1)))
        assert not matches(item, QueryFilter(max_compute_capability=(7, 9)))

    def test_compute_capability_not_nvidia(self, item: CatalogItem):
        item.gpu_vendor = AcceleratorVendor.AMD
        assert not matches(item, QueryFilter(min_compute_capability=(8, 0)))
        assert not matches(item, QueryFilter(max_compute_capability=(8, 0)))

    def test_ti_gpu(self):
        item = CatalogItem(
            instance_name="large",
            location="us-east-1",
            price=1.2,
            cpu=16,
            memory=64.0,
            gpu_count=1,
            gpu_vendor=AcceleratorVendor.NVIDIA,
            gpu_name="RTX3060Ti",  # case-sensitive
            gpu_memory=8.0,
            spot=False,
            provider="aws",
            disk_size=None,
        )
        assert matches(item, QueryFilter(gpu_name=["RTX3060TI"]))

    def test_provider(self, cpu_items):
        assert matches(cpu_items[0], QueryFilter(provider=["verda"]))
        assert matches(cpu_items[0], QueryFilter(provider=["verda"]))
        assert not matches(cpu_items[0], QueryFilter(provider=["aws"]))

        assert matches(cpu_items[1], QueryFilter(provider=["aws"]))
        assert matches(cpu_items[1], QueryFilter(provider=["AWS"]))
        assert not matches(cpu_items[1], QueryFilter(provider=["verda"]))

    def test_provider_with_filter_setattr(self, cpu_items):
        q = QueryFilter()
        q.provider = ["verda"]
        assert matches(cpu_items[0], q)
        q.provider = ["verda"]
        assert matches(cpu_items[0], q)
        q.provider = ["aws"]
        assert not matches(cpu_items[0], q)

    @pytest.mark.parametrize(
        ("item_flags", "query_allowed_flags", "should_match"),
        [
            pytest.param([], ["a"], True, id="matches-if-no-flags"),
            pytest.param([], None, True, id="matches-if-no-flags-and-all-flags-allowed"),
            pytest.param(["a", "b"], None, True, id="matches-if-has-flags-and-all-flags-allowed"),
            pytest.param(
                ["a", "b"], ["a", "b", "c"], True, id="matches-if-has-flags-all-of-which-allowed"
            ),
            pytest.param(["a", "b"], ["a"], False, id="not-matches-if-some-flags-not-allowed"),
        ],
    )
    def test_flags(
        self,
        item: CatalogItem,
        item_flags: list[str],
        query_allowed_flags: list[str] | None,
        should_match: bool,
    ) -> None:
        item.flags = item_flags
        q = QueryFilter(allowed_flags=query_allowed_flags)
        assert matches(item, q) == should_match

    def test_matches_cpu_instances_with_zero_gpu_count_and_gpu_memory(self):
        cpu_item = CatalogItem(
            instance_name="large",
            location="us-east-1",
            price=1.2,
            cpu=16,
            memory=64.0,
            gpu_vendor=None,
            gpu_count=0,
            gpu_name=None,
            gpu_memory=None,
            spot=False,
            provider="aws",
            disk_size=None,
        )
        assert matches(cpu_item, QueryFilter(min_gpu_count=0, min_gpu_memory=24))


@pytest.mark.parametrize(
    ("gpu_name", "expected_memories_gib"),
    [
        ("n150", {12}),
        ("n300", {24}),
        ("tt-galaxy-wh", {12}),
        ("p100a", {28}),
        ("p150", {32}),
        ("p300", {32, 64}),
        ("tt-galaxy-bh", {32}),
    ],
)
def test_tenstorrent_accelerators(gpu_name: str, expected_memories_gib: set[int]) -> None:
    accelerators = find_accelerators(
        names=[gpu_name.upper()],
        vendors=[AcceleratorVendor.TENSTORRENT],
    )

    assert {accelerator.name for accelerator in accelerators} == {gpu_name}
    assert {accelerator.memory for accelerator in accelerators} == expected_memories_gib
    assert get_gpu_vendor(gpu_name.upper()) == AcceleratorVendor.TENSTORRENT


def test_rtx_pro_6000_accelerator() -> None:
    accelerators = find_accelerators(
        names=["RTXPRO6000"],
        vendors=[AcceleratorVendor.NVIDIA],
    )

    assert [accelerator.memory for accelerator in accelerators] == [96]
    assert get_compute_capability("RTXPRO6000") == (12, 0)
    assert get_gpu_vendor("RTXPRO6000") == AcceleratorVendor.NVIDIA


class TestCorrectGpuMemoryGib:
    @pytest.mark.parametrize(
        ("name", "memory_mib", "expected", "warns"),
        [
            ("H100NVL", 95830.0, 94, False),
            ("L40S", 46068.0, 48, False),
            ("A10G", 23028.0, 24, False),
            ("A10", 4096.0, 4, True),
            ("unknown", 8200.1, 8, True),
            ("b300", 288 * 1024, 270, False),
            ("A100", 39 * 1024, 40, False),
            ("A100", 40 * 1024, 40, False),
            ("A100", 79 * 1024, 80, False),
            ("A100", 80 * 1024, 80, False),
            ("A100", 60 * 1024, 60, True),
            ("P100", 12 * 1024, 12, True),
        ],
    )
    def test_normalizes_mib_to_whole_gib(self, name, memory_mib, expected, warns, caplog):
        memory_gib = correct_gpu_memory_gib(name, memory_mib)
        assert memory_gib == expected
        assert isinstance(memory_gib, int)
        assert bool(caplog.records) == warns
        if warns:
            assert f"using rounded VRAM={expected} GiB per GPU" in caplog.text
            assert name in caplog.text

    def test_unknown_model_warns_and_rounds(self, caplog):
        assert (
            correct_gpu_memory_gib(
                "unregistered", 17.25 * 1024, context="provider=example, offer=123"
            )
            == 17
        )
        assert caplog.messages == [
            "Unknown GPU unregistered (provider=example, offer=123): "
            "model is not in KNOWN_ACCELERATORS; reported VRAM=17.25 GiB per GPU; "
            "using rounded VRAM=17 GiB per GPU"
        ]

    def test_unmatched_capacity_warns_and_rounds(self, caplog):
        assert (
            correct_gpu_memory_gib(
                "RTX4090", 49140, context="provider=example, offer=123, location=eu"
            )
            == 48
        )
        assert caplog.messages == [
            "Cannot normalize VRAM for GPU RTX4090 "
            "(provider=example, offer=123, location=eu): reported VRAM=47.98828125 GiB per GPU; "
            "no registered capacity matches within ±7%; registered capacities=[24] GiB; "
            "using rounded VRAM=48 GiB per GPU"
        ]

    @pytest.mark.parametrize(("reported", "expected"), [(40, 40), (41.25, 41)])
    def test_overlapping_capacity_windows(self, reported, expected, mocker, caplog):
        mocker.patch(
            "gpuhunt._internal.constraints.find_accelerators",
            return_value=[
                NvidiaGPUInfo(name="test", memory=40, compute_capability=(8, 0)),
                NvidiaGPUInfo(name="test", memory=42, compute_capability=(8, 0)),
            ],
        )
        assert correct_gpu_memory_gib("test", reported * 1024) == expected
        assert bool(caplog.records) == (reported == 41.25)
        if reported == 41.25:
            assert "multiple registered capacities match within ±7%" in caplog.text
            assert "registered capacities=[40, 42] GiB" in caplog.text
            assert "using rounded VRAM=41 GiB per GPU" in caplog.text
