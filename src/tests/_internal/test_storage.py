import io

import pytest

from gpuhunt._internal import storage
from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, CPUArchitecture


def catalog_item(**kwargs) -> CatalogItem:
    defaults = dict(
        provider="test",
        instance_name="test-instance",
        location="eu-west-1",
        price=1.0,
        cpu=1,
        memory=32.0,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=None,
        spot=False,
        disk_size=None,
    )
    return CatalogItem(**{**defaults, **kwargs})


def row(**kwargs) -> dict[str, str]:
    defaults = {
        "instance_name": "test-instance",
        "location": "eu-west-1",
        "price": "1.0",
        "cpu": "1",
        "memory": "32.0",
        "gpu_count": "0",
        "gpu_name": "",
        "gpu_memory": "",
        "spot": "False",
        "disk_size": "",
    }
    return {**defaults, **kwargs}


class TestItemToRow:
    def test_all_fields(self) -> None:
        item = catalog_item(
            cpu_arch=CPUArchitecture.ARM,
            gpu_vendor=AcceleratorVendor.NVIDIA,
            gpu_count=1,
            gpu_name="A10",
            gpu_memory=24.0,
            disk_size=100.0,
            flags=["f1", "f2"],
            provider_data={"custom_prop": 42},
        )
        assert storage.item_to_row(item) == {
            "instance_name": "test-instance",
            "location": "eu-west-1",
            "price": "1.0",
            "cpu": "1",
            "memory": "32.0",
            "gpu_count": "1",
            "gpu_name": "A10",
            "gpu_memory": "24.0",
            "spot": "False",
            "disk_size": "100.0",
            "gpu_vendor": "nvidia",
            "flags": "f1 f2",
            "cpu_arch": "arm",
            "provider_data": '{"custom_prop": 42}',
        }

    def test_unset_optionals_are_empty(self) -> None:
        item_row = storage.item_to_row(catalog_item())
        assert [item_row[f] for f in ("gpu_name", "gpu_memory", "disk_size", "gpu_vendor")] == [
            "",
            "",
            "",
            "",
        ]


class TestItemFromRow:
    def test_missing_required_field_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            storage.item_from_row(row(price=""), provider="test")

    def test_defaults_for_columns_missing_in_historical_catalogs(self) -> None:
        # No cpu_arch, gpu_vendor, flags, or provider_data columns.
        item = storage.item_from_row(row(), provider="test")
        assert item.cpu_arch == CPUArchitecture.X86
        assert item.gpu_vendor is None
        assert item.flags == []
        assert item.provider_data == {}

    def test_gpu_without_vendor_is_nvidia(self) -> None:
        item = storage.item_from_row(row(gpu_count="1", gpu_name="A100"), provider="test")
        assert item.gpu_vendor == AcceleratorVendor.NVIDIA

    def test_tpu_name_prefix_implies_google(self) -> None:
        item = storage.item_from_row(row(gpu_count="1", gpu_name="tpu-v3"), provider="test")
        assert item.gpu_name == "v3"
        assert item.gpu_vendor == AcceleratorVendor.GOOGLE


class TestLoad:
    def test_round_trip(self, tmp_path) -> None:
        items = [
            catalog_item(),
            catalog_item(
                price=12.0,
                gpu_count=8,
                gpu_name="H100",
                gpu_memory=80.0,
                gpu_vendor=AcceleratorVendor.NVIDIA,
                spot=True,
                disk_size=500.0,
                cpu_arch=CPUArchitecture.ARM,
                flags=["f1"],
                provider_data={"k": 1},
            ),
        ]
        path = str(tmp_path / "test.csv")
        storage.dump(items, path)
        with open(path) as f:
            assert list(storage.load(f, provider="test")) == items

    def test_malformed_row_is_skipped(self) -> None:
        header = ",".join(storage.CATALOG_V2_FIELDS)
        good = "i,loc,1.0,1,1.0,0,,,False,,,,x86,{}"
        bad = ",loc,1.0,1,1.0,0,,,False,,,,x86,{}"
        f = io.StringIO(f"{header}\n{bad}\n{good}\n")
        assert [i.instance_name for i in storage.load(f, provider="test")] == ["i"]
