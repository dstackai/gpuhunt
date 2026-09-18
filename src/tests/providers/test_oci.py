import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests_mock
from oci.identity.models import Region

from gpuhunt._internal.models import AcceleratorVendor, CPUArchitecture
from gpuhunt.providers.oci import (
    COST_ESTIMATOR_URL_TEMPLATE,
    CostEstimatorDataError,
    CostEstimatorProductList,
    CostEstimatorShape,
    CostEstimatorShapeList,
    OCICredentials,
    OCIProvider,
    get_gpu_name,
    normalize_shape_name,
    shape_to_resources,
)

# Trimmed excerpts of the real Cost Estimator payload
# (https://www.oracle.com/a/ocom/docs/cloudestimator2/data/shapes.json).
# The `disk` product of BM.DenseIO.E5.128 reports a fractional qty of 81.6.
SHAPES = {
    "items": [
        {
            "name": "BM.DenseIO.E5.128",
            "hidden": False,
            "status": "ACTIVE",
            "allowPreemptible": False,
            "bundleMemoryQty": 1536,
            "gpuQty": None,
            "gpuMemoryQty": None,
            "processorType": {"value": "amd"},
            "shapeType": {"value": "bm"},
            "subType": {"value": "dense"},
            "products": [
                {"partNumber": "B98202", "qty": 128, "type": {"value": "ocpu"}},
                {"partNumber": "B98203", "qty": 1536, "type": {"value": "memory"}},
                {"partNumber": "B98204", "qty": 81.6, "type": {"value": "disk"}},
            ],
        },
        {
            "name": "BM.Standard.E4.128",
            "hidden": False,
            "status": "ACTIVE",
            "allowPreemptible": False,
            "bundleMemoryQty": 2048,
            "gpuQty": None,
            "gpuMemoryQty": None,
            "processorType": {"value": "amd"},
            "shapeType": {"value": "bm"},
            "subType": {"value": "standard"},
            "products": [
                {"partNumber": "B93113", "qty": 128, "type": {"value": "ocpu"}},
                {"partNumber": "B93114", "qty": 2048, "type": {"value": "memory"}},
            ],
        },
    ]
}


def gpu_shape(
    name: str,
    processor_type: str,
    gpu_qty: int,
    gpu_memory_qty: int,
    part_number: str,
    ocpu_qty: int | None,
    bundle_memory_qty: int | None,
    allow_preemptible: bool = False,
) -> dict:
    return {
        "name": name,
        "hidden": False,
        "status": "ACTIVE",
        "allowPreemptible": allow_preemptible,
        "bundleMemoryQty": bundle_memory_qty,
        "gpuQty": gpu_qty,
        "gpuMemoryQty": gpu_memory_qty,
        "processorType": {"value": processor_type},
        "shapeType": {"value": "bm"},
        "subType": {"value": "gpu"},
        "products": [{"partNumber": part_number, "qty": ocpu_qty, "type": {"value": "ocpu"}}],
    }


# Real GPU shapes as reported by the Cost Estimator on 2026-09-18. Note the data quality
# issues the provider has to work around: `gpuMemoryQty` is the total GPU memory for most
# shapes but the per-GPU memory for BM.GPU.MI355X.8, and it is off for BM.GPU.GB300.4;
# the OCPU quantity and the bundled memory are missing for the newest shapes; the processor
# type of Grace superchips is "blackwell" although their CPUs are Arm.
GPU_SHAPES = {
    "items": [
        gpu_shape("BM.GPU.H100.8", "hopper", 8, 640, "B98415", 112, 2048),
        gpu_shape("BM.GPU4.8", "arm", 8, 320, "B92740", 64, 2048),
        gpu_shape("BM.GPU.A100-v2.8", "arm", 8, 640, "B95907", 128, 2048),
        gpu_shape("BM.GPU.MI300X.8", "amd", 8, 1536, "B109485", 112, 2048),
        gpu_shape("BM.GPU.MI355X.8", "amd", 8, 288, "B111758", 128, 3072),
        gpu_shape("BM.GPU.GB200.4 (NVL72)", "blackwell", 4, 756, "B110979", 128, 960),
        gpu_shape("BM.GPU.GB300.4 (NVL72)", "blackwell", 4, 278, "B112140", None, None),
        gpu_shape("BM.GPU.B300.8", "blackwell", 8, 2100, "B112237", None, None),
        gpu_shape("BM.GPU.RTXPRO.8", "blackwell", 8, 768, "B112613", None, None),
        gpu_shape("VM.GPU3.1", "volta", 1, 16, "B89734", 6, 90, allow_preemptible=True),
    ]
}


def gpu_product(part_number: str, price: float) -> dict:
    return {
        "partNumber": part_number,
        "billingModel": "UCM",
        "pricetype": "HOUR",
        "currencyCodeLocalizations": [
            {"currencyCode": "USD", "prices": [{"model": "PAY_AS_YOU_GO", "value": price}]}
        ],
    }


# Per GPU-hour prices, https://www.oracle.com/a/ocom/docs/cloudestimator2/data/products.json
GPU_PRODUCTS = {
    "items": [
        gpu_product("B98415", 10),  # H100
        gpu_product("B92740", 3.05),  # A100 40GB (BM.GPU4.8)
        gpu_product("B95907", 4),  # A100 80GB (BM.GPU.A100-v2.8)
        gpu_product("B109485", 6),  # MI300X
        gpu_product("B111758", 8.6),  # MI355X
        gpu_product("B110979", 16),  # GB200
        gpu_product("B112140", 18),  # GB300
        gpu_product("B112237", 15),  # B300
        gpu_product("B112613", 4.5),  # RTX PRO 6000
        gpu_product("B89734", 2.95),  # V100
    ]
}


@pytest.fixture
def gpu_shapes() -> dict[str, CostEstimatorShape]:
    shapes = CostEstimatorShapeList.model_validate_json(json.dumps(GPU_SHAPES))
    return {shape.name: shape for shape in shapes.items}


@pytest.fixture
def gpu_products() -> CostEstimatorProductList:
    return CostEstimatorProductList.model_validate_json(json.dumps(GPU_PRODUCTS))


class TestCostEstimatorShapeList:
    def test_fractional_qty_is_truncated(self):
        """Some integral quantities are reported as floats. Pydantic v1 truncated them, v2
        errors out, which would fail the whole document and not just the shape we do not
        even use."""
        shapes = CostEstimatorShapeList.model_validate_json(json.dumps(SHAPES))
        assert [product.qty for product in shapes.items[0].products] == [128, 1536, 81]
        assert shapes.items[1].name == "BM.Standard.E4.128"


class TestNormalizeShapeName:
    @pytest.mark.parametrize(
        ("display_name", "api_name"),
        [
            ("BM.GPU.GB200.4 (NVL72)", "BM.GPU.GB200.4"),
            ("BM.GPU.GB300.4 (NVL72)", "BM.GPU.GB300.4"),
            ("BM.GPU.A100-v2.8", "BM.GPU.A100-v2.8"),
            ("VM.Standard.E4.Flex", "VM.Standard.E4.Flex"),
        ],
    )
    def test_normalize_shape_name(self, display_name, api_name):
        assert normalize_shape_name(display_name) == api_name


class TestGetGpuName:
    @pytest.mark.parametrize(
        ("shape_name", "gpu_name"),
        [
            ("VM.GPU.A10.2", "A10"),
            ("BM.GPU.A100-v2.8", "A100"),
            ("BM.GPU4.8", "A100"),
            ("VM.GPU3.4", "V100"),
            ("VM.GPU2.1", "P100"),
            ("BM.GPU.H100.8", "H100"),
            ("BM.GPU.H200.8", "H200"),
            ("BM.GPU.L40S.4", "L40S"),
            ("BM.GPU.B200.8", "B200"),
            ("BM.GPU.B300.8", "B300"),
            ("BM.GPU.GB200.4 (NVL72)", "GB200"),
            ("BM.GPU.GB300.4 (NVL72)", "GB300"),
            ("BM.GPU.RTXPRO.8", "RTXPRO6000"),
            ("BM.GPU.MI300X.8", "MI300X"),
            ("BM.GPU.MI355X.8", "MI355X"),
            ("VM.Standard2.8", None),
            ("VM.Notgpu.A10", None),
            ("BM.GPU.UNKNOWN.8", None),
        ],
    )
    def test_get_gpu_name(self, shape_name, gpu_name):
        assert get_gpu_name(shape_name) == gpu_name


class TestShapeToResources:
    @pytest.mark.parametrize(
        ("shape_name", "vendor", "gpu_name", "gpu_count", "gpu_memory", "cpu_arch", "vcpus", "ram", "price"),
        [
            # Complete Cost Estimator data, unchanged behaviour
            ("BM.GPU.H100.8", AcceleratorVendor.NVIDIA, "H100", 8, 80, CPUArchitecture.X86, 224, 2048, 80.0),
            # The two A100 variants are told apart by the reported memory
            ("BM.GPU4.8", AcceleratorVendor.NVIDIA, "A100", 8, 40, CPUArchitecture.X86, 128, 2048, 24.4),
            ("BM.GPU.A100-v2.8", AcceleratorVendor.NVIDIA, "A100", 8, 80, CPUArchitecture.X86, 256, 2048, 32.0),
            # AMD accelerators
            ("BM.GPU.MI300X.8", AcceleratorVendor.AMD, "MI300X", 8, 192, CPUArchitecture.X86, 224, 2048, 48.0),
            # gpuMemoryQty is the per-GPU memory here, not the total
            ("BM.GPU.MI355X.8", AcceleratorVendor.AMD, "MI355X", 8, 288, CPUArchitecture.X86, 256, 3072, 68.8),
            # Grace superchips are Arm; reported 756/4 = 189 GB is corrected to the known 186 GB
            ("BM.GPU.GB200.4 (NVL72)", AcceleratorVendor.NVIDIA, "GB200", 4, 186, CPUArchitecture.ARM, 128, 960, 64.0),
            # OCPU qty and bundled memory missing from the Cost Estimator, taken from the docs
            ("BM.GPU.GB300.4 (NVL72)", AcceleratorVendor.NVIDIA, "GB300", 4, 270, CPUArchitecture.ARM, 144, 960, 72.0),
            ("BM.GPU.B300.8", AcceleratorVendor.NVIDIA, "B300", 8, 270, CPUArchitecture.X86, 256, 4096, 120.0),
            ("BM.GPU.RTXPRO.8", AcceleratorVendor.NVIDIA, "RTXPRO6000", 8, 96, CPUArchitecture.X86, 288, 3072, 36.0),
        ],
    )  # fmt: skip
    def test_gpu_shapes(
        self,
        gpu_shapes,
        gpu_products,
        shape_name,
        vendor,
        gpu_name,
        gpu_count,
        gpu_memory,
        cpu_arch,
        vcpus,
        ram,
        price,
    ):
        resources = shape_to_resources(gpu_shapes[shape_name], gpu_products)
        assert resources.gpu.vendor == vendor
        assert resources.gpu.name == gpu_name
        assert resources.gpu.units_count == gpu_count
        assert resources.gpu.unit_memory_gb == gpu_memory
        assert resources.cpu.arch == cpu_arch
        assert resources.cpu.vcpus == vcpus
        assert resources.memory.gbs == ram
        assert resources.total_price() == pytest.approx(price)

    def test_unknown_gpu_is_rejected(self, gpu_shapes, gpu_products):
        shape = gpu_shapes["BM.GPU.H100.8"].model_copy(update={"name": "BM.GPU.UNKNOWN.8"})
        with pytest.raises(CostEstimatorDataError, match="Incomplete GPU parameters"):
            shape_to_resources(shape, gpu_products)

    def test_missing_ocpu_qty_without_known_specs_is_rejected(self, gpu_shapes, gpu_products):
        shape = gpu_shapes["BM.GPU.B300.8"].model_copy(update={"name": "BM.GPU.B300.16"})
        with pytest.raises(CostEstimatorDataError, match="Product quantity not found"):
            shape_to_resources(shape, gpu_products)


class TestOCIProvider:
    @pytest.fixture
    def provider(self) -> OCIProvider:
        with patch("gpuhunt.providers.oci.oci.identity.IdentityClient") as client_cls:
            client_cls.return_value.list_regions.return_value = SimpleNamespace(
                data=[Region(name="us-chicago-1"), Region(name="eu-frankfurt-1")]
            )
            return OCIProvider(
                OCICredentials(
                    user="user", key_content="key", fingerprint="fp", tenancy="tenancy", region="r"
                )
            )

    @pytest.fixture
    def offers(self, provider):
        with requests_mock.Mocker() as m:
            m.get(COST_ESTIMATOR_URL_TEMPLATE.format(resource="shapes.json"), json=GPU_SHAPES)
            m.get(COST_ESTIMATOR_URL_TEMPLATE.format(resource="products.json"), json=GPU_PRODUCTS)
            return provider.get()

    def test_all_gpu_shapes_are_collected(self, offers):
        on_demand = {o.instance_name for o in offers if not o.spot}
        assert on_demand == {
            "BM.GPU.H100.8",
            "BM.GPU4.8",
            "BM.GPU.A100-v2.8",
            "BM.GPU.MI300X.8",
            "BM.GPU.MI355X.8",
            "BM.GPU.GB200.4",
            "BM.GPU.GB300.4",
            "BM.GPU.B300.8",
            "BM.GPU.RTXPRO.8",
            "VM.GPU3.1",
        }
        assert {o.location for o in offers} == {"us-chicago-1", "eu-frankfurt-1"}

    def test_shape_names_match_compute_api(self, offers):
        assert not any("(" in o.instance_name for o in offers)

    def test_vendor_and_arch(self, offers):
        by_name = {o.instance_name: o for o in offers if o.location == "us-chicago-1"}
        assert by_name["BM.GPU.MI300X.8"].gpu_vendor == AcceleratorVendor.AMD
        assert by_name["BM.GPU.H100.8"].gpu_vendor == AcceleratorVendor.NVIDIA
        assert by_name["BM.GPU.GB200.4"].cpu_arch == CPUArchitecture.ARM
        assert by_name["BM.GPU.H100.8"].cpu_arch == CPUArchitecture.X86

    def test_spot_offers(self, offers):
        spot = [o for o in offers if o.spot]
        assert {o.instance_name for o in spot} == {"VM.GPU3.1"}
        assert all(o.flags == ["oci-spot"] for o in spot)
        assert spot[0].price == pytest.approx(2.95 / 2)

    def test_skips_shape_with_bad_data(self, provider, caplog):
        shapes = {
            "items": GPU_SHAPES["items"]
            + [gpu_shape("BM.GPU.UNKNOWN.8", "x", 8, 1, "B98415", 1, 1)]
        }
        with requests_mock.Mocker() as m:
            m.get(COST_ESTIMATOR_URL_TEMPLATE.format(resource="shapes.json"), json=shapes)
            m.get(COST_ESTIMATOR_URL_TEMPLATE.format(resource="products.json"), json=GPU_PRODUCTS)
            offers = provider.get()
        assert not any(o.instance_name == "BM.GPU.UNKNOWN.8" for o in offers)
        assert "Skipping shape BM.GPU.UNKNOWN.8" in caplog.text
