import copy
import logging
import re
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Annotated, TypeVar

import oci
from oci.identity.models import Region
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from requests import Session
from typing_extensions import TypedDict

from gpuhunt._internal.constraints import (
    find_accelerators,
    get_gpu_vendor,
    is_nvidia_superchip,
)
from gpuhunt._internal.errors import ProviderError
from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, CPUArchitecture, QueryFilter
from gpuhunt._internal.utils import get_or_error, to_camel_case
from gpuhunt.providers.base import OfflineProvider

logger = logging.getLogger(__name__)
COST_ESTIMATOR_URL_TEMPLATE = "https://www.oracle.com/a/ocom/docs/cloudestimator2/data/{resource}"
COST_ESTIMATOR_REQUEST_TIMEOUT = 10

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)

# The Cost Estimator reports some integral quantities as fractional floats, e.g. the disk
# `qty` of BM.DenseIO.E5.128 is 81.6. Pydantic v1 truncated such values silently, while v2
# rejects them, failing the whole document. Truncate to keep the v1 behaviour: an unparsable
# shape we do not even use would otherwise break catalog collection entirely.
LaxInt = Annotated[int, BeforeValidator(lambda v: int(v) if isinstance(v, float) else v)]

GPU_VENDORS = (AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD)


@dataclass(frozen=True)
class ShapeSpecs:
    ocpus: int
    memory_gb: int


# The Cost Estimator omits or misreports OCPU and memory of some shapes. Values below come from
# https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm
SHAPE_SPECS_OVERRIDES: dict[str, ShapeSpecs] = {
    "BM.GPU.B300.8": ShapeSpecs(ocpus=128, memory_gb=4096),
    "BM.GPU.GB200.4": ShapeSpecs(ocpus=144, memory_gb=960),
    "BM.GPU.GB300.4": ShapeSpecs(ocpus=144, memory_gb=960),
    "BM.GPU.RTXPRO.8": ShapeSpecs(ocpus=144, memory_gb=3072),
}


class OCICredentials(TypedDict):
    user: str | None
    key_content: str | None
    fingerprint: str | None
    tenancy: str | None
    region: str | None


class OCIProvider(OfflineProvider):
    NAME = "oci"

    def __init__(self, credentials: OCICredentials):
        self.api_client = oci.identity.IdentityClient(
            credentials if all(credentials.values()) else oci.config.from_file()
        )
        self.cost_estimator = CostEstimator()

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        shapes = self.cost_estimator.get_shapes()
        products = self.cost_estimator.get_products()
        regions: list[Region] = get_or_error(
            self.api_client.list_regions(), "list_regions response"
        ).data
        region_names = [get_or_error(region.name, "region name") for region in regions]

        offers: list[CatalogItem] = []

        for shape in shapes.items:
            if (
                shape.hidden
                or shape.status != "ACTIVE"
                or shape.shape_type.value not in ("vm", "bm")
                or shape.sub_type.value not in ("standard", "gpu", "optimized")
                or ".A1." in shape.name
            ):
                continue

            try:
                resources = shape_to_resources(shape, products)
            except CostEstimatorDataError as e:
                logger.warning(
                    "Skipping shape %s due to unexpected Cost Estimator data: %s", shape.name, e
                )
                continue
            for region_name in region_names:
                on_demand_item = CatalogItem(
                    provider=OCIProvider.NAME,
                    instance_name=shape.name,
                    location=region_name,
                    price=resources.total_price(),
                    cpu_arch=resources.cpu.arch,
                    cpu=resources.cpu.vcpus,
                    memory=resources.memory.gbs,
                    gpu_vendor=resources.gpu.vendor,
                    gpu_count=resources.gpu.units_count,
                    gpu_name=resources.gpu.name,
                    gpu_memory=resources.gpu.unit_memory_gb,
                    spot=False,
                    disk_size=None,
                )
                item_variations = [on_demand_item]
                if shape.allow_preemptible:
                    item_variations.append(self._make_spot_offer(on_demand_item))
                offers.extend(item_variations)

        return sorted(offers, key=lambda i: i.price)

    @staticmethod
    def _make_spot_offer(item: CatalogItem) -> CatalogItem:
        item = copy.deepcopy(item)
        item.spot = True
        # > Preemptible capacity costs 50% less than on-demand capacity
        # https://docs.oracle.com/en-us/iaas/Content/Compute/Concepts/preemptible.htm#howitworks__billing
        item.price *= 0.5
        item.flags.append("oci-spot")
        return item


class CostEstimatorTypeField(BaseModel):
    value: str


class CostEstimatorShapeProduct(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel_case)

    type: CostEstimatorTypeField
    part_number: str
    qty: LaxInt | None = None


class CostEstimatorShape(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel_case, frozen=True)

    name_: Annotated[str, Field(alias="name")]
    hidden: bool
    status: str
    allow_preemptible: bool
    bundle_memory_qty: LaxInt | None = None
    gpu_qty: LaxInt | None = None
    gpu_memory_qty: LaxInt | None = None
    processor_type: CostEstimatorTypeField
    shape_type: CostEstimatorTypeField
    sub_type: CostEstimatorTypeField
    products: list[CostEstimatorShapeProduct]

    @cached_property
    def name(self) -> str:
        """Normalized name"""
        return self.name_.removesuffix(" (NVL72)")

    def get_gpu_unit_memory_gb(self) -> float | None:
        # The Cost Estimator misreports gpu_memory_qty for some shapes. Override.
        if ".MI355X." in self.name:
            return 288.0
        if ".GB200." in self.name:
            return 192.0
        if ".GB300." in self.name:
            return 278.0
        if self.gpu_memory_qty and self.gpu_qty:
            return self.gpu_memory_qty / self.gpu_qty
        return None


class CostEstimatorShapeList(BaseModel):
    items: list[CostEstimatorShape]


class CostEstimatorPrice(BaseModel):
    model: str
    value: float


class CostEstimatorPriceLocalization(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel_case)

    currency_code: str
    prices: list[CostEstimatorPrice]


class CostEstimatorProduct(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel_case)

    part_number: str
    billing_model: str
    # The explicit alias takes priority over the generated `priceType`
    price_type: Annotated[str, Field(alias="pricetype")]
    currency_code_localizations: list[CostEstimatorPriceLocalization]

    def find_price_l10n(self, currency_code: str) -> CostEstimatorPriceLocalization | None:
        return next(
            filter(
                lambda price: price.currency_code == currency_code,
                self.currency_code_localizations,
            ),
            None,
        )


class CostEstimatorProductList(BaseModel):
    items: list[CostEstimatorProduct]

    def find(self, part_number: str) -> CostEstimatorProduct | None:
        return next(filter(lambda product: product.part_number == part_number, self.items), None)


class CostEstimator:
    def __init__(self):
        self.session = Session()

    def get_shapes(self) -> CostEstimatorShapeList:
        return self._get("shapes.json", CostEstimatorShapeList)

    def get_products(self) -> CostEstimatorProductList:
        return self._get("products.json", CostEstimatorProductList)

    def _get(self, resource: str, ResponseModel: type[ResponseModelT]) -> ResponseModelT:
        url = COST_ESTIMATOR_URL_TEMPLATE.format(resource=resource)
        resp = self.session.get(url, timeout=COST_ESTIMATOR_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return ResponseModel.model_validate_json(resp.content)


class CostEstimatorDataError(ProviderError):
    pass


@dataclass
class CPUConfiguration:
    vcpus: int
    arch: CPUArchitecture
    price: float


@dataclass
class MemoryConfiguration:
    gbs: int
    price: float


@dataclass
class GPUConfiguration:
    units_count: int
    unit_memory_gb: float | None
    name: str | None
    vendor: AcceleratorVendor | None
    price: float

    def __post_init__(self):
        d = asdict(self)
        if any(d.values()) and not all(d.values()):
            raise CostEstimatorDataError(f"Incomplete GPU parameters: {self}")


@dataclass
class ResourcesConfiguration:
    cpu: CPUConfiguration
    memory: MemoryConfiguration
    gpu: GPUConfiguration

    def total_price(self) -> float:
        return self.cpu.price + self.memory.price + self.gpu.price


def shape_to_resources(
    shape: CostEstimatorShape, products: CostEstimatorProductList
) -> ResourcesConfiguration:
    specs_override = SHAPE_SPECS_OVERRIDES.get(shape.name)
    gpu_name = get_gpu_name(shape.name)
    cpu_arch = get_cpu_arch(shape, gpu_name)
    cpu = None
    gpu = GPUConfiguration(units_count=0, unit_memory_gb=None, name=None, vendor=None, price=0.0)
    memory: MemoryConfiguration | None = None
    if specs_override is not None:
        memory = MemoryConfiguration(gbs=specs_override.memory_gb, price=0.0)
    elif shape.bundle_memory_qty is not None:
        memory = MemoryConfiguration(gbs=shape.bundle_memory_qty, price=0.0)

    for product in shape.products:
        qty = product.qty
        if product.type.value == "ocpu" and specs_override is not None:
            qty = specs_override.ocpus
        if qty is None:
            raise CostEstimatorDataError("Product quantity not found")
        product_details = products.find(product.part_number)
        if product_details is None:
            raise CostEstimatorDataError(f"Could not find product {product.part_number!r}")
        product_price = get_product_price_usd_per_hour(product_details)

        if product.type.value == "ocpu":
            vcpus = qty if cpu_arch == CPUArchitecture.ARM else qty * 2
            if shape.gpu_qty:
                # For GPU shapes the "ocpu" product is priced per GPU-hour
                gpu = GPUConfiguration(
                    units_count=shape.gpu_qty,
                    unit_memory_gb=shape.get_gpu_unit_memory_gb(),
                    name=gpu_name,
                    vendor=get_gpu_vendor(gpu_name),
                    price=product_price * shape.gpu_qty,
                )
                cpu = CPUConfiguration(vcpus=vcpus, arch=cpu_arch, price=0.0)
            else:
                cpu = CPUConfiguration(vcpus=vcpus, arch=cpu_arch, price=product_price * qty)

        elif product.type.value == "memory":
            memory = MemoryConfiguration(gbs=qty, price=product_price * qty)

        else:
            raise CostEstimatorDataError(f"Unknown product type {product.type.value!r}")

    if cpu is None:
        raise CostEstimatorDataError("No ocpu product")
    if memory is None:
        raise CostEstimatorDataError("No memory product")

    return ResourcesConfiguration(cpu, memory, gpu)


def get_product_price_usd_per_hour(product: CostEstimatorProduct) -> float:
    if product.billing_model != "UCM":
        raise CostEstimatorDataError(
            f"Billing model for product {product.part_number!r} is {product.billing_model!r}"
        )
    if product.price_type != "HOUR":
        raise CostEstimatorDataError(
            f"Price type for product {product.part_number!r} is {product.price_type!r}"
        )
    price_l10n = product.find_price_l10n("USD")
    if price_l10n is None:
        raise CostEstimatorDataError(f"No USD price for product {product.part_number!r}")
    if len(price_l10n.prices) != 1:
        raise CostEstimatorDataError(
            f"Product {product.part_number!r} has {len(price_l10n.prices)} USD prices"
        )
    price = price_l10n.prices[0]
    if price.model != "PAY_AS_YOU_GO":
        raise CostEstimatorDataError(
            f"Pricing model for product {product.part_number!r} is {price.model!r}"
        )
    return price.value


def get_gpu_name(shape_name: str) -> str | None:
    parts = re.split(r"[\.-]", shape_name.upper())

    if "GPU4" in parts:
        return "A100"
    if "GPU3" in parts:
        return "V100"
    if "GPU2" in parts:
        return "P100"
    if "RTXPRO" in parts:
        return "RTXPRO6000"

    if "GPU" in parts:
        gpu_name_index = parts.index("GPU") + 1
        if gpu_name_index < len(parts):
            gpu_name = parts[gpu_name_index]
            if accelerators := find_accelerators(names=[gpu_name], vendors=GPU_VENDORS):
                return accelerators[0].name
    return None


def get_cpu_arch(shape: CostEstimatorShape, gpu_name: str | None) -> CPUArchitecture:
    # The Cost Estimator misreports processor_type for some shapes. Override.
    if ".GPU4." in shape.name or ".GPU.A10." in shape.name or ".GPU.A100-v2." in shape.name:
        return CPUArchitecture.X86
    if gpu_name is not None and is_nvidia_superchip(gpu_name):
        return CPUArchitecture.ARM
    return CPUArchitecture.ARM if shape.processor_type.value == "arm" else CPUArchitecture.X86
