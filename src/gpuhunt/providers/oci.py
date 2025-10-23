import copy
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Annotated, Optional

import oci
from oci.identity.models import Region
from pydantic import BaseModel, Field
from requests import Session
from typing_extensions import TypedDict

from gpuhunt._internal.constraints import find_accelerators
from gpuhunt._internal.models import AcceleratorVendor, QueryFilter, RawCatalogItem
from gpuhunt._internal.utils import to_camel_case
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
COST_ESTIMATOR_URL_TEMPLATE = "https://www.oracle.com/a/ocom/docs/cloudestimator2/data/{resource}"
COST_ESTIMATOR_REQUEST_TIMEOUT = 10


class OCICredentials(TypedDict):
    user: Optional[str]
    key_content: Optional[str]
    fingerprint: Optional[str]
    tenancy: Optional[str]
    region: Optional[str]


class OCIProvider(AbstractProvider):
    NAME = "oci"

    def __init__(self, credentials: OCICredentials):
        self.api_client = oci.identity.IdentityClient(
            credentials if all(credentials.values()) else oci.config.from_file()
        )
        self.cost_estimator = CostEstimator()

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        shapes = self.cost_estimator.get_shapes()
        products = self.cost_estimator.get_products()
        regions: list[Region] = self.api_client.list_regions().data

        result = []

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

            on_demand_item = RawCatalogItem(
                instance_name=shape.name,
                location=None,
                price=resources.total_price(),
                cpu=resources.cpu.vcpus,
                memory=resources.memory.gbs,
                gpu_vendor=None,
                gpu_count=resources.gpu.units_count,
                gpu_name=resources.gpu.name,
                gpu_memory=resources.gpu.unit_memory_gb,
                spot=False,
                disk_size=None,
            )
            item_variations = [on_demand_item]
            if shape.allow_preemptible:
                item_variations.append(self._make_spot_item(on_demand_item))
            for item in item_variations:
                result.extend(self._duplicate_item_in_regions(item, regions))

        return sorted(result, key=lambda i: i.price)

    @staticmethod
    def _make_spot_item(item: RawCatalogItem) -> RawCatalogItem:
        item = copy.deepcopy(item)
        item.spot = True
        # > Preemptible capacity costs 50% less than on-demand capacity
        # https://docs.oracle.com/en-us/iaas/Content/Compute/Concepts/preemptible.htm#howitworks__billing
        item.price *= 0.5
        item.flags.append("oci-spot")
        return item

    @staticmethod
    def _duplicate_item_in_regions(
        item: RawCatalogItem, regions: Iterable[Region]
    ) -> list[RawCatalogItem]:
        result = []
        for region in regions:
            regional_item = copy.deepcopy(item)
            regional_item.location = region.name
            result.append(regional_item)
        return result


class CostEstimatorTypeField(BaseModel):
    value: str


class CostEstimatorShapeProduct(BaseModel):
    type: CostEstimatorTypeField
    part_number: str
    qty: Optional[int]

    class Config:
        alias_generator = to_camel_case


class CostEstimatorShape(BaseModel):
    name: str
    hidden: bool
    status: str
    allow_preemptible: bool
    bundle_memory_qty: Optional[int]
    gpu_qty: Optional[int]
    gpu_memory_qty: Optional[int]
    processor_type: CostEstimatorTypeField
    shape_type: CostEstimatorTypeField
    sub_type: CostEstimatorTypeField
    products: list[CostEstimatorShapeProduct]

    class Config:
        alias_generator = to_camel_case

    def is_arm_cpu(self):
        is_ampere_gpu = self.sub_type.value == "gpu" and (
            "GPU4" in self.name or "GPU.A10" in self.name
        )
        # the data says A10 and A100 GPU instances are ARM, but they are not
        return self.processor_type.value == "arm" and not is_ampere_gpu

    def get_gpu_unit_memory_gb(self) -> Optional[float]:
        if self.gpu_memory_qty and self.gpu_qty:
            return self.gpu_memory_qty / self.gpu_qty
        return None


class CostEstimatorShapeList(BaseModel):
    items: list[CostEstimatorShape]


class CostEstimatorPrice(BaseModel):
    model: str
    value: float


class CostEstimatorPriceLocalization(BaseModel):
    currency_code: str
    prices: list[CostEstimatorPrice]

    class Config:
        alias_generator = to_camel_case


class CostEstimatorProduct(BaseModel):
    part_number: str
    billing_model: str
    price_type: Annotated[str, Field(alias="pricetype")]
    currency_code_localizations: list[CostEstimatorPriceLocalization]

    class Config:
        alias_generator = to_camel_case

    def find_price_l10n(self, currency_code: str) -> Optional[CostEstimatorPriceLocalization]:
        return next(
            filter(
                lambda price: price.currency_code == currency_code,
                self.currency_code_localizations,
            ),
            None,
        )


class CostEstimatorProductList(BaseModel):
    items: list[CostEstimatorProduct]

    def find(self, part_number: str) -> Optional[CostEstimatorProduct]:
        return next(filter(lambda product: product.part_number == part_number, self.items), None)


class CostEstimator:
    def __init__(self):
        self.session = Session()

    def get_shapes(self) -> CostEstimatorShapeList:
        return self._get("shapes.json", CostEstimatorShapeList)

    def get_products(self) -> CostEstimatorProductList:
        return self._get("products.json", CostEstimatorProductList)

    def _get(self, resource: str, ResponseModel: type[BaseModel]):
        url = COST_ESTIMATOR_URL_TEMPLATE.format(resource=resource)
        resp = self.session.get(url, timeout=COST_ESTIMATOR_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return ResponseModel.parse_raw(resp.content)


class CostEstimatorDataError(Exception):
    pass


@dataclass
class CPUConfiguration:
    vcpus: int
    price: float


@dataclass
class MemoryConfiguration:
    gbs: int
    price: float


@dataclass
class GPUConfiguration:
    units_count: int
    unit_memory_gb: Optional[float]
    name: Optional[str]
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
    cpu = None
    gpu = GPUConfiguration(units_count=0, unit_memory_gb=None, name=None, price=0.0)
    memory: Optional[MemoryConfiguration] = None
    if shape.bundle_memory_qty is not None:
        memory = MemoryConfiguration(gbs=shape.bundle_memory_qty, price=0.0)

    for product in shape.products:
        if product.qty is None:
            raise CostEstimatorDataError("Product quantity not found")
        product_details = products.find(product.part_number)
        if product_details is None:
            raise CostEstimatorDataError(f"Could not find product {product.part_number!r}")
        product_price = get_product_price_usd_per_hour(product_details)

        if product.type.value == "ocpu":
            vcpus = product.qty if shape.is_arm_cpu() else product.qty * 2
            if shape.gpu_qty:
                gpu = GPUConfiguration(
                    units_count=shape.gpu_qty,
                    unit_memory_gb=shape.get_gpu_unit_memory_gb(),
                    name=get_gpu_name(shape.name),
                    price=product_price * shape.gpu_qty,
                )
                cpu = CPUConfiguration(vcpus=vcpus, price=0.0)
            else:
                cpu = CPUConfiguration(vcpus=vcpus, price=product_price * product.qty)

        elif product.type.value == "memory":
            memory = MemoryConfiguration(gbs=product.qty, price=product_price * product.qty)

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


def get_gpu_name(shape_name: str) -> Optional[str]:
    parts = re.split(r"[\.-]", shape_name.upper())

    if "GPU4" in parts:
        return "A100"
    if "GPU3" in parts:
        return "V100"
    if "GPU2" in parts:
        return "P100"

    if "GPU" in parts:
        gpu_name_index = parts.index("GPU") + 1
        if gpu_name_index < len(parts):
            gpu_name = parts[gpu_name_index]

            if accelerators := find_accelerators(
                names=[gpu_name], vendors=[AcceleratorVendor.NVIDIA]
            ):
                return accelerators[0].name
    return None
