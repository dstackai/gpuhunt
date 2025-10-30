import copy
import enum
import importlib.resources
import json
import logging
import re
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, cast

import google.cloud.billing_v1 as billing_v1
import google.cloud.compute_v1 as compute_v1
from google.cloud import tpu_v2
from google.cloud.billing_v1 import CloudCatalogClient, ListSkusRequest
from google.cloud.billing_v1.types.cloud_catalog import Sku
from google.cloud.location import locations_pb2
from typing_extensions import NotRequired, TypedDict

from gpuhunt._internal.models import AcceleratorVendor, QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
compute_service = "services/6F81-5844-456A"
AcceleratorDetails = namedtuple("AcceleratorDetails", ["name", "memory"])
# As of 2024-14-08, this mapping contains only Nvidia accelerators; update gpu_vendor
# inferring code in fill_gpu_vendors_and_names() if a non-Nvidia accelerator is added
accelerator_details = {
    "nvidia-b200": AcceleratorDetails("B200", 180.0),
    "nvidia-a100-80gb": AcceleratorDetails("A100", 80.0),
    "nvidia-h100-80gb": AcceleratorDetails("H100", 80.0),
    "nvidia-h100-mega-80gb": AcceleratorDetails("H100", 80.0),
    "nvidia-l4": AcceleratorDetails("L4", 24.0),
    "nvidia-tesla-a100": AcceleratorDetails("A100", 40.0),
    "nvidia-tesla-k80": AcceleratorDetails("K80", 12.0),
    "nvidia-tesla-p100": AcceleratorDetails("P100", 16.0),
    "nvidia-tesla-p4": AcceleratorDetails("P4", 8.0),
    "nvidia-tesla-t4": AcceleratorDetails("T4", 16.0),
    "nvidia-tesla-v100": AcceleratorDetails("V100", 16.0),
    "nvidia-rtx-pro-6000": AcceleratorDetails("RTXPRO6000", 96.0),
}
CpuMemory = namedtuple("CpuMemory", ["cpu", "memory"])
accelerator_limits = {
    "nvidia-tesla-k80": [
        CpuMemory(8, 52),
        CpuMemory(16, 104),
        CpuMemory(32, 208),
        CpuMemory(64, 208),
    ],
    "nvidia-tesla-p100": [CpuMemory(16, 104), CpuMemory(32, 208), CpuMemory(96, 624)],
    "nvidia-tesla-p4": [CpuMemory(24, 156), CpuMemory(48, 312), CpuMemory(96, 624)],
    "nvidia-tesla-t4": [CpuMemory(48, 312), CpuMemory(48, 312), CpuMemory(96, 624)],
    "nvidia-tesla-v100": [
        CpuMemory(12, 78),
        CpuMemory(24, 156),
        CpuMemory(48, 312),
        CpuMemory(96, 624),
    ],
}
accelerator_counts = [1, 2, 4, 8, 16]
hours_in_month = 730  # according to GCP pricing
multi_token_vm_families = ["a3-megagpu"]

# https://cloud.google.com/compute/docs/disks/local-ssd#lssd_disks_fixed
local_ssd_sizes_gib = {
    "c3-standard-4-lssd": 1 * 375,
    "c3-standard-8-lssd": 2 * 375,
    "c3-standard-22-lssd": 4 * 375,
    "c3-standard-44-lssd": 8 * 375,
    "c3-standard-88-lssd": 16 * 375,
    "c3-standard-176-lssd": 32 * 375,
    "c3d-standard-8-lssd": 1 * 375,
    "c3d-standard-16-lssd": 1 * 375,
    "c3d-standard-30-lssd": 2 * 375,
    "c3d-standard-60-lssd": 4 * 375,
    "c3d-standard-90-lssd": 8 * 375,
    "c3d-standard-180-lssd": 16 * 375,
    "c3d-standard-360-lssd": 32 * 375,
    "c3d-highmem-8-lssd": 1 * 375,
    "c3d-highmem-16-lssd": 1 * 375,
    "c3d-highmem-30-lssd": 2 * 375,
    "c3d-highmem-60-lssd": 4 * 375,
    "c3d-highmem-90-lssd": 8 * 375,
    "c3d-highmem-180-lssd": 16 * 375,
    "c3d-highmem-360-lssd": 32 * 375,
    "a3-megagpu-8g": 16 * 375,
    "a3-highgpu-8g": 16 * 375,
    "a2-ultragpu-1g": 1 * 375,
    "a2-ultragpu-2g": 2 * 375,
    "a2-ultragpu-4g": 4 * 375,
    "a2-ultragpu-8g": 8 * 375,
    "z3-standard-88-lssd": 12 * 3000,
    "z3-standard-176-lssd": 12 * 3000,
}


@dataclass
class TPUHardwareSpec:
    name: str
    cpu: int
    memory_gb: int
    hbm_gb: int


TPU_HARDWARE_SPECS = [
    TPUHardwareSpec(name="v2-8", cpu=96, memory_gb=334, hbm_gb=64),
    TPUHardwareSpec(name="v3-8", cpu=96, memory_gb=334, hbm_gb=128),
    TPUHardwareSpec(name="v5litepod-1", cpu=24, memory_gb=48, hbm_gb=16),
    TPUHardwareSpec(name="v5litepod-2", cpu=112, memory_gb=192, hbm_gb=16),
    TPUHardwareSpec(name="v5litepod-8", cpu=224, memory_gb=384, hbm_gb=128),
    TPUHardwareSpec(name="v5p-8", cpu=208, memory_gb=448, hbm_gb=95),
    TPUHardwareSpec(name="v6e-1", cpu=44, memory_gb=176, hbm_gb=32),
    TPUHardwareSpec(name="v6e-4", cpu=180, memory_gb=720, hbm_gb=128),
    TPUHardwareSpec(name="v6e-8", cpu=180, memory_gb=1440, hbm_gb=256),
]


# For newer TPUs, the specs are described in the docs: https://cloud.google.com/tpu/docs/v6e
# For older TPUs, the specs are collected manually from running instances.
TPU_HARDWARE_SPECS = [
    TPUHardwareSpec(name="v2-8", cpu=96, memory_gb=334, hbm_gb=64),
    TPUHardwareSpec(name="v3-8", cpu=96, memory_gb=334, hbm_gb=128),
    TPUHardwareSpec(name="v5litepod-1", cpu=24, memory_gb=48, hbm_gb=16),
    TPUHardwareSpec(name="v5litepod-4", cpu=112, memory_gb=192, hbm_gb=64),
    TPUHardwareSpec(name="v5litepod-8", cpu=224, memory_gb=384, hbm_gb=128),
    TPUHardwareSpec(name="v5p-8", cpu=208, memory_gb=448, hbm_gb=95),
    TPUHardwareSpec(name="v6e-1", cpu=44, memory_gb=176, hbm_gb=32),
    TPUHardwareSpec(name="v6e-4", cpu=180, memory_gb=720, hbm_gb=128),
    TPUHardwareSpec(name="v6e-8", cpu=180, memory_gb=1440, hbm_gb=256),
]
TPU_NAME_TO_HARDWARE_SPEC = {spec.name: spec for spec in TPU_HARDWARE_SPECS}


def load_tpu_pricing() -> dict:
    return json.loads(
        importlib.resources.files("gpuhunt.resources").joinpath("tpu_pricing.json").read_text()
    )


# A manually filled TPU pricing table from the pricing page.
# On-demand TPUs - https://cloud.google.com/tpu/pricing?hl=en.
# Spot TPUs - https://cloud.google.com/spot-vms/pricing?hl=en.
# It's needed since the TPU pricing API does not return prices for all regions.
# The API may also return 1-year Commitment prices instead of on-demand prices.
TPU_PRICING_TABLE = load_tpu_pricing()


class GCPProvider(AbstractProvider):
    NAME = "gcp"

    def __init__(self, project: str):
        # todo credentials
        self.project = project
        self.machine_types_client = compute_v1.MachineTypesClient()
        self.accelerator_types_client = compute_v1.AcceleratorTypesClient()
        self.regions_client = compute_v1.RegionsClient()
        self.cloud_catalog_client = billing_v1.CloudCatalogClient()

    def list_preconfigured_instances(self) -> list[RawCatalogItem]:
        def _list_zone_instances(zone: str) -> list[RawCatalogItem]:
            zone_instances = []
            logger.info("Fetching instances for zone %s", zone)
            for machine_type in self.machine_types_client.list(project=self.project, zone=zone):
                if machine_type.deprecated.state == compute_v1.DeprecationStatus.State.DEPRECATED:
                    continue
                gpu = None
                if machine_type.accelerators:
                    accelerator = machine_type.accelerators[0].guest_accelerator_type
                    gpu = accelerator_details.get(accelerator)
                    if gpu is None:
                        logger.warning("Unknown accelerator type: %s", accelerator)
                        continue

                instance = RawCatalogItem(
                    instance_name=machine_type.name,
                    location=zone,
                    cpu=machine_type.guest_cpus,
                    memory=round(machine_type.memory_mb / 1024, 1),
                    gpu_count=(machine_type.accelerators[0].guest_accelerator_count if gpu else 0),
                    # gpu_name is canonicalized and gpu_vendor is set later
                    # in fill_gpu_vendors_and_names(), for now we use AcceleratorType.name
                    # as a name (it contains a vendor prefix like "nvidia-")
                    gpu_name=(
                        machine_type.accelerators[0].guest_accelerator_type if gpu else None
                    ),
                    gpu_vendor=None,
                    gpu_memory=gpu.memory if gpu else None,
                    price=None,
                    spot=None,
                    disk_size=None,
                )
                zone_instances.append(instance)
            return zone_instances

        instances = []
        futures = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for region in self.regions_client.list(project=self.project):
                for zone_url in region.zones:
                    zone = zone_url.split("/")[-1]
                    futures.append(ex.submit(_list_zone_instances, zone))
        for future in as_completed(futures):
            instances.extend(future.result())
        return instances

    def add_gpus(self, instances: list[RawCatalogItem]):
        def _list_zone_instances(
            zone: str, zone_n1_instances: list[RawCatalogItem]
        ) -> list[RawCatalogItem]:
            logger.info("Fetching GPUs for zone %s", zone)
            zone_instances = []
            for accelerator in self.accelerator_types_client.list(project=self.project, zone=zone):
                if accelerator.name not in accelerator_limits:
                    continue
                for n, limit in zip(accelerator_counts, accelerator_limits[accelerator.name]):
                    for instance in zone_n1_instances:
                        if instance.cpu > limit.cpu or instance.memory > limit.memory:
                            continue
                        i = copy.deepcopy(instance)
                        i.gpu_count = n
                        i.gpu_name = accelerator.name
                        i.gpu_memory = accelerator_details[accelerator.name].memory
                        zone_instances.append(i)
            return zone_instances

        n1_instances = defaultdict(list)
        for instance in instances:
            if instance.instance_name.startswith("n1-"):
                n1_instances[instance.location].append(instance)

        instances_with_gpus = []
        futures = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for zone, zone_n1_instances in n1_instances.items():
                futures.append(ex.submit(_list_zone_instances, zone, zone_n1_instances))
        for future in as_completed(futures):
            instances_with_gpus.extend(future.result())
        instances += instances_with_gpus

    def fill_prices(self, instances: list[RawCatalogItem]) -> list[RawCatalogItem]:
        logger.info("Fetching prices")
        skus = self.cloud_catalog_client.list_skus(parent=compute_service)
        prices = Prices()
        prices.add_skus(skus)

        offers = []
        for instance in instances:
            for capacity_type in CapacityType:
                price = prices.get_instance_price(instance, capacity_type)
                if price is None:
                    continue

                offer = copy.deepcopy(instance)
                offer.price = round(price, 6)
                offer.spot = capacity_type is CapacityType.SPOT
                cast(GCPCatalogItemProviderData, offer.provider_data)["is_dws_calendar_mode"] = (
                    capacity_type is CapacityType.DWS_CALENDAR_MODE
                )
                offers.append(offer)
        return offers

    def fill_gpu_vendors_and_names(self, offers: list[RawCatalogItem]) -> None:
        # Modifies offers in the list in-place
        for offer in offers:
            accelerator_type = offer.gpu_name
            if not accelerator_type:
                continue
            offer.gpu_name = accelerator_details[accelerator_type].name
            if accelerator_type.startswith("nvidia-"):
                offer.gpu_vendor = AcceleratorVendor.NVIDIA.value
            else:
                logger.warning("Unknown accelerator vendor: %s", accelerator_type)

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        instances = self.list_preconfigured_instances()
        self.add_gpus(instances)
        offers = self.fill_prices(instances)
        self.fill_gpu_vendors_and_names(offers)
        offers.extend(get_tpu_offers(self.project))
        set_flags(offers)
        offers = add_legacy_g4_preview(offers)
        return sorted(offers, key=lambda i: i.price)

    @classmethod
    def filter(cls, offers: list[RawCatalogItem]) -> list[RawCatalogItem]:
        return [
            i
            for i in offers
            if (
                any(
                    i.instance_name.startswith(family)
                    for family in [
                        "m4-",
                        "c4-",
                        "n4-",
                        "h3-",
                        "n2-",
                        "e2-medium",
                        "e2-standard-",
                        "e2-highmem-",
                        "e2-highcpu-",
                        "m1-",
                        "a2-",
                        "g2-",
                    ]
                )
                or (i.gpu_name and i.gpu_name not in ["K80", "P4"])
            )
            and not (
                # Filter out on-demand offers that are not actually available on demand.
                # https://cloud.google.com/compute/docs/accelerator-optimized-machines#consumption_option_availability_by_machine_type
                i.spot == False
                and not cast(GCPCatalogItemProviderData, i.provider_data).get(
                    "is_dws_calendar_mode"
                )
                and (
                    i.instance_name.startswith("a4x-")
                    or i.instance_name.startswith("a4-")
                    or i.instance_name.startswith("a3-ultragpu-")
                    or (
                        i.instance_name.startswith("a3-highgpu-")
                        and (i.gpu_count is None or i.gpu_count < 8)
                    )
                )
            )
        ]


class GCPCatalogItemProviderData(TypedDict):
    is_dws_calendar_mode: NotRequired[bool]


class CapacityType(enum.Enum):
    ON_DEMAND = enum.auto()
    SPOT = enum.auto()
    DWS_CALENDAR_MODE = enum.auto()


RegionCapacityType = tuple[str, CapacityType]
PricePerRegionCapacityType = dict[RegionCapacityType, float]


class Prices:
    def __init__(self):
        self.cpu: defaultdict[str, PricePerRegionCapacityType] = defaultdict(dict)
        self.gpu: defaultdict[str, PricePerRegionCapacityType] = defaultdict(dict)
        self.ram: defaultdict[str, PricePerRegionCapacityType] = defaultdict(dict)
        self.local_ssd: PricePerRegionCapacityType = dict()
        self.gpu_slice: defaultdict[str, PricePerRegionCapacityType] = defaultdict(dict)

    def add_skus(self, skus: Iterable[Sku]) -> None:
        for sku in skus:
            if sku.category.usage_type not in ["OnDemand", "Preemptible"]:
                continue
            if any(
                word in sku.description
                for word in [
                    "Sole Tenancy",
                    "Reserved",
                    "Premium",
                    "Custom",
                    "suspended",
                ]
            ):
                continue
            if sku.category.resource_family == "Compute":
                self.add_compute_sku(sku)
            elif sku.category.resource_family == "Storage":
                self.add_storage_sku(sku)

    def add_compute_sku(self, sku: Sku) -> None:
        if "(1 gpu slice)" in sku.description:
            self.add_compute_gpu_slice_sku(sku)
            return

        if "RTX 6000 96GB" in sku.description:
            family = "nvidia-rtx-pro-6000"
            resource = "gpu"
        else:
            r = re.match(
                r"^(?:spot preemptible )?(.+) (gpu|ram|core)",
                sku.description,
                flags=re.IGNORECASE,
            )
            if not r:
                return
            family, resource = r.groups()
            resource = resource.lower()

        if resource == "gpu":
            family = family.replace(" ", "-").lower()
            family = {
                "nvidia-tesla-a100-80gb": "nvidia-a100-80gb",
                "nvidia-h100-80gb-mega": "nvidia-h100-mega-80gb",
                "nvidia-h100-80gb-plus": "nvidia-h100-mega-80gb",
            }.get(family, family)
        else:
            r = re.match(r"^([a-z]\d.?) ", family.lower())
            if r:
                family = r.group(1)
            else:
                family = {
                    "Memory-optimized Instance": "m1",
                    "Compute optimized Instance": "c2",
                    "Compute optimized": "c2",
                    "A3Plus Instance": "a3-megagpu",
                }.get(family, family)

        price = self._calculate_sku_price(sku)
        resource_prices = {
            "core": self.cpu,
            "gpu": self.gpu,
            "ram": self.ram,
        }[resource]
        self._add_price(sku, resource_prices[family], price)

    def add_compute_gpu_slice_sku(self, sku: Sku) -> None:
        if sku.description.startswith(
            "Spot Preemptible A4 Nvidia B200"
        ) or sku.description.startswith("DWS Calendar Mode A4 Nvidia B200"):
            gpu = "nvidia-b200"
        else:
            return
        price = self._calculate_sku_price(sku)
        self._add_price(sku, self.gpu_slice[gpu], price)

    def add_storage_sku(self, sku: Sku) -> None:
        if sku.description.lower().startswith("ssd backed local storage"):
            price = self._calculate_sku_price(sku) / hours_in_month
            self._add_price(sku, self.local_ssd, price)

    @staticmethod
    def _calculate_sku_price(sku: Sku) -> float:
        price = sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price
        return price.units + price.nanos / 1e9

    @staticmethod
    def _add_price(sku: Sku, family_prices: PricePerRegionCapacityType, price: float) -> None:
        if sku.category.usage_type == "Preemptible":
            capacity_type = CapacityType.SPOT
        elif "DWS Calendar Mode" in sku.description:
            capacity_type = CapacityType.DWS_CALENDAR_MODE
        else:
            capacity_type = CapacityType.ON_DEMAND
        for region in sku.service_regions:
            family_prices[(region, capacity_type)] = price

    def get_instance_price(
        self, instance: RawCatalogItem, capacity_type: CapacityType
    ) -> Optional[float]:
        vm_family = self.get_vm_family(instance.instance_name)
        if vm_family in ["g1", "f1", "m2"]:  # shared-core and reservation-only
            return None

        region_capacity_type = (instance.location[:-2], capacity_type)

        # For some instances, the price is proportional to the number of GPUs
        if instance.gpu_name and region_capacity_type in self.gpu_slice[instance.gpu_name]:
            return instance.gpu_count * self.gpu_slice[instance.gpu_name][region_capacity_type]

        # For others, the price consists of several components
        price = 0
        if (
            region_capacity_type not in self.cpu[vm_family]
            or region_capacity_type not in self.ram[vm_family]
        ):
            return None
        price += instance.cpu * self.cpu[vm_family][region_capacity_type]
        price += instance.memory * self.ram[vm_family][region_capacity_type]
        if instance.gpu_name:
            if region_capacity_type not in self.gpu[instance.gpu_name]:
                return None
            price += instance.gpu_count * self.gpu[instance.gpu_name][region_capacity_type]
        if instance.instance_name in local_ssd_sizes_gib:
            price += (
                local_ssd_sizes_gib[instance.instance_name] * self.local_ssd[region_capacity_type]
            )

        return price

    @staticmethod
    def get_vm_family(instance_name: str) -> str:
        for family in multi_token_vm_families:
            if instance_name.startswith(family):
                return family
        return instance_name.split("-")[0]


def set_flags(catalog_items: list[RawCatalogItem]) -> None:
    for item in catalog_items:
        if cast(GCPCatalogItemProviderData, item.provider_data).get("is_dws_calendar_mode"):
            item.flags.append("gcp-dws-calendar-mode")
        if item.instance_name.startswith("a4-"):
            item.flags.append("gcp-a4")
        elif item.instance_name.startswith("g4-standard-"):
            item.flags.append("gcp-g4")


# TODO: drop when dstack 0.19.33 is no longer relevant
def add_legacy_g4_preview(catalog_items: list[RawCatalogItem]) -> list[RawCatalogItem]:
    """
    For each g4-standard-* instance, add a duplicate item with the "gcp-g4-preview" flag.

    This is only needed for dstack 0.19.33, where the flag "gcp-g4-preview"
    is used instead of "gcp-g4".
    """
    new_items = []
    for item in catalog_items:
        new_items.append(item)
        if item.instance_name.startswith("g4-standard-"):
            preview_item = copy.deepcopy(item)
            preview_item.flags.remove("gcp-g4")
            preview_item.flags.append("gcp-g4-preview")
            new_items.append(preview_item)
    return new_items


def get_tpu_offers(project_id: str) -> list[RawCatalogItem]:
    logger.info("Fetching TPU offers")
    raw_catalog_items: list[RawCatalogItem] = []
    catalog_items: list[dict] = get_catalog_items(project_id)
    # For some TPU offers in some regions, GCP does not list prices at all. Skip such offers.
    filtered_catalog_items = [item for item in catalog_items if item["price"] is not None]
    for item in filtered_catalog_items:
        hardware_spec = get_tpu_hardware_spec(item["instance_name"])
        if hardware_spec is None:
            logger.debug("No TPU hardware spec for %s", item["instance_name"])
            continue
        on_demand_item = RawCatalogItem(
            instance_name=item["instance_name"],
            location=item["location"],
            price=item["price"],
            cpu=hardware_spec.cpu,
            memory=hardware_spec.memory_gb,
            gpu_vendor=AcceleratorVendor.GOOGLE.value,
            gpu_count=1,
            gpu_name=item["instance_name"],
            gpu_memory=hardware_spec.hbm_gb,
            spot=False,
            disk_size=None,
        )
        raw_catalog_items.append(on_demand_item)
        if item["spot"]:
            spot_item = copy.deepcopy(on_demand_item)
            spot_item.price = item["spot"]
            spot_item.spot = True
            raw_catalog_items.append(spot_item)
    return raw_catalog_items


def get_catalog_items(project_id: str) -> list[dict]:
    """
    Returns TPU configurations with pricing info.
    Each configuration contains on-demand price and spot price but any price can be missing.
    This is because the API does not return prices for all regions.
    As a backup, the prices are taken from the pricing table on the GCP website,
    but it also does not contain all the prices.
    Even when creating some TPUs in some regions via the GCP console,
    the price is not shown (e.g. v6e in us-south1).
    """
    tpu_prices: list[dict] = get_tpu_prices()
    configs: list[dict] = get_tpu_configs(project_id)
    for config in configs:
        instance_name = config["instance_name"]
        location = config["location"].rsplit("-", 1)[0]  # Remove the part after the last '-'
        no_of_chips = config["no_of_chips"]
        tpu_version, no_of_cores = instance_name.rsplit("-", 1)
        no_of_cores = int(no_of_cores)
        if tpu_version in ["v5litepod", "v5p", "v6e"]:
            # For TPU-v5 series, the API provides per chip price.
            on_demand_base_price = find_base_price_v5(
                tpu_version, location, tpu_prices, spot=False
            )
            if on_demand_base_price is not None:
                on_demand_price = on_demand_base_price * no_of_chips
            else:
                on_demand_price = find_tpu_price_static_src(
                    tpu_version, no_of_cores, location, no_of_chips, False
                )
            spot_base_price = find_base_price_v5(tpu_version, location, tpu_prices, spot=True)
            if spot_base_price is not None:
                spot_price = spot_base_price * no_of_chips
            else:
                spot_price = find_tpu_price_static_src(
                    tpu_version, no_of_cores, location, no_of_chips, True
                )
        elif tpu_version in ["v2", "v3", "v4"]:
            # For TPU-v2 and TPU-v3, the pricing API provides the prices of 8 TPU cores.
            # For TPU-v4, the API provides the price of TPU-v4 pods.
            if no_of_cores > 8 or tpu_version == "v4":
                base_instance_name = f"{tpu_version}-8"
                base_no_of_chips = find_no_of_chips(base_instance_name, configs)
                on_demand_base_price = find_base_price(
                    tpu_version, location, tpu_prices, spot=False, is_pod=True
                )
                if on_demand_base_price is not None and base_no_of_chips is not None:
                    on_demand_price = (on_demand_base_price / base_no_of_chips) * no_of_chips
                else:
                    on_demand_price = find_tpu_price_static_src(
                        tpu_version, no_of_cores, location, no_of_chips, False
                    )
                spot_base_price = find_base_price(
                    tpu_version, location, tpu_prices, spot=True, is_pod=True
                )
                if spot_base_price is not None and base_no_of_chips is not None:
                    spot_price = (spot_base_price / base_no_of_chips) * no_of_chips
                else:
                    spot_price = find_tpu_price_static_src(
                        tpu_version, no_of_cores, location, no_of_chips, True
                    )
            elif no_of_cores == 8:
                on_demand_price = find_base_price(
                    tpu_version, location, tpu_prices, spot=False, is_pod=False
                )
                spot_price = find_base_price(
                    tpu_version, location, tpu_prices, spot=True, is_pod=False
                )
            else:
                logger.warning("Unknown TPU type %s", instance_name)
                continue
        else:
            logger.warning("Unknown TPU version %s. Skipping offer.", tpu_version)
            continue
        if on_demand_price is None:
            logger.debug("Failed to find on-demand price for %s in %s", instance_name, location)
        if spot_price is None:
            logger.debug("Failed to find spot price for %s in %s", instance_name, location)
        config["price"] = on_demand_price
        config["spot"] = spot_price
    return configs


def get_tpu_prices() -> list[dict]:
    client = CloudCatalogClient()
    tpu_configs = []
    # E000-3F24-B8AA contains prices for TPU versions v2,v3,v4.
    # 6F81-5844-456A contains prices for newer TPU versions v5p, v5litepod(v5e), v6e.
    service_names = ["services/E000-3F24-B8AA", "services/6F81-5844-456A"]
    for service_name in service_names:
        request = ListSkusRequest(parent=service_name)
        response = client.list_skus(request=request)
        for sku in response.skus:
            if sku.category.resource_group != "TPU":
                continue
            if sku.category.usage_type not in ["OnDemand", "Preemptible"]:
                continue
            tpu_version = extract_tpu_version(sku.description)
            if tpu_version:
                is_pod = True if "Pod" in sku.description else False
                spot = True if "Preemptible" in sku.description else False
                price = sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price
                price = price.units + price.nanos / 1e9
                tpu_configs.append(
                    {
                        "instance_name": tpu_version,
                        "is_pod": is_pod,
                        "spot": spot,
                        "regions": sku.service_regions,
                        "price": price,
                        "description": sku.description,
                    }
                )
    return tpu_configs


def find_base_price(
    tpu_version: str, location: str, tpu_prices: list[dict], spot: bool, is_pod: bool
) -> Optional[float]:
    for price_info in tpu_prices:
        if (
            price_info["instance_name"] == tpu_version
            and any(loc.startswith(location) for loc in price_info["regions"])
            and price_info["spot"] == spot
            and price_info["is_pod"] == is_pod
        ):
            return price_info["price"]
    return None


def find_no_of_chips(instance_name: str, configs: list[dict]):
    for config in configs:
        if config["instance_name"] == instance_name:
            return config["no_of_chips"]
    return None


def find_tpu_price_static_src(
    tpu_version: str, num_cores: int, tpu_region: str, no_of_chips: int, spot: bool
) -> Optional[float]:
    # The pricing page names v5litepod as v5e
    tpu_version = "v5e" if tpu_version == "v5litepod" else tpu_version
    # The pricing page lists different (device and pod) prices per chip for v2 and v3.
    # The device is the smallest configuration with 8 TPU cores (e.g. v3-8).
    # TPU Pod connects mulitple TPU devices (e.g. v3-32).
    # Not applicable for newer generations.
    tpu_type = f"TPU {tpu_version}"
    if tpu_version in ["v2", "v3", "v4"]:
        is_pod = num_cores > 8 or tpu_version == "v4"
        tpu_type = f"TPU {tpu_version} pod" if is_pod else f"TPU {tpu_version} device"
    price_key = "On Demand (USD)"
    if spot:
        price_key = "Spot (USD)"
    try:
        return TPU_PRICING_TABLE[tpu_type][tpu_region][price_key] * no_of_chips
    except KeyError:
        logger.debug(f"KeyError for {tpu_type} {tpu_region} {price_key}")
        return None


def find_base_price_v5(
    tpu_version: str, location: str, tpu_prices: list[dict], spot: bool
) -> Optional[float]:
    for price_info in tpu_prices:
        if (
            price_info["instance_name"] == tpu_version
            and any(loc.startswith(location) for loc in price_info["regions"])
            and price_info["spot"] == spot
        ):
            return price_info["price"]
    return None


def get_tpu_configs(project_id: str) -> list[dict]:
    def _list_zone_configs(zone: str) -> list[dict]:
        zone_instances = []
        if zone in ["us-east1-b"]:
            # These zones return
            # google.api_core.exceptions.ServiceUnavailable: 503 502:Bad Gateway
            return []
        parent = f"projects/{project_id}/locations/{zone}"
        request = tpu_v2.ListAcceleratorTypesRequest(
            parent=parent,
        )
        page_result = client.list_accelerator_types(request=request)
        for response in page_result:
            no_of_chips = get_no_of_chips(response.accelerator_configs[0].topology)
            zone_instances.append(
                {
                    "instance_name": response.type_,
                    "location": zone,
                    "no_of_chips": no_of_chips,
                }
            )
        return zone_instances

    client = tpu_v2.TpuClient()
    instances: list[dict] = []
    futures = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for zone in get_locations(project_id):
            futures.append(ex.submit(_list_zone_configs, zone))
    for future in as_completed(futures):
        instances.extend(future.result())
    return instances


def get_no_of_chips(expression: str) -> int:
    factors = expression.split("x")
    factors = map(int, factors)
    product = 1
    for factor in factors:
        product *= factor
    return product


def get_locations(project_id: str) -> list[str]:
    client = tpu_v2.TpuClient()
    parent = f"projects/{project_id}"
    list_locations_request = client.list_locations(locations_pb2.ListLocationsRequest(name=parent))
    locations = [loc.location_id for loc in list_locations_request.locations]
    # TPU V4 only available in location us-central2-b only.
    # us-central2-b needs to be enabled in the project.
    return locations


def extract_tpu_version(input_string: str) -> Optional[str]:
    # The regular expression pattern to find a substring starting with 'Tpu'
    pattern = r"\bTpu[-\w]*\b"
    # Search for the first match of the pattern
    match = re.search(pattern, input_string, re.IGNORECASE)
    if match:
        tpu_match = match.group().lower()
        # The regular expression pattern to find the version part
        version_pattern = r"v\d+[a-z]*"
        version_match = re.search(version_pattern, tpu_match)
        if version_match:
            # Name of v5e in gcp console is v5litepod
            version = "v5litepod" if version_match.group() == "v5e" else version_match.group()
            return version
    return None


def get_tpu_hardware_spec(instance_name: str) -> Optional[TPUHardwareSpec]:
    return TPU_NAME_TO_HARDWARE_SPEC.get(instance_name)
