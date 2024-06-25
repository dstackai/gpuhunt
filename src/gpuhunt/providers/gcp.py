import copy
import importlib
import json
import logging
import re
from collections import defaultdict, namedtuple
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import google.cloud.billing_v1 as billing_v1
import google.cloud.compute_v1 as compute_v1
from google.cloud import tpu_v2
from google.cloud.billing_v1 import CloudCatalogClient, ListSkusRequest
from google.cloud.billing_v1.types.cloud_catalog import Sku
from google.cloud.location import locations_pb2
from google.cloud.location.locations_pb2 import ListLocationsResponse

from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
compute_service = "services/6F81-5844-456A"
AcceleratorDetails = namedtuple("AcceleratorDetails", ["name", "memory"])
accelerator_details = {
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


def load_tpu_pricing():
    resource_package = "gpuhunt.resources"
    resource_name = "tpu_pricing.json"

    with importlib.resources.open_text(resource_package, resource_name) as f:
        return json.load(f)


tpu_pricing: dict = load_tpu_pricing()


class GCPProvider(AbstractProvider):
    NAME = "gcp"

    def __init__(self, project: str):
        # todo credentials
        self.project = project
        self.machine_types_client = compute_v1.MachineTypesClient()
        self.accelerator_types_client = compute_v1.AcceleratorTypesClient()
        self.regions_client = compute_v1.RegionsClient()
        self.cloud_catalog_client = billing_v1.CloudCatalogClient()

    def list_preconfigured_instances(self) -> List[RawCatalogItem]:
        instances = []
        for region in self.regions_client.list(project=self.project):
            for zone_url in region.zones:
                zone = zone_url.split("/")[-1]
                logger.info("Fetching instances for zone %s", zone)
                for machine_type in self.machine_types_client.list(
                    project=self.project, zone=zone
                ):
                    if (
                        machine_type.deprecated.state
                        == compute_v1.DeprecationStatus.State.DEPRECATED
                    ):
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
                        gpu_count=(
                            machine_type.accelerators[0].guest_accelerator_count if gpu else 0
                        ),
                        gpu_name=(
                            machine_type.accelerators[0].guest_accelerator_type if gpu else None
                        ),
                        gpu_memory=gpu.memory if gpu else None,
                        price=None,
                        spot=None,
                        disk_size=None,
                    )
                    instances.append(instance)
        return instances

    def add_gpus(self, instances: List[RawCatalogItem]):
        n1_instances = defaultdict(list)
        for instance in instances:
            if instance.instance_name.startswith("n1-"):
                n1_instances[instance.location].append(instance)

        instances_with_gpus = []
        for zone, zone_n1_instances in n1_instances.items():
            logger.info("Fetching GPUs for zone %s", zone)
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
                        instances_with_gpus.append(i)
        instances += instances_with_gpus

    def fill_prices(self, instances: List[RawCatalogItem]) -> List[RawCatalogItem]:
        logger.info("Fetching prices")
        skus = self.cloud_catalog_client.list_skus(parent=compute_service)
        prices = Prices()
        prices.add_skus(skus)

        offers = []
        for instance in instances:
            for spot in (False, True):
                price = prices.get_instance_price(instance, spot)
                if price is None:
                    continue

                offer = copy.deepcopy(instance)
                offer.price = round(price, 6)
                offer.spot = spot
                if offer.gpu_name:
                    offer.gpu_name = accelerator_details[offer.gpu_name].name
                offers.append(offer)
        return offers

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        instances = self.list_preconfigured_instances()
        self.add_gpus(instances)
        offers = self.fill_prices(instances)
        # Add tpu offerings
        offers.extend(get_tpu_offers(self.project))
        return sorted(offers, key=lambda i: i.price)

    @classmethod
    def filter(cls, offers: List[RawCatalogItem]) -> List[RawCatalogItem]:
        return [
            i
            for i in offers
            if any(
                i.instance_name.startswith(family)
                for family in [
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
        ]


RegionSpot = Tuple[str, bool]
PricePerRegionSpot = Dict[RegionSpot, float]


class Prices:
    def __init__(self):
        self.cpu: DefaultDict[str, PricePerRegionSpot] = defaultdict(dict)
        self.gpu: DefaultDict[str, PricePerRegionSpot] = defaultdict(dict)
        self.ram: DefaultDict[str, PricePerRegionSpot] = defaultdict(dict)
        self.local_ssd: PricePerRegionSpot = dict()

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

        price = sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price
        price = price.units + price.nanos / 1e9
        resource_prices = {
            "core": self.cpu,
            "gpu": self.gpu,
            "ram": self.ram,
        }[resource]
        self._add_price(sku, resource_prices[family], price)

    def add_storage_sku(self, sku: Sku) -> None:
        if sku.description.lower().startswith("ssd backed local storage"):
            price = sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price
            price = price.units + price.nanos / 1e9 / hours_in_month
            self._add_price(sku, self.local_ssd, price)

    @staticmethod
    def _add_price(sku: Sku, family_prices: PricePerRegionSpot, price: float) -> None:
        spot = sku.category.usage_type == "Preemptible"
        for region in sku.service_regions:
            family_prices[(region, spot)] = price

    def get_instance_price(self, instance: RawCatalogItem, spot: bool) -> Optional[float]:
        vm_family = self.get_vm_family(instance.instance_name)
        if vm_family in ["g1", "f1", "m2"]:  # shared-core and reservation-only
            return None

        region_spot = (instance.location[:-2], spot)
        if region_spot not in self.cpu[vm_family]:
            return None

        price = 0
        price += instance.cpu * self.cpu[vm_family][region_spot]
        price += instance.memory * self.ram[vm_family][region_spot]
        if instance.gpu_name:
            if region_spot not in self.gpu[instance.gpu_name]:
                return None
            price += instance.gpu_count * self.gpu[instance.gpu_name][region_spot]
        if instance.instance_name in local_ssd_sizes_gib:
            price += local_ssd_sizes_gib[instance.instance_name] * self.local_ssd[region_spot]

        return price

    @staticmethod
    def get_vm_family(instance_name: str) -> str:
        for family in multi_token_vm_families:
            if instance_name.startswith(family):
                return family
        return instance_name.split("-")[0]


def get_tpu_offers(project_id: str) -> List[RawCatalogItem]:
    logger.info("Fetching tpu offers")
    raw_catalog_items: List[RawCatalogItem] = []
    catalog_items: List[dict] = get_catalog_items(project_id)
    filtered_catalog_items = [item for item in catalog_items if item["price"] is not None]
    for item in filtered_catalog_items:
        on_demand_item = RawCatalogItem(
            instance_name=item["instance_name"],
            location=item["location"],
            price=item["price"],
            cpu=0,
            memory=0,
            gpu_count=1,
            gpu_name=f'tpu-{item["instance_name"]}',
            gpu_memory=0,
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


def get_catalog_items(project_id: str) -> List[dict]:
    tpu_prices: List[dict] = get_tpu_prices()
    configs: List[dict] = get_tpu_configs(project_id)
    for config in configs:
        instance_name = config["instance_name"]
        location = config["location"].rsplit("-", 1)[0]  # Remove the part after the last '-'
        no_of_chips = config["no_of_chips"]
        tpu_version, no_of_cores = instance_name.rsplit("-", 1)
        no_of_cores = int(no_of_cores)
        if tpu_version in ["v5litepod", "v5p"]:
            # For TPU-v5 series, api provides per chip price.
            # Verify per chip price in the following link.https://cloud.google.com/tpu/pricing
            is_pod = True
            on_demand_base_price = find_base_price_v5(
                tpu_version, location, tpu_prices, spot=False
            )
            spot_base_price = find_base_price_v5(tpu_version, location, tpu_prices, spot=True)
            if on_demand_base_price is not None:
                on_demand_price = on_demand_base_price * no_of_chips
            else:
                on_demand_price = find_tpu_price_static_src(
                    tpu_version, no_of_cores, location, no_of_chips, False
                )
            if spot_base_price is not None:
                spot_price = spot_base_price * no_of_chips
            else:
                spot_price = find_tpu_price_static_src(
                    tpu_version, no_of_cores, location, no_of_chips, True
                )
        elif tpu_version in ["v2", "v3", "v4"]:
            # For TPU-v2 and TPU-v3, the pricing API provides the prices of 8 TPU cores.
            # For TPU-v4, api only provides the price of TPU-v4 pods.
            if no_of_cores > 8 or tpu_version == "v4":
                is_pod = True
                base_instance_name = f"{tpu_version}-8"
                base_no_of_chips = find_no_of_chips(base_instance_name, configs)
                on_demand_base_price = find_base_price(
                    tpu_version, location, tpu_prices, spot=False, is_pod=True
                )
                spot_base_price = find_base_price(
                    tpu_version, location, tpu_prices, spot=True, is_pod=True
                )

                if on_demand_base_price is not None and base_no_of_chips is not None:
                    on_demand_price = (on_demand_base_price / base_no_of_chips) * no_of_chips
                else:
                    on_demand_price = find_tpu_price_static_src(
                        tpu_version, no_of_cores, location, no_of_chips, False
                    )
                if spot_base_price is not None and base_no_of_chips is not None:
                    spot_price = (spot_base_price / base_no_of_chips) * no_of_chips
                else:
                    spot_price = find_tpu_price_static_src(
                        tpu_version, no_of_cores, location, no_of_chips, True
                    )

            elif no_of_cores == 8:
                is_pod = False
                on_demand_base_price = find_base_price(
                    tpu_version, location, tpu_prices, spot=False, is_pod=False
                )
                spot_base_price = find_base_price(
                    tpu_version, location, tpu_prices, spot=True, is_pod=False
                )
                on_demand_price = (
                    on_demand_base_price if on_demand_base_price is not None else None
                )
                spot_price = spot_base_price if spot_base_price is not None else None
                base_no_of_chips = no_of_chips

        config["price"] = on_demand_price
        config["spot"] = spot_price
        config["is_pod"] = is_pod
        config["base_price"] = on_demand_base_price
        config["base_no_of_chips"] = base_no_of_chips
        config["spot_base_price"] = spot_base_price
    return configs


def get_tpu_prices() -> List[dict]:
    client = CloudCatalogClient()
    tpu_configs = []
    # E000-3F24-B8AA contains prices for TPU versions v2,v3,v4.
    # 6F81-5844-456A contains prices for TPU versions v5p and v5litepod(v5e)
    service_names = ["services/E000-3F24-B8AA", "services/6F81-5844-456A"]

    # Loop through each service name and list SKUs
    for service_name in service_names:
        # Create the request
        request = ListSkusRequest(parent=service_name)

        # List SKUs
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
    instance_name: str, location: str, tpu_prices: List[dict], spot: bool, is_pod: bool
) -> Optional[float]:
    for price_info in tpu_prices:
        if (
            price_info["instance_name"] == instance_name
            and any(loc.startswith(location) for loc in price_info["regions"])
            and price_info["spot"] == spot
            and price_info["is_pod"] == is_pod
        ):
            return price_info["price"]
    return None


def find_no_of_chips(instance_name: str, configs: List[dict]):
    for config in configs:
        if config["instance_name"] == instance_name:
            return config["no_of_chips"]
    return None


def find_tpu_price_static_src(
    tpu_version: str, num_cores: int, tpu_region: str, no_of_chips: int, spot: bool
) -> Optional[float]:
    # Pricing table names v5litepod as v5e
    tpu_version = "v5e" if tpu_version == "v5litepod" else tpu_version
    is_pod = num_cores > 8 or tpu_version == "v4"
    tpu_type = f"TPU {tpu_version} pod" if is_pod else f"TPU {tpu_version} device"
    # Comment here
    if tpu_version == "v5p" or tpu_version == "v5e":
        tpu_type = f"TPU {tpu_version}"
    try:
        on_demand_price = tpu_pricing[tpu_type][tpu_region]["On Demand (USD)"] * no_of_chips
        spot_price = tpu_pricing[tpu_type][tpu_region]["Spot (USD)"] * no_of_chips
        return on_demand_price if not spot else spot_price
    except KeyError:
        logger.warning(
            f'key error for {tpu_type} {tpu_region} {"On Demand (USD)" if spot else "Spot (USD)"}'
        )
        return None


def find_base_price_v5(
    instance_name: str, location: str, tpu_prices: List[dict], spot: bool
) -> Optional[float]:
    for price_info in tpu_prices:
        if (
            price_info["instance_name"] == instance_name
            and any(loc.startswith(location) for loc in price_info["regions"])
            and price_info["spot"] == spot
        ):
            return price_info["price"]
    return None


def get_tpu_configs(project_id: str) -> List[dict]:
    instances: List[dict] = []
    client = tpu_v2.TpuClient()
    for location in get_locations(project_id):
        parent = f"projects/{project_id}/locations/{location}"
        request = tpu_v2.ListAcceleratorTypesRequest(
            parent=parent,
        )
        # request = tpu_v1.ListAcceleratorTypesRequest(parent=parent)
        page_result = client.list_accelerator_types(request=request)
        for response in page_result:
            no_of_chips = get_no_of_chips(response.accelerator_configs[0].topology)
            instances.append(
                {
                    "instance_name": response.type_,
                    "location": location,
                    "no_of_chips": no_of_chips,
                    "topology": response.accelerator_configs[0].topology,
                }
            )
    return instances


def get_no_of_chips(expression: str) -> int:
    # Split the expression by 'x'
    factors = expression.split("x")
    # Convert each factor to an integer
    factors = map(int, factors)
    # Calculate the product by multiplying all factors
    product = 1
    for factor in factors:
        product *= factor
    return product


def get_locations(project_id: str) -> List[str]:
    client = tpu_v2.TpuClient()
    # Initialize request argument(s)
    parent = f"projects/{project_id}"
    list_locations_request: ListLocationsResponse = client.list_locations(
        locations_pb2.ListLocationsRequest(name=parent)
    )
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
