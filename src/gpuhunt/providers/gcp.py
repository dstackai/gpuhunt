import copy
import logging
import re
from collections import defaultdict, namedtuple
from typing import List, Optional

import google.cloud.billing_v1 as billing_v1
import google.cloud.compute_v1 as compute_v1

from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
compute_service = "services/6F81-5844-456A"
AcceleratorDetails = namedtuple("AcceleratorDetails", ["name", "memory"])
accelerator_details = {
    "nvidia-a100-80gb": AcceleratorDetails("A100", 80.0),
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
                        gpu = accelerator_details[
                            machine_type.accelerators[0].guest_accelerator_type
                        ]
                    instance = RawCatalogItem(
                        instance_name=machine_type.name,
                        location=zone,
                        cpu=machine_type.guest_cpus,
                        memory=round(machine_type.memory_mb / 1024, 1),
                        gpu_count=machine_type.accelerators[0].guest_accelerator_count
                        if gpu
                        else 0,
                        gpu_name=machine_type.accelerators[0].guest_accelerator_type
                        if gpu
                        else None,
                        gpu_memory=gpu.memory if gpu else None,
                        price=None,
                        spot=None,
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
        # fetch per-unit prices
        families = {
            "gpu": defaultdict(dict),
            "ram": defaultdict(dict),
            "core": defaultdict(dict),
        }
        skus = self.cloud_catalog_client.list_skus(parent=compute_service)
        for sku in skus:
            if sku.category.resource_family != "Compute":
                continue
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
            r = re.match(
                r"^(?:spot preemptible )?(.+) (gpu|ram|core)",
                sku.description,
                flags=re.IGNORECASE,
            )
            if not r:
                continue

            family, resource = r.groups()
            resource = resource.lower()
            if resource == "gpu":
                family = family.replace(" ", "-").lower()
                family = {"nvidia-tesla-a100-80gb": "nvidia-a100-80gb"}.get(family, family)
            else:
                r = re.match(r"^([a-z]\d.?) ", family.lower())
                if r:
                    family = r.group(1)
                else:
                    family = {
                        "Memory-optimized Instance": "m1",
                        "Compute optimized Instance": "c2",
                        "Compute optimized": "c2",
                    }.get(family, family)

            price = sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price
            price = price.units + price.nanos / 1e9
            spot = sku.category.usage_type == "Preemptible"
            for region in sku.service_regions:
                families[resource][family][(region, spot)] = price

        # apply per-unit prices to instances
        offers = []
        for instance in instances:
            vm_family = instance.instance_name.split("-")[0]
            if vm_family in [
                "g1",
                "f1",
                "m2",
            ]:  # ignore shared-core and reservation-only
                continue
            for spot in (False, True):
                region_spot = (instance.location[:-2], spot)

                price = 0
                if region_spot not in families["core"][vm_family]:
                    continue
                price += instance.cpu * families["core"][vm_family][region_spot]
                price += instance.memory * families["ram"][vm_family][region_spot]
                if instance.gpu_name:
                    if region_spot not in families["gpu"][instance.gpu_name]:
                        continue
                    price += instance.gpu_count * families["gpu"][instance.gpu_name][region_spot]

                offer = copy.deepcopy(instance)
                offer.price = round(price, 6)
                offer.spot = spot
                if offer.gpu_name:
                    offer.gpu_name = accelerator_details[offer.gpu_name].name
                offers.append(offer)
        return offers

    def get(self, query_filter: Optional[QueryFilter] = None) -> List[RawCatalogItem]:
        instances = self.list_preconfigured_instances()
        self.add_gpus(instances)
        return self.fill_prices(instances)

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
