import json
import logging
import math
import os
import re
import time
from collections import namedtuple
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests
import requests.adapters
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient

from gpuhunt import AcceleratorVendor
from gpuhunt._internal.models import CatalogItem, QueryFilter
from gpuhunt._internal.utils import get_or_error
from gpuhunt.providers.base import OfflineProvider

logger = logging.getLogger(__name__)
prices_url = "https://prices.azure.com/api/retail/prices"
retail_prices_page_size = 1000
prices_version = "2023-01-01-preview"
prices_filters = [
    "serviceName eq 'Virtual Machines'",
    "priceType eq 'Consumption'",
    "contains(productName, 'Windows') eq false",
    "contains(productName, 'Dedicated') eq false",
    "contains(meterName, 'Low Priority') eq false",  # retires in 2025
]
VMSeries = namedtuple("VMSeries", ["pattern", "gpu_name", "gpu_memory"])
gpu_vm_series = [
    VMSeries(r"NC(\d+)ads_A100_v4", "A100", 80.0),  # NC A100 v4-series [A100 80GB]
    VMSeries(r"NC(\d+)ads_A10_v4", "A10", 24.0),  # NC A10 v4-series [A10]
    VMSeries(r"NC(\d+)as_T4_v3", "T4", 16.0),  # NCasT4_v3-series [T4]
    VMSeries(r"NC(\d+)r?s_v3", "V100", 16.0),  # NCv3-series [V100 16GB]
    VMSeries(r"NC(\d+)adi?s_H100_v5", "H100NVL", 94.0),  # NC H100 v5-series [H100 NVL 94GB]
    VMSeries(r"ND(\d+)amsr_A100_v4", "A100", 80.0),  # NDm A100 v4-series [8xA100 80GB]
    VMSeries(r"ND(\d+)asr_v4", "A100", 40.0),  # ND A100 v4-series [8xA100 40GB]
    VMSeries(r"ND(\d+)rs_v2", "V100", 32.0),  # NDv2-series [8xV100 32GB]
    VMSeries(r"ND(\d+)isr_H200_v5", "H200", 141.0),  # ND H200 v5-series [8xH200 141GB]
    VMSeries(r"NG(\d+)adm?s_V620_v1", "V620", None),  # NGads V620-series [V620]  # todo
    VMSeries(r"NV(\d+)adm?s_A10_v5", "A10", 24.0),  # NVadsA10 v5-series [A10]
    VMSeries(r"NV(\d+)as_v4", "MI25", None),  # NVv4-series [MI25]  # todo
    VMSeries(r"NV(\d+)s_v3", "M60", None),  # NVv3-series [M60]  # todo
]
# https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-previous-gen
retired_vm_series = [
    r"Basic_A(\d+)",
    r"Standard_A(\d+)",
    r"Standard_D(\d+)",
    r"Standard_DC(\d+)s",
    r"Standard_DS(\d+)",
    r"Standard_F(\d+)",
    r"Standard_F(\d+)s",
    r"Standard_G(\d+)",
    r"Standard_GS(\d+)",
    r"Standard_L(\d+)s",
    r"Standard_NC(\d+)r?",
    r"Standard_NC(\d+)r?s_v2",
    r"Standard_ND(\d+)r?s",
    r"Standard_NV(\d+)",
    r"Standard_NV(\d+)s_v2",
]


@dataclass
class _InstanceSpec:
    instance_name: str
    cpu: int
    memory: float
    gpu_count: int
    gpu_name: str | None
    gpu_memory: float | None
    gpu_vendor: AcceleratorVendor | None

    def to_catalog_item(self, *, location: str, price: float, spot: bool) -> CatalogItem:
        return CatalogItem(
            provider=AzureProvider.NAME,
            instance_name=self.instance_name,
            location=location,
            price=price,
            cpu=self.cpu,
            memory=self.memory,
            gpu_count=self.gpu_count,
            gpu_name=self.gpu_name,
            gpu_memory=self.gpu_memory,
            spot=spot,
            disk_size=None,
            gpu_vendor=self.gpu_vendor,
        )


class AzureProvider(OfflineProvider):
    NAME = "azure"

    def __init__(
        self,
        subscription_id: str,
        credential: TokenCredential | None = None,
        cache_dir: str | None = None,
    ):
        self.cache_dir = cache_dir
        self.client = ComputeManagementClient(
            credential=credential or DefaultAzureCredential(),
            subscription_id=subscription_id,
        )

    def get_pages(self, regions: Iterable[str], threads: int = 8) -> Iterable[list[dict]]:
        """
        Yields pricing pages, paginating one region at a time.

        Prices are paginated with `$skip`, which only adds up to the full list as long as the
        underlying data does not change. A region is only a few pages long, so a price update
        during collection cannot shift rows out of the pages that are left to read.
        """

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(self._get_region_pages, region) for region in regions]
            for future in as_completed(futures):
                yield from future.result()

    def _get_region_pages(self, region: str) -> list[list[dict]]:
        pages = []
        page_id = 0
        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
        while True:
            cached_page = None
            if self.cache_dir is not None:
                cached_page = os.path.join(self.cache_dir, f"{region}-{page_id:02}.json")
            if cached_page is not None and os.path.exists(cached_page):
                with open(cached_page) as f:
                    data = json.load(f)
            else:
                logger.info("Fetching pricing page %s for %s", page_id, region)
                res = session.get(
                    prices_url,
                    params={
                        "$filter": " and ".join([*prices_filters, f"armRegionName eq '{region}'"]),
                        "$skip": page_id * retail_prices_page_size,
                    },
                )
                if res.status_code == 429:
                    logger.warning("Got 429 for %s: sleep 3 & retry", region)
                    time.sleep(3)
                    continue
                res.raise_for_status()
                if cached_page is not None:
                    with open(cached_page, "w") as f:
                        f.write(res.text)
                data = res.json()
            if not data["Items"]:
                return pages
            pages.append(data["Items"])
            page_id += 1

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        offers: list[CatalogItem] = []
        instance_name_to_spec_map, regions = self.get_specs_and_regions()
        for page in self.get_pages(regions):
            for sku_item in page:
                if is_retired(sku_item["armSkuName"]):
                    continue
                if not sku_item["armSkuName"]:
                    continue
                price = float(sku_item["retailPrice"])
                if math.isclose(price, 0):
                    continue
                spec = instance_name_to_spec_map.get(sku_item["armSkuName"])
                if spec is None:
                    continue
                offer = spec.to_catalog_item(
                    location=sku_item["armRegionName"],
                    price=price,
                    spot="Spot" in sku_item["meterName"],
                )
                offers.append(offer)
        return sorted(offers, key=lambda i: i.price)

    def get_specs_and_regions(self) -> tuple[dict[str, _InstanceSpec], set[str]]:
        """
        Returns the VM specs available to the subscription, keyed by instance name, along with
        the regions those VMs are offered in.
        """

        logger.info("Fetching instance details")
        instance_name_to_spec_map = {}
        regions: set[str] = set()
        resources = self.client.resource_skus.list()
        for resource in resources:
            assert resource.name is not None
            if resource.resource_type != "virtualMachines":
                continue
            regions.update(location.lower() for location in resource.locations or [])
            if is_retired(resource.name):
                continue
            capabilities = {
                pair.name: pair.value
                for pair in get_or_error(resource.capabilities, "resource capabilities")
            }
            cpu = capabilities.get("vCPUs")
            memory = capabilities.get("MemoryGB")
            if not cpu:
                logger.warning("Instance CPU is missing: %s", resource.name)
                continue
            if not memory:
                logger.warning("Instance memory is missing: %s", resource.name)
                continue
            gpu_count, gpu_name, gpu_memory = 0, None, None
            if "GPUs" in capabilities:
                gpu_count = int(get_or_error(capabilities["GPUs"], "GPUs capability"))
                gpu_name, gpu_memory = get_gpu_name_memory(resource.name)
                if gpu_name is None and gpu_count:
                    logger.warning("Can't parse VM name: %s", resource.name)
                    continue
            instance_name_to_spec_map[resource.name] = _InstanceSpec(
                instance_name=resource.name,
                cpu=int(cpu),
                memory=float(memory),
                gpu_vendor=AcceleratorVendor.NVIDIA if gpu_count else None,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                gpu_memory=gpu_memory,
            )
        logger.info(
            "Found %s instance types in %s regions", len(instance_name_to_spec_map), len(regions)
        )
        return instance_name_to_spec_map, regions

    @classmethod
    def filter(cls, offers: list[CatalogItem]) -> list[CatalogItem]:
        vm_series = [
            VMSeries(r"D(\d+)s_v6", None, None),  # Dsv6-series
            VMSeries(
                r"E(2|4|8|16|20|32|48|64|96)s_v6", None, None
            ),  # Esv6-series (E128 and E192i are not yet GA)
            VMSeries(r"F(\d+)s_v2", None, None),  # Fsv2-series
            VMSeries(r"NC(\d+)s_v3", "V100", 16 * 1024),  # NCv3-series [V100 16GB]
            VMSeries(r"NC(\d+)as_T4_v3", "T4", 16 * 1024),  # NCasT4_v3-series [T4]
            VMSeries(
                r"NC(\d+)adi?s_H100_v5", "H100NVL", 94 * 1024
            ),  # NC H100 v5-series [H100 NVL 94GB]
            VMSeries(r"ND(\d+)rs_v2", "V100", 32 * 1024),  # NDv2-series [8xV100 32GB]
            VMSeries(
                r"ND(\d+)isr_H200_v5", "H200", 141 * 1024
            ),  # ND H200 v5-series [8xH200 141GB]
            VMSeries(r"NV(\d+)adm?s_A10_v5", "A10", 24 * 1024),  # NVadsA10 v5-series [A10]
            VMSeries(r"NC(\d+)ads_A100_v4", "A100", 80 * 1024),  # NC A100 v4-series [A100 80GB]
            VMSeries(r"ND(\d+)asr_v4", "A100", 40 * 1024),  # ND A100 v4-series [8xA100 40GB]
            VMSeries(
                r"ND(\d+)amsr_A100_v4", "A100", 80 * 1024
            ),  # NDm A100 v4-series [8xA100 80GB]
            # The deprecated series are collected for older dstack versions
            VMSeries(
                r"D(\d+)s_v3", None, None
            ),  # Dsv3-series (deprecated in favor of Dsv6-series)
            VMSeries(
                r"E(\d+)i?s_v4", None, None
            ),  # Esv4-series (deprecated in favor of Esv6-series)
            VMSeries(
                r"E(\d+)-(\d+)s_v4", None, None
            ),  # Esv4-series (constrained vCPU, deprecated in favor of Esv6-series)
        ]
        vm_series_pattern = re.compile(
            f"^Standard_({'|'.join(series.pattern for series in vm_series)})$"
        )
        return [i for i in offers if vm_series_pattern.match(i.instance_name)]


def get_gpu_name_memory(vm_name: str) -> tuple[str | None, float | None]:
    for pattern, gpu_name, gpu_memory in gpu_vm_series:
        m = re.match(f"^Standard_{pattern}$", vm_name)
        if m is None:
            continue
        if gpu_name == "A10" and vm_name.endswith("_v4"):
            gpu_memory = gpu_memory * min(1.0, int(m.group(1)) / 16)
        elif gpu_name == "A10" and vm_name.endswith("_v5"):
            gpu_memory = gpu_memory * min(1.0, int(m.group(1)) / 36)

        return gpu_name, gpu_memory
    return None, None


def is_retired(name: str) -> bool:
    if re.match(f"^({'|'.join(retired_vm_series)})$", name):
        return True
    return False
