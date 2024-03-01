import json
import logging
import math
import os
import re
import time
from collections import namedtuple
from queue import Queue
from threading import Thread
from typing import Iterable, List, Optional, Tuple

import requests
import requests.adapters
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient

from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

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
    VMSeries(r"ND(\d+)amsr_A100_v4", "A100", 80.0),  # NDm A100 v4-series [8xA100 80GB]
    VMSeries(r"ND(\d+)asr_v4", "A100", 40.0),  # ND A100 v4-series [8xA100 40GB]
    VMSeries(r"ND(\d+)rs_v2", "V100", 32.0),  # NDv2-series [8xV100 32GB]
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


class AzureProvider(AbstractProvider):
    NAME = "azure"

    def __init__(
        self,
        subscription_id: str,
        credential: Optional[TokenCredential] = None,
        cache_dir: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        self.client = ComputeManagementClient(
            credential=credential or DefaultAzureCredential(),
            subscription_id=subscription_id,
        )

    def get_pages(self, threads: int = 8) -> Iterable[List[dict]]:
        q = Queue()
        workers = [
            Thread(target=self._get_pages_worker, args=(q, threads, i), daemon=True)
            for i in range(threads)
        ]
        for worker in workers:
            worker.start()

        exited = 0
        while exited < threads:
            page = q.get()
            if page is None:
                exited += 1
            else:
                yield page
            q.task_done()

    def _get_pages_worker(self, q: Queue, stride: int, worker_id: int):
        page_id = worker_id
        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
        try:
            while True:
                cached_page = None
                if self.cache_dir is not None:
                    cached_page = os.path.join(self.cache_dir, f"{page_id:04}.json")
                if cached_page is not None and os.path.exists(cached_page):
                    with open(cached_page, "r") as f:
                        data = json.load(f)
                else:
                    logger.info("Worker %s fetches pricing page %s", worker_id, page_id)
                    res = session.get(
                        prices_url,
                        params={
                            "api-version": "2023-01-01-preview",
                            "$filter": " and ".join(prices_filters),
                            "$skip": page_id * retail_prices_page_size,
                        },
                    )
                    if res.status_code == 429:
                        logger.warning("Worker %s got 429: sleep 3 & retry", worker_id)
                        time.sleep(3)
                        continue
                    res.raise_for_status()
                    if cached_page is not None:
                        with open(cached_page, "w") as f:
                            f.write(res.text)
                    data = res.json()
                if not data["Items"]:
                    logger.info("Worker %s exited", worker_id)
                    return
                q.put(data["Items"])
                page_id += stride
        except Exception as e:
            logger.exception("Worker %s failed: %s", worker_id, e)
        finally:
            q.put(None)

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        offers = []
        for page in self.get_pages():
            for item in page:
                if is_retired(item["armSkuName"]):
                    continue
                if not item["armSkuName"]:
                    continue
                price = float(item["retailPrice"])
                if math.isclose(price, 0):
                    continue
                offer = RawCatalogItem(
                    instance_name=item["armSkuName"],
                    location=item["armRegionName"],
                    price=price,
                    spot="Spot" in item["meterName"],
                    cpu=None,
                    memory=None,
                    gpu_count=None,
                    gpu_name=None,
                    gpu_memory=None,
                    disk_size=None,
                )
                offers.append(offer)
        offers = self.fill_details(offers)
        return sorted(offers, key=lambda i: i.price)

    def fill_details(self, offers: List[RawCatalogItem]) -> List[RawCatalogItem]:
        logger.info("Fetching instance details")
        instances = {}
        resources = self.client.resource_skus.list()
        for resource in resources:
            if resource.resource_type != "virtualMachines":
                continue
            if is_retired(resource.name):
                continue
            capabilities = {pair.name: pair.value for pair in resource.capabilities}
            gpu_count, gpu_name, gpu_memory = 0, None, None
            if "GPUs" in capabilities:
                gpu_count = int(capabilities["GPUs"])
                gpu_name, gpu_memory = get_gpu_name_memory(resource.name)
                if gpu_name is None and gpu_count:
                    logger.warning("Can't parse VM name: %s", resource.name)
                    continue
            instances[resource.name] = RawCatalogItem(
                instance_name=resource.name,
                cpu=capabilities["vCPUs"],
                memory=float(capabilities["MemoryGB"]),
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                gpu_memory=gpu_memory,
                location=None,
                price=None,
                spot=None,
                disk_size=None,
            )
        with_details = []
        without_details = []
        for offer in offers:
            if (resources := instances.get(offer.instance_name)) is None:
                without_details.append(offer)
                continue
            offer.cpu = resources.cpu
            offer.memory = resources.memory
            offer.gpu_count = resources.gpu_count
            offer.gpu_name = resources.gpu_name
            offer.gpu_memory = resources.gpu_memory
            with_details.append(offer)
        return with_details + without_details

    @classmethod
    def filter(cls, offers: List[RawCatalogItem]) -> List[RawCatalogItem]:
        vm_series = [
            VMSeries(r"D(\d+)s_v3", None, None),  # Dsv3-series
            VMSeries(r"E(\d+)i?s_v4", None, None),  # Esv4-series
            VMSeries(r"E(\d+)-(\d+)s_v4", None, None),  # Esv4-series (constrained vCPU)
            VMSeries(r"NC(\d+)s_v3", "V100", 16 * 1024),  # NCv3-series [V100 16GB]
            VMSeries(r"NC(\d+)as_T4_v3", "T4", 16 * 1024),  # NCasT4_v3-series [T4]
            VMSeries(r"ND(\d+)rs_v2", "V100", 32 * 1024),  # NDv2-series [8xV100 32GB]
            VMSeries(r"NV(\d+)adm?s_A10_v5", "A10", 24 * 1024),  # NVadsA10 v5-series [A10]
            VMSeries(r"NC(\d+)ads_A100_v4", "A100", 80 * 1024),  # NC A100 v4-series [A100 80GB]
            VMSeries(r"ND(\d+)asr_v4", "A100", 40 * 1024),  # ND A100 v4-series [8xA100 40GB]
            VMSeries(
                r"ND(\d+)amsr_A100_v4", "A100", 80 * 1024
            ),  # NDm A100 v4-series [8xA100 80GB]
        ]
        vm_series_pattern = re.compile(
            f"^Standard_({'|'.join(series.pattern for series in vm_series)})$"
        )
        return [i for i in offers if vm_series_pattern.match(i.instance_name)]


def get_gpu_name_memory(vm_name: str) -> Tuple[Optional[str], Optional[float]]:
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
