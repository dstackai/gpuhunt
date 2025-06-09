import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
from requests import RequestException

from gpuhunt._internal.constraints import KNOWN_AMD_GPUS
from gpuhunt._internal.models import AcceleratorVendor, QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
API_URL = "https://api.runpod.io/graphql"


class RunpodProvider(AbstractProvider):
    NAME = "runpod"

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        offers = self.fetch_offers()
        return sorted(offers, key=lambda i: i.price)

    @staticmethod
    def fetch_offers() -> list[RawCatalogItem]:
        query_variables = build_query_variables()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(get_pods, query_variable) for query_variable in query_variables
            ]
        pods_by_query = []
        for future in futures:
            try:
                pods_by_query.append(future.result())
            except RequestException as e:
                logger.exception("Failed to get pods data: %s", e)

        catalog_items = []
        for query_variable, pods in zip(query_variables, pods_by_query):
            for pod in pods:
                catalog_items.extend(make_catalog_items(query_variable, pod))
        return catalog_items

    @classmethod
    def filter(cls, offers: list[RawCatalogItem]) -> list[RawCatalogItem]:
        return [
            o
            for o in offers
            if o.location
            not in [
                "AR",  # network problems, unusable
            ]
        ]


def gpu_vendor_and_name(gpu_id: str) -> Optional[tuple[AcceleratorVendor, str]]:
    if not gpu_id:
        return None
    return GPU_MAP.get(gpu_id)


def build_query_variables() -> list[dict]:
    """Prepare different combinations of API query filters to cover all available GPUs."""

    gpu_types = make_request({"query": gpu_types_query, "variables": {}})
    data_centers = [dc["id"] for dc in gpu_types["data"]["dataCenters"] if dc["listed"]]
    max_gpu_count = max(gpu["maxGpuCount"] for gpu in gpu_types["data"]["gpuTypes"])

    variables = []
    for gpu_count in range(1, max_gpu_count + 1):
        # Secure cloud is queryable by datacenter ID
        for dc_id in data_centers:
            variables.append(
                {
                    "secureCloud": True,
                    "dataCenterId": dc_id,
                    "gpuCount": gpu_count,
                    "minDisk": None,
                    "minMemoryInGb": None,
                    "minVcpuCount": None,
                }
            )
        # Community cloud is queryable by country code
        for country_code in gpu_types["data"]["countryCodes"]:
            variables.append(
                {
                    "secureCloud": False,
                    "countryCode": country_code,
                    "gpuCount": gpu_count,
                    "minDisk": None,
                    "minMemoryInGb": None,
                    "minVcpuCount": None,
                }
            )

    return variables


def get_pods(query_variable: dict) -> list[dict]:
    resp = make_request(
        {
            "query": query_pod_types,
            "variables": {"lowestPriceInput": query_variable},
        }
    )
    return resp["data"]["gpuTypes"]


def make_catalog_items(query_variable: dict, pod: dict) -> list[RawCatalogItem]:
    if pod["lowestPrice"]["stockStatus"] is None:
        return []
    listed_gpu_vendor_and_name = gpu_vendor_and_name(pod["id"])
    if listed_gpu_vendor_and_name is None:
        logger.warning(f"{pod['id']} missing in runpod GPU_MAP")
        return []
    if query_variable["secureCloud"]:
        location = query_variable["dataCenterId"]
        on_demand_gpu_price = pod["securePrice"]
        spot_gpu_price = pod["secureSpotPrice"]
    else:
        location = query_variable["countryCode"]
        on_demand_gpu_price = pod["communityPrice"]
        spot_gpu_price = pod["communitySpotPrice"]
    item_template = RawCatalogItem(
        instance_name=pod["id"],
        location=location,
        price=None,  # set below
        cpu=pod["lowestPrice"]["minVcpu"],
        memory=pod["lowestPrice"]["minMemory"],
        gpu_vendor=listed_gpu_vendor_and_name[0],
        gpu_count=query_variable["gpuCount"],
        gpu_name=listed_gpu_vendor_and_name[1],
        gpu_memory=pod["memoryInGb"],
        spot=None,  # set below
        disk_size=None,
    )
    items = []
    if on_demand_gpu_price:
        item = copy.deepcopy(item_template)
        item.spot = False
        item.price = item.gpu_count * on_demand_gpu_price
        items.append(item)
    if spot_gpu_price:
        item = copy.deepcopy(item_template)
        item.spot = True
        item.price = item.gpu_count * spot_gpu_price
        items.append(item)
    return items


def make_request(payload: dict):
    resp = requests.post(API_URL, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_gpu_map() -> dict[str, tuple[AcceleratorVendor, str]]:
    payload_gpus = {
        "query": "query GpuTypes { gpuTypes { id manufacturer displayName memoryInGb } }"
    }
    response = make_request(payload_gpus)
    gpu_map: dict[str, tuple[AcceleratorVendor, str]] = {}
    for gpu_type in response["data"]["gpuTypes"]:
        try:
            vendor = AcceleratorVendor.cast(gpu_type["manufacturer"])
        except ValueError:
            continue
        gpu_name = get_gpu_name(vendor, gpu_type["displayName"])
        if gpu_name:
            gpu_map[gpu_type["id"]] = (vendor, gpu_name)
    return gpu_map


def get_gpu_name(vendor: AcceleratorVendor, name: str) -> Optional[str]:
    if vendor == AcceleratorVendor.NVIDIA:
        return get_nvidia_gpu_name(name)
    if vendor == AcceleratorVendor.AMD:
        return get_amd_gpu_name(name)
    return None


def get_nvidia_gpu_name(name: str) -> Optional[str]:
    if "B200" in name:
        return "B200"
    if "V100" in name:
        return "V100"
    if name == "H100 NVL":
        return "H100NVL"
    if name.startswith(("A", "L", "H")):
        gpu_name, _, _ = name.partition(" ")
        return gpu_name
    if name.startswith("RTX A"):
        return name.lstrip("RTX ").replace(" ", "")
    if name.startswith("RTX"):
        return name.replace(" ", "")
    return None


def get_amd_gpu_name(name: str) -> Optional[str]:
    for gpu in KNOWN_AMD_GPUS:
        if gpu.name == name:
            return name
    return None


GPU_MAP = get_gpu_map()

gpu_types_query = """
query GpuTypes {
  countryCodes
  dataCenters {
    id
    name
    listed
    __typename
  }
  gpuTypes {
    maxGpuCount
    maxGpuCount
    maxGpuCountCommunityCloud
    maxGpuCountSecureCloud
    minPodGpuCount
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    __typename
  }
}
"""

query_pod_types = """
query GpuTypes($lowestPriceInput: GpuLowestPriceInput, $gpuTypesInput: GpuTypeFilter) {
  gpuTypes(input: $gpuTypesInput) {
    lowestPrice(input: $lowestPriceInput) {
      minimumBidPrice
      uninterruptablePrice
      minVcpu
      minMemory
      stockStatus
      compliance
      countryCode
      __typename
    }
    maxGpuCount
    id
    displayName
    memoryInGb
    securePrice
    secureSpotPrice
    communityPrice
    communitySpotPrice
    oneMonthPrice
    threeMonthPrice
    sixMonthPrice
    secureSpotPrice
    __typename
  }
}
"""
