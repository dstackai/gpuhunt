import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from typing import List, Optional

import requests
from requests import RequestException

from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
API_URL = "https://api.runpod.io/graphql"


class RunpodProvider(AbstractProvider):
    NAME = "runpod"

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        offers = self.fetch_offers()
        return sorted(offers, key=lambda i: i.price)

    def fetch_offers(self) -> List[RawCatalogItem]:
        offers = [get_raw_catalog(pod_type) for pod_type in self.list_pods()]
        return list(chain.from_iterable(offers))

    @staticmethod
    def list_pods() -> List[dict]:
        payload_gpu_types = {"query": gpu_types_query, "variables": {}}
        try:
            gpu_types = make_request(payload_gpu_types)
        except RequestException as e:
            logger.exception("Failed to make request for GPU types: %s", e)
            raise

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(get_pods, query_variable)
                for query_variable in build_query_variables(gpu_types)
            ]
        offers = []
        for future in as_completed(futures):
            try:
                result = future.result()
                offers.append(result)
            except RequestException as e:
                logger.exception("Failed to get pods data: %s", e)
        return list(chain.from_iterable(offers))


def gpu_name(name: str) -> Optional[str]:
    if not name:
        return None
    result = GPU_MAP.get(name)
    return result


def build_query_variables(gpu_types: List[dict]) -> List[dict]:
    # Filter dataCenters by 'listed: True'
    listed_data_centers = [dc["id"] for dc in gpu_types["data"]["dataCenters"] if dc["listed"]]

    # Find the maximum of maxGpuCount
    max_gpu_count = max(gpu["maxGpuCount"] for gpu in gpu_types["data"]["gpuTypes"])

    # Generate the variables list
    variables = []
    for dc_id in listed_data_centers:
        for gpu_count in range(1, max_gpu_count + 1):
            variables.append(
                {
                    "dataCenterId": dc_id,
                    "gpuCount": gpu_count,  # gpuCount is mandatory
                    "minDisk": None,
                    "minMemoryInGb": None,
                    "minVcpuCount": None,
                    "secureCloud": None,
                }
            )

    return variables


def get_pods(query_variable: dict) -> List[dict]:
    pods = make_request(get_pods_query_payload(query_variable))["data"]["gpuTypes"]
    offers = []
    for pod in pods:
        listed_gpu_name = gpu_name(pod["id"])
        availability = pod["lowestPrice"]["stockStatus"]
        if listed_gpu_name is not None and availability is not None:
            offers.append(
                get_offers(pod, query_variable["dataCenterId"], query_variable["gpuCount"])
            )
        elif listed_gpu_name is None and availability is not None:
            logger.warning(f"{pod['id']} missing in runpod GPU_MAP")
    return offers


def get_offers(pod: dict, data_center_id, gpu_count) -> dict:
    offer = {
        "id": pod["id"],
        "data_center_id": data_center_id,
        "secure_price": pod["securePrice"],
        "secure_spot_price": pod["secureSpotPrice"],
        "community_price": pod["communityPrice"],
        "community_spot_price": pod["communitySpotPrice"],
        "cpu": pod["lowestPrice"]["minVcpu"],
        "memory": pod["lowestPrice"]["minMemory"],
        "gpu": gpu_count,
        "display_name": pod["displayName"],
        "gpu_memory": pod["memoryInGb"],
        "gpu_name": gpu_name(pod["id"]),
    }
    return offer


def get_pods_query_payload(query_variable: dict) -> dict:
    payload_secure_gpu_types = {
        "query": query_pod_types,
        "variables": {"lowestPriceInput": query_variable},
    }
    return payload_secure_gpu_types


def make_request(payload: dict):
    resp = requests.post(API_URL, json=payload)
    if resp.ok:
        data = resp.json()
        return data
    resp.raise_for_status()


def get_raw_catalog(offer: dict) -> List[RawCatalogItem]:
    catalog_items = []

    # Check if both community_price and secure_price are present
    if offer["community_price"] is not None and offer["secure_price"] is not None:
        # Handle the secure_price on demand case
        catalog_items.append(
            RawCatalogItem(
                instance_name=offer["id"],
                location=offer["data_center_id"],
                price=float(offer["secure_price"] * offer["gpu"]),
                cpu=offer["cpu"],
                memory=offer["memory"],
                gpu_count=offer["gpu"],
                gpu_name=offer["gpu_name"],
                gpu_memory=offer["gpu_memory"],
                spot=False,
                disk_size=None,
            )
        )
        secure_spot = offer["secure_spot_price"] is not None and offer["secure_spot_price"] > 0
        if secure_spot:
            catalog_items.append(
                RawCatalogItem(
                    instance_name=offer["id"],
                    location=offer["data_center_id"],
                    price=float(offer["secure_spot_price"] * offer["gpu"]),
                    cpu=offer["cpu"],
                    memory=offer["memory"],
                    gpu_count=offer["gpu"],
                    gpu_name=offer["gpu_name"],
                    gpu_memory=offer["gpu_memory"],
                    spot=True,
                    disk_size=None,
                )
            )

        # Handle the community_price case
        catalog_items.append(
            RawCatalogItem(
                instance_name=offer["id"],
                location=offer["data_center_id"],
                price=float(offer["community_price"] * offer["gpu"]),
                cpu=offer["cpu"],
                memory=offer["memory"],
                gpu_count=offer["gpu"],
                gpu_name=offer["gpu_name"],
                gpu_memory=offer["gpu_memory"],
                spot=False,
                disk_size=None,
            )
        )
        community_spot = (
            offer["community_spot_price"] is not None and offer["community_spot_price"] > 0
        )
        if community_spot:
            catalog_items.append(
                RawCatalogItem(
                    instance_name=f'{offer["id"]}',
                    location=offer["data_center_id"],
                    price=float(offer["community_spot_price"] * offer["gpu"]),
                    cpu=offer["cpu"],
                    memory=offer["memory"],
                    gpu_count=offer["gpu"],
                    gpu_name=offer["gpu_name"],
                    gpu_memory=offer["gpu_memory"],
                    spot=True,
                    disk_size=None,
                )
            )
    else:
        # Handle the case where only one price is present
        price = (
            offer["secure_price"]
            if offer["secure_price"] is not None
            else offer["community_price"]
        )
        spot_price = (
            offer["secure_spot_price"]
            if offer["secure_price"] is not None
            else offer["community_spot_price"]
        )
        catalog_items.append(
            RawCatalogItem(
                instance_name=offer["id"],
                location=offer["data_center_id"],
                price=float(price * offer["gpu"]),
                cpu=offer["cpu"],
                memory=offer["memory"],
                gpu_count=offer["gpu"],
                gpu_name=offer["gpu_name"],
                gpu_memory=offer["gpu_memory"],
                spot=False,
                disk_size=None,
            )
        )
        spot = spot_price is not None and spot_price > 0
        if spot:
            catalog_items.append(
                RawCatalogItem(
                    instance_name=f'{offer["id"]}',
                    location=offer["data_center_id"],
                    price=float(spot_price * offer["gpu"]),
                    cpu=offer["cpu"],
                    memory=offer["memory"],
                    gpu_count=offer["gpu"],
                    gpu_name=offer["gpu_name"],
                    gpu_memory=offer["gpu_memory"],
                    spot=True,
                    disk_size=None,
                )
            )
    return catalog_items


def get_gpu_map():
    payload_gpus = {"query": "query GpuTypes { gpuTypes { id displayName memoryInGb } }"}
    gpu_types = make_request(payload_gpus)
    gpu_map = {
        item["id"]: get_gpu_name(item["displayName"])
        for item in gpu_types["data"]["gpuTypes"]
        if get_gpu_name(item["displayName"]) is not None
    }
    return gpu_map


def get_gpu_name(name: str) -> Optional[str]:
    if "V100" in name:
        return "V100"
    if name.startswith(("A", "L", "H")):
        gpu_name, _, _ = name.partition(" ")
        return gpu_name
    if name.startswith("RTX"):
        gpu_name = name.replace(" ", "")
        return gpu_name
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
