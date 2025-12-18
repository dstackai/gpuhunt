import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, cast

import requests
from requests import RequestException
from typing_extensions import NotRequired, TypedDict

from gpuhunt._internal.constraints import find_accelerators
from gpuhunt._internal.models import AcceleratorVendor, QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
API_URL = "https://api.runpod.io/graphql"


class RunpodCatalogItemProviderData(TypedDict):
    # `pod_counts` is the number of pods that can be deployed in a cluster for the given offer.
    # Used to distinguish Runpod Clusters offers and check if multinode runs fit into clusters.
    pod_counts: NotRequired[list[int]]


class RunpodProvider(AbstractProvider):
    NAME = "runpod"
    # Minimum CUDA version on the host. Used to filter available offers
    # and should also be used when provisioning pods.
    MIN_CUDA_VERSION = "12.8"

    def __init__(self) -> None:
        self._gpu_map = get_gpu_map()

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        offers = self._fetch_offers()
        return sorted(offers, key=lambda i: i.price or 0)

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

    def _fetch_offers(self) -> list[RawCatalogItem]:
        query_variables = self._build_query_variables()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self._get_pods, query_variable)
                for query_variable in query_variables
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
                catalog_items.extend(self._make_catalog_items(query_variable, pod))

        cluster_catalog_items = self._fetch_cluster_offers()
        catalog_items.extend(cluster_catalog_items)
        return catalog_items

    def _build_query_variables(self) -> list[dict]:
        """Prepare different combinations of API query filters to cover all available GPUs."""

        gpu_types = _make_request({"query": gpu_types_query, "variables": {}})
        data_centers = [dc["id"] for dc in gpu_types["data"]["dataCenters"] if dc["listed"]]
        max_gpu_count = max(gpu["maxGpuCount"] for gpu in gpu_types["data"]["gpuTypes"])

        variables = []
        for gpu_count in range(1, max_gpu_count + 1):
            # Secure cloud is queryable by datacenter ID
            for dc_id in data_centers:
                variables.append(
                    {
                        "GpuTypeFilter": {
                            "cluster": False,
                        },
                        "lowestPriceInput": {
                            "secureCloud": True,
                            "dataCenterId": dc_id,
                            "gpuCount": gpu_count,
                            "minDisk": None,
                            "minMemoryInGb": None,
                            "minVcpuCount": None,
                            "minCudaVersion": RunpodProvider.MIN_CUDA_VERSION,
                        },
                    }
                )
            # Community cloud is queryable by country code
            for country_code in gpu_types["data"]["countryCodes"]:
                if country_code is None:
                    continue
                variables.append(
                    {
                        "GpuTypeFilter": {
                            "cluster": False,
                        },
                        "lowestPriceInput": {
                            "secureCloud": False,
                            "countryCode": country_code,
                            "gpuCount": gpu_count,
                            "minDisk": None,
                            "minMemoryInGb": None,
                            "minVcpuCount": None,
                            "minCudaVersion": RunpodProvider.MIN_CUDA_VERSION,
                        },
                    }
                )
        return variables

    def _get_pods(self, query_variables: dict) -> list[dict]:
        resp = _make_request(
            {
                "query": query_pod_types,
                "variables": query_variables,
            }
        )
        return resp["data"]["gpuTypes"]

    def _make_catalog_items(self, query_variables: dict, pod: dict) -> list[RawCatalogItem]:
        lowest_price_input_variables = query_variables["lowestPriceInput"]
        if pod["lowestPrice"]["stockStatus"] is None:
            return []
        listed_gpu_vendor_and_name = self._get_gpu_vendor_and_name(pod["id"])
        if listed_gpu_vendor_and_name is None:
            logger.warning(f"{pod['id']} missing in runpod GPU_MAP")
            return []
        if lowest_price_input_variables["secureCloud"]:
            location = lowest_price_input_variables["dataCenterId"]
            on_demand_gpu_price = pod["securePrice"]
            spot_gpu_price = pod["secureSpotPrice"]
        else:
            location = lowest_price_input_variables["countryCode"]
            on_demand_gpu_price = pod["communityPrice"]
            spot_gpu_price = pod["communitySpotPrice"]
        item_template = RawCatalogItem(
            instance_name=pod["id"],
            location=location,
            price=None,  # set below
            cpu=pod["lowestPrice"]["minVcpu"],
            memory=pod["lowestPrice"]["minMemory"],
            gpu_vendor=listed_gpu_vendor_and_name[0],
            gpu_count=lowest_price_input_variables["gpuCount"],
            gpu_name=listed_gpu_vendor_and_name[1],
            gpu_memory=pod["memoryInGb"],
            spot=None,  # set below
            disk_size=None,
            provider_data={},
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

    def _fetch_cluster_offers(self) -> list[RawCatalogItem]:
        cluster_catalog_items = []
        query_variables = {
            "gpuTypesInput": {
                "cluster": True,
            },
            "lowestPriceInput": {
                "gpuCount": 8,  # Needed to get CPU and RAM
            },
        }
        pod_type = self._get_pods(query_variables)
        for pod_type in pod_type:
            listed_gpu_vendor_and_name = self._get_gpu_vendor_and_name(pod_type["id"])
            if listed_gpu_vendor_and_name is None:
                logger.warning(f"{pod_type['id']} missing in runpod GPU_MAP")
                continue
            gpu_vendor, gpu_name = listed_gpu_vendor_and_name
            # Runpod returns no CPU and memory if the offer is out of stock.
            # Out of stock appears to be different from no capacity meaning
            # the offer is not available for a prolonged period.
            cpu = pod_type["lowestPrice"].get("minVcpu")
            memory = pod_type["lowestPrice"].get("minMemory")
            if cpu is None:
                logger.warning(f"{pod_type['id']} cluster offer missing minVcpu")
                continue
            if memory is None:
                logger.warning(f"{pod_type['id']} cluster offer missing minMemory")
                continue
            for location in pod_type["nodeGroupDatacenters"]:
                catalog_item = RawCatalogItem(
                    instance_name=pod_type["id"],
                    location=location["id"],
                    price=pod_type["clusterPrice"] * pod_type["maxGpuCount"],
                    cpu=cpu,
                    memory=memory,
                    gpu_vendor=gpu_vendor,
                    gpu_count=pod_type["maxGpuCount"],
                    gpu_name=gpu_name,
                    gpu_memory=pod_type["memoryInGb"],
                    spot=False,
                    disk_size=None,
                    flags=["runpod-cluster"],
                    # The API does not return supported pod counts but for now it's always 2-8 for all offers.
                    provider_data=cast(
                        dict, RunpodCatalogItemProviderData(pod_counts=list(range(2, 9)))
                    ),
                )
                cluster_catalog_items.append(catalog_item)
        return cluster_catalog_items

    def _get_gpu_vendor_and_name(
        self,
        gpu_id: str,
    ) -> Optional[tuple[AcceleratorVendor, str]]:
        if not gpu_id:
            return None
        return self._gpu_map.get(gpu_id)


def get_gpu_map() -> dict[str, tuple[AcceleratorVendor, str]]:
    payload_gpus = {
        "query": "query GpuTypes { gpuTypes { id manufacturer displayName memoryInGb } }"
    }
    response = _make_request(payload_gpus)
    gpu_map: dict[str, tuple[AcceleratorVendor, str]] = {}
    for gpu_type in response["data"]["gpuTypes"]:
        try:
            vendor = AcceleratorVendor.cast(gpu_type["manufacturer"])
        except ValueError:
            continue
        gpu_name = _get_gpu_name(vendor, gpu_type["displayName"])
        if gpu_name:
            gpu_map[gpu_type["id"]] = (vendor, gpu_name)
    return gpu_map


def _make_request(payload: dict):
    resp = requests.post(API_URL, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_gpu_name(vendor: AcceleratorVendor, name: str) -> Optional[str]:
    if vendor == AcceleratorVendor.NVIDIA:
        return _get_nvidia_gpu_name(name)
    if vendor == AcceleratorVendor.AMD:
        return _get_amd_gpu_name(name)
    return None


def _get_nvidia_gpu_name(name: str) -> Optional[str]:
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


def _get_amd_gpu_name(name: str) -> Optional[str]:
    if accelerators := find_accelerators(names=[name], vendors=[AcceleratorVendor.AMD]):
        return accelerators[0].name
    return None


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
    nodeGroupDatacenters {
        id
    }
    clusterPrice
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
