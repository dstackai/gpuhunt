import logging
import math
import re
from typing import cast

import requests
from typing_extensions import NotRequired, TypedDict

from gpuhunt._internal.constraints import correct_gpu_memory_gib, find_accelerators
from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, JSONObject, QueryFilter
from gpuhunt.providers.base import OnlineProvider

logger = logging.getLogger(__name__)

# Public, unauthenticated feed of every node currently rentable on Lium. Documented at
# https://docs.lium.io/developers/public-nodes-feed with a compatibility promise: fields are only
# ever added under /public/v1, breaking changes ship as /public/v2.
NODES_URL = "https://lium.io/api/public/v1/nodes"
TIMEOUT = 30


class LiumProvider(OnlineProvider):
    """Online provider for Lium (https://lium.io) GPU pods.

    Lium publishes its whole rentable inventory as one public JSON feed with per-GPU hourly
    prices, so no credentials are needed. Any GPU count between ``min_rentable_gpu_count`` and
    ``available_gpu_count`` is a valid order and gets a proportional share of the host's CPUs,
    free disk and RAM (after the RAM the host keeps for itself). Memory and disk are lower bounds
    in whole GiB/GB, so a filter never matches an offer that cannot hold the request. Most nodes publish
    ``min_rentable_gpu_count == gpu_count``, so they get a single whole-node offer. One offer is
    produced per rentable GPU
    count so that requests for 1, 2, ... GPUs all match, as the jarvislabs provider does.
    """

    NAME = "lium"

    @classmethod
    def from_env(cls) -> "LiumProvider":
        return cls()

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        response = requests.get(NODES_URL, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
            raise ValueError("Unexpected response from Lium public nodes feed")

        offers: list[CatalogItem] = []
        for node in data["nodes"]:
            if not isinstance(node, dict):
                logger.warning("Skipping malformed Lium node: %r", node)
                continue
            offers.extend(_make_offers(node))
        return sorted(offers, key=lambda i: i.price)


class LiumCatalogItemProviderData(TypedDict):
    # Deep link to the rent page of the node, as published in the feed.
    rent_url: NotRequired[str]


def _make_offers(node: dict) -> list[CatalogItem]:
    node_id = node.get("id")
    if not node_id:
        logger.warning("Skipping Lium node without id: %r", node)
        return []
    node_id = str(node_id)

    # Only `id` and `rent_url` are guaranteed non-null by the feed contract.
    try:
        gpu_model = str(_required(node, "gpu_model"))
        gpu_count = int(_required(node, "gpu_count"))
        available_gpu_count = int(_required(node, "available_gpu_count"))
        price_per_gpu_hour = float(_required(node, "price_per_gpu_hour"))
        gpu_memory_gb = float(_required(node, "gpu_memory_gb"))
        cpu_count = int(_required(node, "cpu_count"))
        ram_gb = float(_required(node, "ram_gb"))
    except (TypeError, ValueError) as e:
        logger.warning("Skipping Lium node %s: %s", node_id, e)
        return []
    disk_free_gb = node.get("disk_free_gb")
    if (
        isinstance(disk_free_gb, bool)
        or not isinstance(disk_free_gb, int | float)
        or disk_free_gb <= 0
    ):
        # with disk_size=None an offer would pass every disk filter, so no free disk means no offer
        logger.warning("Skipping Lium node %s: no free disk reported", node_id)
        return []
    if gpu_count <= 0 or available_gpu_count <= 0 or price_per_gpu_hour <= 0:
        logger.warning("Skipping Lium node %s: non-positive GPU count or price", node_id)
        return []
    available_gpu_count = min(available_gpu_count, gpu_count)
    # The feed publishes the smallest order checkout accepts, and sets it to `gpu_count` wherever
    # the node cannot be split. If it is ever missing, only the whole node is a safe offer.
    min_gpu_count = max(1, int(node.get("min_rentable_gpu_count") or gpu_count))
    if min_gpu_count > available_gpu_count:
        logger.debug(
            "Skipping Lium node %s: %d GPUs free, %d is the smallest order",
            node_id,
            available_gpu_count,
            min_gpu_count,
        )
        return []

    location = get_location(node.get("country_code"), node.get("region"))
    if not location:
        logger.warning("Skipping Lium node %s: no location", node_id)
        return []
    gpu_name = get_dstack_gpu_name(gpu_model)
    if not find_accelerators(names=[gpu_name], vendors=[AcceleratorVendor.NVIDIA]):
        logger.warning("Skipping Lium node %s: unknown gpu_model %r", node_id, gpu_model)
        return []
    gpu_memory = correct_gpu_memory_gib(
        gpu_name,
        gpu_memory_gb * 1024,
        context=f"provider={LiumProvider.NAME}, node={node_id}, location={location}",
    )
    provider_data = LiumCatalogItemProviderData()
    rent_url = node.get("rent_url")
    if rent_url:
        provider_data["rent_url"] = str(rent_url)

    # The feed rounds RAM and free disk to the nearest GB, so start from the lowest true value.
    host_ram_gb = ram_gb - 0.5
    pod_ram_gb = host_ram_gb - get_host_ram_reserve_gb(host_ram_gb)
    disk_free_gb = max(disk_free_gb - 0.5, 0)
    offers: list[CatalogItem] = []
    for count in range(min_gpu_count, available_gpu_count + 1):
        # A rental gets the host's CPUs, RAM (after the host reserve) and free disk in
        # proportion to its GPU share.
        share = count / gpu_count
        cpu = int(cpu_count * share)
        # Lium sizes the pod's memory limit in whole GiB, rounded down.
        memory = max(1, math.floor(pod_ram_gb * share))
        if cpu < 1:
            logger.warning(
                "Skipping Lium node %s with %d GPU(s): less than one CPU",
                node_id,
                count,
            )
            continue
        disk_size = get_storage_quota_gb(disk_free_gb * share)
        if disk_size < 1:
            logger.debug(
                "Skipping Lium node %s with %d GPU(s): its storage quota rounds to zero",
                node_id,
                count,
            )
            continue
        offers.append(
            CatalogItem(
                provider=LiumProvider.NAME,
                instance_name=node_id,
                location=location,
                price=round(price_per_gpu_hour * count, 5),
                cpu=cpu,
                memory=float(memory),
                gpu_vendor=AcceleratorVendor.NVIDIA,
                gpu_count=count,
                gpu_name=gpu_name,
                gpu_memory=float(gpu_memory),
                spot=node.get("tier") == "spot",
                disk_size=float(disk_size),
                provider_data=cast(JSONObject, dict(provider_data)),
            )
        )
    return offers


def get_storage_quota_gb(disk_gb: float) -> int:
    """
    Disk in GB available to a job on a pod with this share of the host's free disk.

    A pod's share is split into a volume (2/3) and the container storage quota (1/3), each rounded
    down, as documented at https://docs.lium.io/pod-users/create-pod. Jobs run on the container
    filesystem, so only the storage quota counts.

    >>> get_storage_quota_gb(1466.5 / 8)
    61
    >>> get_storage_quota_gb(2)
    0
    """
    return int(disk_gb / 3)


def get_host_ram_reserve_gb(ram_gb: float) -> float:
    """
    RAM in GiB that the host keeps for its own services before splitting the rest between GPUs.

    Documented at https://docs.lium.io/pod-users/create-pod as ``max(4, 1 % of host RAM)``.

    >>> get_host_ram_reserve_gb(16)
    4
    >>> round(get_host_ram_reserve_gb(2048), 2)
    20.48
    """
    return max(4, ram_gb * 0.01)


def _required(node: dict, name: str) -> str | int | float:
    value = node.get(name)
    if value is None or isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"missing {name}")
    return value


# Tokens that describe a form factor, memory size or marketing line rather than the GPU itself.
_GPU_NAME_NOISE = re.compile(
    r"\b(nvidia|geforce|tesla|quadro|tensor core gpu|generation|blackwell|server edition|"
    r"workstation edition|pcie|sxm\d?|hbm\d\w*|ac|pc|\d+\s*gb)\b",
    flags=re.IGNORECASE,
)


def get_dstack_gpu_name(gpu_model: str) -> str:
    """
    Convert a Lium ``gpu_model`` to the name used in gpuhunt's `KNOWN_NVIDIA_GPUS`.

    The result is only published if it is a known GPU, see `_make_offers`.

    >>> get_dstack_gpu_name("RTX 5090")
    'RTX5090'
    >>> get_dstack_gpu_name("H100 80GB HBM3")
    'H100'
    >>> get_dstack_gpu_name("A100-SXM4-80GB")
    'A100'
    >>> get_dstack_gpu_name("B300 SXM6 AC")
    'B300'
    >>> get_dstack_gpu_name("B300 SXM6 PC")
    'B300'
    >>> get_dstack_gpu_name("RTX PRO 6000 Blackwell Server Edition")
    'RTXPRO6000'
    >>> get_dstack_gpu_name("RTX 6000 Ada Generation")
    'RTX6000Ada'
    >>> get_dstack_gpu_name("RTX A6000")
    'A6000'
    >>> get_dstack_gpu_name("H200 NVL")
    'H200NVL'
    >>> get_dstack_gpu_name("Tesla V100 Tensor Core GPU")
    'V100'
    """
    name = gpu_model.replace("-", " ")
    name = _GPU_NAME_NOISE.sub(" ", name)
    name = re.sub(r"^RTX A(\d)", r"A\1", name.strip())
    return re.sub(r"\s+", "", name)


def get_location(country_code: object, region: object) -> str:
    """
    Build a location such as ``us-ca`` from the feed's ISO country code and subdivision code.

    >>> get_location("US", "CA")
    'us-ca'
    >>> get_location("JP", "13")
    'jp-13'
    >>> get_location("DE", None)
    'de'
    >>> get_location(None, "CA")
    ''
    """
    if not country_code:
        return ""
    location = str(country_code).strip().lower()
    if region:
        location = f"{location}-{str(region).strip().lower()}"
    return location.replace(" ", "")
