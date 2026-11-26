"""
Runpod GPU and CPU pods in Secure Cloud, queried from the REST API v2 catalog.

Offers depend on the query filter: the catalog only lists data centers with a host that can
fit the requested vCPU and RAM, and reports the minimum vCPU and RAM of such hosts.
Community Cloud and Instant Clusters are not supported.
"""

import functools
import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gpuhunt._internal.constraints import correct_gpu_memory_gib, find_accelerators, matches
from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, CPUArchitecture, QueryFilter
from gpuhunt.providers.base import OnlineProvider, get_creds_env

logger = logging.getLogger(__name__)

API_URL = "https://api.runpod.io/v2"
TIMEOUT = 30
MAX_WORKERS = 8
# The CPU pod container disk limit is vCPU count x this value in GB. Pod creation rejects larger
# disks, but the v2 catalog does not expose the limit, so it is hardcoded per CPU flavor group.
CPU_DISK_LIMIT_PER_VCPU_BY_GROUP = {
    "CPU3": 10,
    "CPU5": 15,
}
NO_AVAILABILITY = "NONE"

T = TypeVar("T")
R = TypeVar("R")


class RunpodProvider(OnlineProvider):
    NAME = "runpod"
    # Minimum CUDA version on the host. Used to filter available offers
    # and should also be used when provisioning pods.
    MIN_CUDA_VERSION = "12.8"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "RunpodProvider":
        return cls(api_key=get_creds_env("RUNPOD_API_KEY"))

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        q = query_filter or QueryFilter()
        if q.spot or q.cpu_arch not in (None, CPUArchitecture.X86):
            return []
        fetchers: list[Callable[[_Client, QueryFilter], list[CatalogItem]]] = []
        if q.max_gpu_count is None or q.max_gpu_count > 0:
            fetchers.append(_get_gpu_items)
        if _cpu_items_requested(q):
            fetchers.append(_get_cpu_items)
        client = _Client(self.api_key)
        started_at = time.monotonic()
        try:
            results = _map_all(lambda fetch: fetch(client, q), fetchers)
        finally:
            client.close()
        items = [item for result in results if result is not None for item in result]
        logger.debug(
            "Fetched %d Runpod offers with %d requests in %.2fs for %s",
            len(items),
            client.requests,
            time.monotonic() - started_at,
            q,
        )
        return sorted(items, key=lambda i: i.price)


@dataclass(frozen=True)
class _GpuType:
    id: str
    vendor: AcceleratorVendor
    name: str
    memory: float
    max_count: int
    price: float


def _get_gpu_items(client: "_Client", q: QueryFilter) -> list[CatalogItem]:
    """
    Requests `/catalog/gpus` once per GPU count. The first request also returns GPU types
    and their max counts, which limit the remaining counts to request.
    """
    min_count = max(q.min_gpu_count or 1, 1)
    gpus_by_count: dict[int, list[dict]] = {min_count: _get_gpus(client, q, min_count)}
    gpu_types: dict[str, _GpuType] = {}
    for gpu in gpus_by_count[min_count]:
        gpu_type = _parse_gpu_type(gpu)
        if gpu_type is not None and _gpu_type_matches(gpu_type, q):
            gpu_types[gpu_type.id] = gpu_type
    max_count = max((t.max_count for t in gpu_types.values()), default=0)
    if q.max_gpu_count is not None:
        max_count = min(max_count, q.max_gpu_count)
    counts = list(range(min_count + 1, max_count + 1))
    for count, gpus in zip(counts, _map_all(lambda c: _get_gpus(client, q, c), counts)):
        if gpus is not None:
            gpus_by_count[count] = gpus

    items = []
    for count, gpus in gpus_by_count.items():
        for gpu in gpus:
            gpu_type = gpu_types.get(gpu["id"])
            if gpu_type is not None:
                items.extend(_make_gpu_items(gpu_type, gpu, count))
    return items


def _get_gpus(client: "_Client", q: QueryFilter, count: int) -> list[dict]:
    params: dict[str, str | int] = {
        "include": "AVAILABILITY",
        "product": "POD",
        "cloud": "SECURE",
        "minCudaVersion": RunpodProvider.MIN_CUDA_VERSION,
        "count": count,
    }
    # The filters are per GPU and are multiplied by `count`, the same way as in pod creation.
    # Rounding up guarantees that `count` x filter satisfies the requested minimum.
    if q.min_cpu:
        params["minVcpuCountPerGpu"] = math.ceil(q.min_cpu / count)
    if q.min_memory:
        params["minRamPerGpu"] = math.ceil(q.min_memory / count)
    return client.get("catalog/gpus", params)["gpus"]


def _parse_gpu_type(gpu: dict) -> _GpuType | None:
    if not gpu.get("secure"):
        return None
    max_count = (gpu.get("maxCount") or {}).get("secure") or 0
    price = (gpu.get("price") or {}).get("secure") or 0
    if max_count < 1 or price <= 0:
        return None
    try:
        vendor = AcceleratorVendor.cast(gpu["manufacturer"])
    except ValueError:
        return None
    name = _get_gpu_name(vendor, gpu["name"])
    if name is None:
        logger.debug("Skipping unsupported Runpod GPU %s", gpu["id"])
        return None
    memory = _get_gpu_memory(gpu["id"], name, gpu["memory"])
    return _GpuType(
        id=gpu["id"], vendor=vendor, name=name, memory=memory, max_count=max_count, price=price
    )


@functools.lru_cache
def _get_gpu_memory(gpu_id: str, name: str, memory_gb: float) -> int:
    # Cached to warn about unknown GPUs once per process rather than on every query
    return correct_gpu_memory_gib(
        name, memory_gb * 1024, context=f"provider={RunpodProvider.NAME}, gpu={gpu_id}"
    )


def _gpu_type_matches(gpu_type: _GpuType, q: QueryFilter) -> bool:
    """Checks the filters that do not depend on the GPU count or the host."""
    item = CatalogItem(
        provider=RunpodProvider.NAME,
        instance_name=gpu_type.id,
        location="",
        price=gpu_type.price,
        cpu=0,
        memory=0,
        gpu_vendor=gpu_type.vendor,
        gpu_count=1,
        gpu_name=gpu_type.name,
        gpu_memory=gpu_type.memory,
        spot=False,
        disk_size=None,
    )
    type_filter = QueryFilter(
        gpu_vendor=q.gpu_vendor,
        gpu_name=q.gpu_name,
        min_gpu_memory=q.min_gpu_memory,
        max_gpu_memory=q.max_gpu_memory,
        min_compute_capability=q.min_compute_capability,
        max_compute_capability=q.max_compute_capability,
    )
    return matches(item, type_filter)


def _make_gpu_items(gpu_type: _GpuType, gpu: dict, count: int) -> list[CatalogItem]:
    if count > gpu_type.max_count or gpu.get("availability", NO_AVAILABILITY) == NO_AVAILABILITY:
        return []
    # The minimums of hosts that satisfy the filters, across all listed data centers
    cpu = gpu.get("minVcpuCount")
    memory = gpu.get("minRam")
    if not cpu or not memory:
        logger.debug("Skipping Runpod GPU %s x%d without minVcpuCount or minRam", gpu["id"], count)
        return []
    return [
        CatalogItem(
            provider=RunpodProvider.NAME,
            instance_name=gpu_type.id,
            location=dc["id"],
            price=gpu_type.price * count,
            cpu=cpu,
            memory=memory,
            gpu_vendor=gpu_type.vendor,
            gpu_count=count,
            gpu_name=gpu_type.name,
            gpu_memory=gpu_type.memory,
            spot=False,
            disk_size=None,
        )
        for dc in _available_data_centers(gpu)
    ]


def _cpu_items_requested(q: QueryFilter) -> bool:
    """Whether CPU-only items can match `q`, see `matches`."""
    if q.min_gpu_count == 0:
        return True
    if q.min_gpu_count is not None:
        return False
    return not (
        q.gpu_vendor
        or q.gpu_name is not None
        or q.min_compute_capability is not None
        or q.max_compute_capability is not None
        or q.min_gpu_memory
        or q.min_total_gpu_memory
    )


def _get_cpu_items(client: "_Client", q: QueryFilter) -> list[CatalogItem]:
    """
    Requests `/catalog/cpus` once per vCPU count. vCPU counts are powers of two.
    The first request also returns CPU flavors and their max vCPU counts, which limit
    the remaining vCPU counts to request.
    """
    min_vcpu = _next_power_of_two(max(q.min_cpu or 2, 2))
    cpus_by_vcpu: dict[int, list[dict]] = {min_vcpu: _get_cpus(client, min_vcpu)}
    max_vcpu = max((cpu["vcpu"]["max"] for cpu in cpus_by_vcpu[min_vcpu]), default=0)
    if q.max_cpu is not None:
        max_vcpu = min(max_vcpu, q.max_cpu)
    vcpus: list[int] = []
    vcpu = min_vcpu * 2
    while vcpu <= max_vcpu:
        vcpus.append(vcpu)
        vcpu *= 2
    for vcpu, cpus in zip(vcpus, _map_all(lambda v: _get_cpus(client, v), vcpus)):
        if cpus is not None:
            cpus_by_vcpu[vcpu] = cpus

    items = []
    for vcpu, cpus in cpus_by_vcpu.items():
        for cpu in cpus:
            items.extend(_make_cpu_items(cpu, vcpu))
    return items


def _get_cpus(client: "_Client", vcpu: int) -> list[dict]:
    params: dict[str, str | int] = {
        "include": "AVAILABILITY",
        "product": "POD",
        "vcpuCount": vcpu,
    }
    return client.get("catalog/cpus", params)["cpus"]


def _make_cpu_items(cpu: dict, vcpu: int) -> list[CatalogItem]:
    if not cpu["vcpu"]["min"] <= vcpu <= cpu["vcpu"]["max"]:
        return []
    if cpu.get("availability", NO_AVAILABILITY) == NO_AVAILABILITY:
        return []
    price_per_vcpu = (cpu.get("price") or {}).get("securePerVcpu")
    if not price_per_vcpu:
        return []
    disk_limit_per_vcpu = CPU_DISK_LIMIT_PER_VCPU_BY_GROUP.get(cpu["group"])
    if disk_limit_per_vcpu is None:
        logger.warning("Skipping Runpod CPU %s: unknown disk limit of %s", cpu["id"], cpu["group"])
        return []
    memory = vcpu * cpu["ramGbPerVcpu"]
    return [
        CatalogItem(
            provider=RunpodProvider.NAME,
            # dstack parses the flavor ID from this name
            instance_name=f"{cpu['id']}-{vcpu}-{memory:g}",
            location=dc["id"],
            price=price_per_vcpu * vcpu,
            cpu=vcpu,
            memory=memory,
            gpu_count=0,
            gpu_name=None,
            gpu_memory=None,
            spot=False,
            # The maximum container disk size
            disk_size=float(vcpu * disk_limit_per_vcpu),
        )
        for dc in _available_data_centers(cpu)
    ]


def _available_data_centers(resource: dict) -> list[dict]:
    return [
        dc
        for dc in resource.get("dataCenters") or []
        if dc.get("availability", NO_AVAILABILITY) != NO_AVAILABILITY
    ]


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _map_all(func: Callable[[T], R], args: Iterable[T]) -> list[R | None]:
    """
    Calls `func` for each of `args` concurrently. A failed call is logged and its result
    is `None`. Raises the first error if all calls failed.
    """
    args = list(args)
    if not args:
        return []
    with ThreadPoolExecutor(max_workers=min(len(args), MAX_WORKERS)) as executor:
        futures = [executor.submit(func, arg) for arg in args]
    results: list[R | None] = []
    errors: list[Exception] = []
    for future in futures:
        try:
            results.append(future.result())
        except Exception as e:
            logger.error("Failed to fetch Runpod offers: %r", e)
            errors.append(e)
            results.append(None)
    if len(errors) == len(args):
        raise errors[0]
    return results


class _Client:
    def __init__(self, api_key: str):
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        # GPU and CPU requests run concurrently, each with up to MAX_WORKERS threads
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=2 * MAX_WORKERS)
        self._session.mount("https://", adapter)
        # `get` is called from multiple threads
        self._lock = threading.Lock()
        self.requests = 0

    def get(self, path: str, params: dict) -> dict:
        with self._lock:
            self.requests += 1
        resp = self._session.get(f"{API_URL}/{path}", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._session.close()


def _get_gpu_name(vendor: AcceleratorVendor, name: str) -> str | None:
    if vendor == AcceleratorVendor.NVIDIA:
        return _get_nvidia_gpu_name(name)
    if vendor == AcceleratorVendor.AMD:
        return _get_amd_gpu_name(name)
    return None


def _get_nvidia_gpu_name(name: str) -> str | None:
    # Match the full GPU only; B300 MIG slices have different resources.
    if name == "B300":
        return "B300"
    if "B200" in name:
        return "B200"
    if "V100" in name:
        return "V100"
    if name == "H100 NVL":
        return "H100NVL"
    if "H200 NVL" in name:
        return "H200NVL"
    if name.startswith(("A", "L", "H")):
        gpu_name, _, _ = name.partition(" ")
        return gpu_name
    if name.startswith("RTX A"):
        return name.lstrip("RTX ").replace(" ", "")
    if name.startswith("RTX"):
        return name.replace(" ", "")
    return None


def _get_amd_gpu_name(name: str) -> str | None:
    if accelerators := find_accelerators(names=[name], vendors=[AcceleratorVendor.AMD]):
        return accelerators[0].name
    return None
