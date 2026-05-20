import logging
import os

import requests
from requests import Response

from gpuhunt._internal.models import AcceleratorVendor, QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

API_URL = "https://backendprod.jarvislabs.net"
SERVER_META_PATH = "/misc/server_meta"
TIMEOUT = 30
# JarvisLabs exposes offer regions in server_meta, but VM provisioning calls must be sent
# to region-specific API hosts and server_meta does not include those hosts. Keep this
# allowlist in sync with the known provisioning hosts and do not advertise offers for
# unknown regions, otherwise dstack may select capacity it cannot create.
JARVISLABS_REGION_URLS = {
    "india-01": "https://backendprod.jarvislabs.net",
    "india-noida-01": "https://backendn.jarvislabs.net",
    "europe-01": "https://backendeu.jarvislabs.net",
}
# dstack provisions JarvisLabs GPU VMs by passing a GPU type back to the API.
# Keep ambiguous API names with spaces out of the catalog; otherwise the
# normalized gpuhunt name cannot be converted back safely without provider_data.
JARVISLABS_GPU_NAME_OVERRIDES = {
    "A100-80GB": ("A100", 80.0),
}


class JarvisLabsProvider(AbstractProvider):
    NAME = "jarvislabs"

    def __init__(self, api_key: str | None = None, api_url: str | None = None):
        self.api_key = api_key or os.getenv("JL_API_KEY")
        if not self.api_key:
            raise ValueError("Set the JL_API_KEY environment variable.")

        self.api_url = (api_url or os.getenv("JARVISLABS_API_URL", API_URL)).rstrip("/")

    def get(
        self, query_filter: QueryFilter | None = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        offers = self.fetch_offers(query_filter=query_filter)
        return sorted(offers, key=lambda i: i.price)

    def fetch_offers(self, query_filter: QueryFilter | None = None) -> list[RawCatalogItem]:
        response = self._make_request("GET", SERVER_META_PATH)
        return convert_response_to_raw_catalog_items(response.json())

    def _make_request(self, method: str, path: str) -> Response:
        response = requests.request(
            method=method,
            url=f"{self.api_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response


def convert_response_to_raw_catalog_items(data: dict) -> list[RawCatalogItem]:
    offers = []
    for gpu in data.get("server_meta") or []:
        offers.extend(_make_gpu_catalog_items(gpu))
    offers.extend(_make_cpu_catalog_items(data.get("cpu_meta") or {}))
    return offers


def _make_gpu_catalog_items(gpu: dict) -> list[RawCatalogItem]:
    region = gpu.get("region")
    if not region:
        return []
    workload_type = gpu.get("workload_type")
    # JarvisLabs returns `None` for older VM-capable rows, e.g. EU H100/H200.
    # Confirmed by provisioning an H100 VM from a `None` row.
    if workload_type not in ("vm", None):
        return []
    if region not in JARVISLABS_REGION_URLS:
        logger.warning(
            "Skipping JarvisLabs GPU VM offer in unsupported region %s; "
            "JarvisLabs does not expose provisioning endpoint discovery",
            region,
        )
        return []

    gpu_type = gpu.get("gpu_type")
    if not gpu_type:
        logger.warning("Skipping JarvisLabs GPU offer without gpu_type: %s", gpu)
        return []

    price = _as_float(gpu.get("price_per_hour"))
    if price is None:
        logger.warning("Skipping JarvisLabs GPU offer without price: %s", gpu_type)
        return []

    gpu_spec = _gpu_name_and_memory(gpu_type, gpu.get("vram"))
    if gpu_spec is None:
        logger.warning("Skipping JarvisLabs GPU offer with ambiguous gpu_type: %s", gpu_type)
        return []
    gpu_name, gpu_memory = gpu_spec
    if gpu_memory is None:
        logger.warning("Skipping JarvisLabs GPU offer with unknown VRAM: %s", gpu_type)
        return []

    cpu_per_gpu = _as_int(gpu.get("cpus_per_gpu"))
    ram_per_gpu = _as_float(gpu.get("ram_per_gpu"))
    if cpu_per_gpu is None or ram_per_gpu is None:
        logger.warning("Skipping JarvisLabs GPU offer without CPU/RAM: %s", gpu_type)
        return []

    items = _make_gpu_catalog_items_for_price(
        region=region,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
        price=price,
        cpu_per_gpu=cpu_per_gpu,
        ram_per_gpu=ram_per_gpu,
        available_devices=_available_devices(gpu),
        max_gpus_per_instance=_max_gpus_per_instance(gpu),
        spot=False,
    )

    spot_price = _as_float(gpu.get("spot_price"))
    if spot_price is not None:
        items.extend(
            _make_gpu_catalog_items_for_price(
                region=region,
                gpu_name=gpu_name,
                gpu_memory=gpu_memory,
                price=spot_price,
                cpu_per_gpu=cpu_per_gpu,
                ram_per_gpu=ram_per_gpu,
                available_devices=_spot_available_devices(gpu),
                max_gpus_per_instance=_max_gpus_per_instance(gpu),
                spot=True,
            )
        )
    return items


def _make_gpu_catalog_items_for_price(
    *,
    region: str,
    gpu_name: str,
    gpu_memory: float,
    price: float,
    cpu_per_gpu: int,
    ram_per_gpu: float,
    available_devices: int,
    max_gpus_per_instance: int,
    spot: bool,
) -> list[RawCatalogItem]:
    items = []
    for gpu_count in _supported_gpu_counts(
        available_devices=available_devices,
        max_gpus_per_instance=max_gpus_per_instance,
    ):
        items.append(
            RawCatalogItem(
                instance_name=_gpu_instance_name(gpu_name, gpu_count),
                location=region,
                price=round(price * gpu_count, 5),
                cpu=cpu_per_gpu * gpu_count,
                memory=ram_per_gpu * gpu_count,
                gpu_vendor=AcceleratorVendor.NVIDIA.value,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                gpu_memory=gpu_memory,
                spot=spot,
                disk_size=None,
            )
        )
    return items


def _make_cpu_catalog_items(cpu_meta: dict) -> list[RawCatalogItem]:
    offers = []
    # The JarvisLabs SDK resolves CPU VMs from cpu_meta.combinations and creates them via
    # templates/vm/cpu/create; cpu_meta.workload_type is not the GPU workload selector.
    for combo in cpu_meta.get("combinations") or []:
        if not combo.get("available"):
            continue
        vcpus = _as_int(combo.get("vcpus"))
        ram_gb = _as_float(combo.get("ram_gb"))
        price = _as_float(combo.get("price"))
        if vcpus is None or ram_gb is None or price is None:
            logger.warning("Skipping JarvisLabs CPU offer with incomplete specs: %s", combo)
            continue
        for region, available in (combo.get("regions") or {}).items():
            if not available:
                continue
            if region not in JARVISLABS_REGION_URLS:
                logger.warning(
                    "Skipping JarvisLabs CPU VM offer in unsupported region %s; "
                    "JarvisLabs does not expose provisioning endpoint discovery",
                    region,
                )
                continue
            offers.append(
                RawCatalogItem(
                    instance_name=f"cpu-{vcpus}x{int(ram_gb)}",
                    location=region,
                    price=price,
                    cpu=vcpus,
                    memory=ram_gb,
                    gpu_vendor=None,
                    gpu_count=0,
                    gpu_name=None,
                    gpu_memory=None,
                    spot=False,
                    disk_size=None,
                )
            )
    return offers


def _supported_gpu_counts(*, available_devices: int, max_gpus_per_instance: int) -> list[int]:
    if available_devices <= 0 or max_gpus_per_instance <= 0:
        return []
    return list(range(1, min(available_devices, max_gpus_per_instance) + 1))


def _available_devices(gpu: dict) -> int:
    return (
        _as_int(gpu.get("effective_num_free_devices")) or _as_int(gpu.get("num_free_devices")) or 0
    )


def _spot_available_devices(gpu: dict) -> int:
    return _as_int(gpu.get("spot_num_free_devices")) or 0


def _max_gpus_per_instance(gpu: dict) -> int:
    return _as_int(gpu.get("num_gpus")) or 1


def _gpu_name_and_memory(gpu_type: str, vram: object) -> tuple[str, float | None] | None:
    if any(c.isspace() for c in gpu_type):
        return None
    gpu_name, default_memory = JARVISLABS_GPU_NAME_OVERRIDES.get(gpu_type, (gpu_type, None))
    return gpu_name, _as_float(vram) or default_memory


def _gpu_instance_name(gpu_name: str, gpu_count: int) -> str:
    return f"{gpu_name}-{gpu_count}x"


def _as_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
