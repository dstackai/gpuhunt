import heapq
import io
import logging
import os
import threading
import time
import urllib.request
import zipfile
from collections.abc import Container
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import gpuhunt._internal.constraints as constraints
import gpuhunt._internal.storage as storage
from gpuhunt._internal.errors import GPUHuntError
from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, CPUArchitecture, QueryFilter
from gpuhunt._internal.utils import parse_compute_capability
from gpuhunt.providers.base import AbstractProvider

logger = logging.getLogger(__name__)

# Every provider is published independently under `{CATALOG_URL}/{provider}`, so a provider
# that fails to be collected does not hold back the others.
CATALOG_URL = "https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v3"
OFFLINE_PROVIDERS = [
    "aws",
    "azure",
    "verda",
    "gcp",
    "lambdalabs",
    "nebius",
    "oci",
    "runpod",
    "cloudrift",
]
ONLINE_PROVIDERS = [
    "crusoe",
    "digitalocean",
    "hotaisle",
    "jarvislabs",
    "seeweb",
    "vastai",
    "vultr",
]
RELOAD_INTERVAL = 15 * 60  # 15 minutes


@dataclass
class ProviderCatalog:
    version: str
    items: list[CatalogItem]


class Catalog:
    def __init__(self, balance_resources: bool = True, auto_reload: bool = True):
        """
        Args:
            balance_resources: increase min resources to better match the chosen GPU
            auto_reload: if `True`, the catalog will be automatically loaded from the S3 bucket every 4 hours
        """
        self.catalog: dict[str, ProviderCatalog] = {}
        self.loaded_at = None
        self.providers: list[AbstractProvider] = []
        self.balance_resources = balance_resources
        self.auto_reload = auto_reload
        self._load_lock = threading.Lock()

    def query(
        self,
        *,
        provider: str | list[str] | None = None,
        cpu_arch: CPUArchitecture | str | None = None,
        min_cpu: int | None = None,
        max_cpu: int | None = None,
        min_memory: float | None = None,
        max_memory: float | None = None,
        min_gpu_count: int | None = None,
        max_gpu_count: int | None = None,
        gpu_vendor: AcceleratorVendor | str | None = None,
        gpu_name: str | list[str] | None = None,
        min_gpu_memory: float | None = None,
        max_gpu_memory: float | None = None,
        min_total_gpu_memory: float | None = None,
        max_total_gpu_memory: float | None = None,
        min_disk_size: int | None = None,
        max_disk_size: int | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        min_compute_capability: str | tuple[int, int] | None = None,
        max_compute_capability: str | tuple[int, int] | None = None,
        spot: bool | None = None,
        allowed_flags: Container[str] | None = None,
    ) -> list[CatalogItem]:
        """
        Query the catalog for matching offers

        Args:
            provider: name of the provider to filter by. If not specified, all providers will be used
            cpu_arch: CPU architecture to filter by. If not specified, all architectures will be used
            min_cpu: minimum number of CPUs
            max_cpu: maximum number of CPUs
            min_memory: minimum amount of RAM in GB
            max_memory: maximum amount of RAM in GB
            min_gpu_count: minimum number of GPUs
            max_gpu_count: maximum number of GPUs
            gpu_vendor: accelerator vendor to filter by. If not specified, all vendors will be used
            gpu_name: name of the GPU to filter by. If not specified, all GPUs will be used
            min_gpu_memory: minimum amount of GPU VRAM in GB for each GPU
            max_gpu_memory: maximum amount of GPU VRAM in GB for each GPU
            min_total_gpu_memory: minimum amount of GPU VRAM in GB for all GPUs combined
            max_total_gpu_memory: maximum amount of GPU VRAM in GB for all GPUs combined
            min_disk_size: minimum disk size in GB
            max_disk_size: maximum disk size in GB
            min_price: minimum price per hour in USD
            max_price: maximum price per hour in USD
            min_compute_capability: minimum compute capability of the GPU
            max_compute_capability: maximum compute capability of the GPU
            spot: if `False`, only ondemand offers will be returned. If `True`, only spot offers will be returned
            allowed_flags: only offers with all flags allowed will be returned. `None` allows all flags

        Returns:
            list of matching offers
        """
        if self.auto_reload and (
            self.loaded_at is None or time.monotonic() - self.loaded_at > RELOAD_INTERVAL
        ):
            with self._load_lock:
                if self.auto_reload and (
                    self.loaded_at is None or time.monotonic() - self.loaded_at > RELOAD_INTERVAL
                ):
                    self._load()

        query_filter = QueryFilter(
            provider=[provider] if isinstance(provider, str) else provider,
            cpu_arch=CPUArchitecture.cast(cpu_arch) if cpu_arch else None,
            min_cpu=min_cpu,
            max_cpu=max_cpu,
            min_memory=min_memory,
            max_memory=max_memory,
            min_gpu_count=min_gpu_count,
            max_gpu_count=max_gpu_count,
            gpu_vendor=AcceleratorVendor.cast(gpu_vendor) if gpu_vendor else None,
            gpu_name=[gpu_name] if isinstance(gpu_name, str) else gpu_name,
            min_gpu_memory=min_gpu_memory,
            max_gpu_memory=max_gpu_memory,
            min_total_gpu_memory=min_total_gpu_memory,
            max_total_gpu_memory=max_total_gpu_memory,
            min_disk_size=min_disk_size,
            max_disk_size=max_disk_size,
            min_price=min_price,
            max_price=max_price,
            min_compute_capability=parse_compute_capability(min_compute_capability),
            max_compute_capability=parse_compute_capability(max_compute_capability),
            spot=spot,
            allowed_flags=allowed_flags,
        )

        if query_filter.provider is not None:
            # validate providers
            for p in query_filter.provider:
                if p.lower() not in OFFLINE_PROVIDERS + ONLINE_PROVIDERS:
                    raise ValueError(f"Unknown provider: {p}")
        else:
            query_filter.provider = OFFLINE_PROVIDERS + list(
                set(p.NAME for p in self.providers if p.NAME in ONLINE_PROVIDERS)
            )

        # fetch providers
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []

            for provider_name in ONLINE_PROVIDERS:
                if provider_name in map(str.lower, query_filter.provider):
                    futures.append(
                        executor.submit(
                            self._get_online_provider_items,
                            provider_name,
                            query_filter,
                        )
                    )

            for provider_name in OFFLINE_PROVIDERS:
                if provider_name in map(str.lower, query_filter.provider):
                    futures.append(
                        executor.submit(
                            self._get_offline_provider_items,
                            provider_name,
                            query_filter,
                        )
                    )

            completed, _ = wait(futures)
            # The merge preserves provider-specific order, picking the cheapest offer at each step.
            # The final list is not strictly sorted by the price.
            items = list(heapq.merge(*[f.result() for f in completed], key=lambda i: i.price))
        return items

    def load(self, version: str | None = None):
        """
        Fetch the catalogs of all offline providers from the S3 bucket. Thread-safe.

        Args:
            version: specific version of the catalogs to download. Applies to all providers.
                If not specified, the latest version of each provider will be used

        Raises:
            GPUHuntError: if no provider could be loaded, or if any provider could not be
                loaded at the requested `version`
        """
        with self._load_lock:
            self._load(version)

    def _load(self, version: str | None = None):
        base_url = _get_base_url()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                provider: executor.submit(self._load_provider, base_url, provider, version)
                for provider in OFFLINE_PROVIDERS
            }
        loaded: dict[str, ProviderCatalog] = {}
        loaded_providers = 0
        error: Exception | None = None
        for provider, future in futures.items():
            try:
                provider_catalog = future.result()
            except Exception as e:
                # A pinned version is requested as a whole. Falling back to the previously
                # loaded catalog would leave the caller with a mix of versions
                if version is not None:
                    raise GPUHuntError(f"Failed to load {provider} catalog {version}") from e
                logger.exception(
                    "Failed to load %s catalog. Keeping the previously loaded catalog, if any.",
                    provider,
                )
                error = e
                continue
            loaded_providers += 1
            if provider_catalog is not None:
                loaded[provider] = provider_catalog
        if error is not None and not loaded_providers:
            raise GPUHuntError("Failed to load catalogs of all providers") from error
        self.catalog.update(loaded)
        self.loaded_at = time.monotonic()

    def _load_provider(
        self, base_url: str, provider: str, version: str | None
    ) -> ProviderCatalog | None:
        """
        Returns:
            the downloaded catalog or `None` if the loaded catalog is already up-to-date
        """
        if version is None:
            version = _get_latest_version(base_url, provider)
        loaded_catalog = self.catalog.get(provider)
        if loaded_catalog is not None and loaded_catalog.version == version:
            logger.debug("The %s catalog %s is up-to-date", provider, version)
            return None
        logger.debug("Downloading the %s catalog %s...", provider, version)
        with urllib.request.urlopen(f"{base_url}/{provider}/{version}/catalog.zip") as f:
            data = f.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zip_file:
            with zip_file.open(f"{provider}.csv", "r") as csv_file:
                items = list(storage.load(io.TextIOWrapper(csv_file, "utf-8"), provider=provider))
        return ProviderCatalog(version=version, items=items)

    def add_provider(self, provider: AbstractProvider):
        """
        Add provider for querying offers

        Args:
            provider: provider to add
        """
        self.providers.append(provider)

    def _get_offline_provider_items(
        self, provider_name: str, query_filter: QueryFilter
    ) -> list[CatalogItem]:
        logger.debug("Loading items for offline provider %s", provider_name)
        items = []
        # Set this env var to use a local catalog instead of the s3 catalog
        catalog_dir = os.getenv("GPUHUNT_CATALOG_DIR")
        if catalog_dir is not None:
            with open(Path(catalog_dir) / f"{provider_name}.csv", "rb") as csv_file:
                for item in storage.load(
                    io.TextIOWrapper(csv_file, "utf-8"), provider=provider_name
                ):
                    if constraints.matches(item, query_filter):
                        items.append(item)
            return items

        if self.loaded_at is None:
            logger.error("Catalog not loaded. Returning zero items.")
            return []
        provider_catalog = self.catalog.get(provider_name)
        if provider_catalog is None:
            logger.error(f"No catalog for offline provider {provider_name}. Returning zero items.")
            return []
        for item in provider_catalog.items:
            if constraints.matches(item, query_filter):
                items.append(item)
        return items

    def _get_online_provider_items(
        self, provider_name: str, query_filter: QueryFilter
    ) -> list[CatalogItem]:
        logger.debug("Loading items for online provider %s", provider_name)
        items = []
        found = False
        for provider in self.providers:
            if provider.NAME != provider_name:
                continue
            found = True
            for item in provider.get(
                query_filter=query_filter, balance_resources=self.balance_resources
            ):
                if constraints.matches(item, query_filter):
                    items.append(item)
        if not found:
            raise ValueError(f"Provider is not loaded: {provider_name}")
        return items


def _get_base_url() -> str:
    # Set this env var to use catalogs published somewhere other than the S3 bucket
    return os.getenv("GPUHUNT_CATALOG_URL", CATALOG_URL).rstrip("/")


def _get_latest_version(base_url: str, provider: str) -> str:
    with urllib.request.urlopen(f"{base_url}/{provider}/version") as f:
        return f.read().decode("utf-8").strip()
