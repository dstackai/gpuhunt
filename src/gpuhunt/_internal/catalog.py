import csv
import dataclasses
import heapq
import io
import logging
import os
import time
import urllib.request
import zipfile
from collections.abc import Container
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional, Union

import gpuhunt._internal.constraints as constraints
from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, CPUArchitecture, QueryFilter
from gpuhunt._internal.utils import parse_compute_capability
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

VERSION_URL = "https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v2/version"
CATALOG_URL = "https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v2/{version}/catalog.zip"
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
ONLINE_PROVIDERS = ["cudo", "digitalocean", "hotaisle", "tensordock", "vastai", "vultr"]
RELOAD_INTERVAL = 15 * 60  # 15 minutes


class Catalog:
    def __init__(self, balance_resources: bool = True, auto_reload: bool = True):
        """
        Args:
            balance_resources: increase min resources to better match the chosen GPU
            auto_reload: if `True`, the catalog will be automatically loaded from the S3 bucket every 4 hours
        """
        self.catalog = None
        self.loaded_at = None
        self.providers: list[AbstractProvider] = []
        self.balance_resources = balance_resources
        self.auto_reload = auto_reload

    def query(
        self,
        *,
        provider: Optional[Union[str, list[str]]] = None,
        cpu_arch: Optional[Union[CPUArchitecture, str]] = None,
        min_cpu: Optional[int] = None,
        max_cpu: Optional[int] = None,
        min_memory: Optional[float] = None,
        max_memory: Optional[float] = None,
        min_gpu_count: Optional[int] = None,
        max_gpu_count: Optional[int] = None,
        gpu_vendor: Optional[Union[AcceleratorVendor, str]] = None,
        gpu_name: Optional[Union[str, list[str]]] = None,
        min_gpu_memory: Optional[float] = None,
        max_gpu_memory: Optional[float] = None,
        min_total_gpu_memory: Optional[float] = None,
        max_total_gpu_memory: Optional[float] = None,
        min_disk_size: Optional[int] = None,
        max_disk_size: Optional[int] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_compute_capability: Optional[Union[str, tuple[int, int]]] = None,
        max_compute_capability: Optional[Union[str, tuple[int, int]]] = None,
        spot: Optional[bool] = None,
        allowed_flags: Optional[Container[str]] = None,
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
            self.load()

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

    def load(self, version: Optional[str] = None):
        """
        Fetch the catalog from the S3 bucket

        Args:
            version: specific version of the catalog to download. If not specified, the latest version will be used
        """
        catalog_url = os.getenv("GPUHUNT_CATALOG_URL")
        if catalog_url is None:
            if version is None:
                version = self.get_latest_version()
            catalog_url = CATALOG_URL.format(version=version)
        logger.debug("Downloading catalog %s...", version)
        with urllib.request.urlopen(catalog_url) as f:
            self.loaded_at = time.monotonic()
            self.catalog = io.BytesIO(f.read())

    @staticmethod
    def get_latest_version() -> str:
        """
        Get the latest version of the catalog from the S3 bucket
        """
        with urllib.request.urlopen(VERSION_URL) as f:
            return f.read().decode("utf-8").strip()

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
                reader = csv.DictReader(io.TextIOWrapper(csv_file, "utf-8"))
                for row in reader:
                    item = CatalogItem.from_dict(row, provider=provider_name)
                    if constraints.matches(item, query_filter):
                        items.append(item)
        else:
            if self.catalog is None:
                logger.warning("Catalog not loaded")
                return items
            with zipfile.ZipFile(self.catalog) as zip_file:
                with zip_file.open(f"{provider_name}.csv", "r") as csv_file:
                    reader = csv.DictReader(io.TextIOWrapper(csv_file, "utf-8"))
                    for row in reader:
                        item = CatalogItem.from_dict(row, provider=provider_name)
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
            for i in provider.get(
                query_filter=query_filter, balance_resources=self.balance_resources
            ):
                item = CatalogItem(provider=provider_name, **dataclasses.asdict(i))
                if constraints.matches(item, query_filter):
                    items.append(item)
        if not found:
            raise ValueError(f"Provider is not loaded: {provider_name}")
        return items
