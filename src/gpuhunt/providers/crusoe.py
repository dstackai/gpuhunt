import base64
import copy
import datetime
import hashlib
import hmac
import logging
import os
from collections import defaultdict
from typing import Optional

import requests

from gpuhunt._internal.models import AcceleratorVendor, QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

API_URL = "https://api.crusoecloud.com"
API_VERSION = "/v1alpha5"
SIGNATURE_VERSION = "1.0"
TIMEOUT = 30

GPU_TYPE_MAP: dict[str, tuple[str, AcceleratorVendor, float]] = {
    # gpu_type -> (gpuhunt_name, vendor, vram_gb)
    "A100-PCIe-40GB": ("A100", AcceleratorVendor.NVIDIA, 40),
    "A100-PCIe-80GB": ("A100", AcceleratorVendor.NVIDIA, 80),
    "A100-SXM-80GB": ("A100", AcceleratorVendor.NVIDIA, 80),
    "H100-SXM-80GB": ("H100", AcceleratorVendor.NVIDIA, 80),
    "L40S-48GB": ("L40S", AcceleratorVendor.NVIDIA, 48),
    "A40-PCIe-48GB": ("A40", AcceleratorVendor.NVIDIA, 48),
    "MI300X-192GB": ("MI300X", AcceleratorVendor.AMD, 192),
    # TODO: The following GPUs are listed on https://crusoe.ai/cloud/pricing but not yet
    # returned by the instance types API. Add them once Crusoe exposes them:
    #   - H200 141GB ($4.29/GPU-hr on-demand, spot: contact sales)
    #   - GB200 186GB (contact sales)
    #   - B200 180GB (contact sales)
    #   - MI355X 288GB ($3.45 listed but not confirmed; also missing from KNOWN_AMD_GPUS)
}

# Per-GPU-hour pricing from https://crusoe.ai/cloud/pricing
GPU_PRICING: dict[str, tuple[float, Optional[float]]] = {
    # gpu_type -> (on_demand_per_gpu_hr, spot_per_gpu_hr or None)
    "A100-PCIe-40GB": (1.45, 1.00),
    "A100-PCIe-80GB": (1.65, 1.20),
    "A100-SXM-80GB": (1.95, 1.30),
    "H100-SXM-80GB": (3.90, 1.60),
    "L40S-48GB": (1.00, 0.50),
    "A40-PCIe-48GB": (0.90, 0.40),
    "MI300X-192GB": (3.45, 0.95),
}

# Per-vCPU-hour pricing from https://crusoe.ai/cloud/pricing
CPU_PRICING: dict[str, float] = {
    # product_name prefix -> per_vcpu_hr
    "c1a": 0.04,
    "s1a": 0.09,
}


class CrusoeProvider(AbstractProvider):
    NAME = "crusoe"

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        self.access_key = access_key or os.getenv("CRUSOE_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("CRUSOE_SECRET_KEY")
        self.project_id = project_id or os.getenv("CRUSOE_PROJECT_ID")

        if not self.access_key:
            raise ValueError("Set the CRUSOE_ACCESS_KEY environment variable.")
        if not self.secret_key:
            raise ValueError("Set the CRUSOE_SECRET_KEY environment variable.")
        if not self.project_id:
            raise ValueError("Set the CRUSOE_PROJECT_ID environment variable.")

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        instance_types = self._get_instance_types()
        type_specs = {t["product_name"]: t for t in instance_types}

        # Note: capacities reflect hardware availability, not project quotas.
        # Quota enforcement should be done on the dstack side via
        # GET /projects/{project_id}/quotas, which returns per-instance-family
        # quota with max/used/available fields.
        capacities = self._get_capacities()
        available = _get_available_type_locations(capacities)

        offers = []
        for product_name, locations in available.items():
            spec = type_specs.get(product_name)
            if spec is None:
                logger.warning("Capacity for unknown instance type %s, skipping", product_name)
                continue

            items = _make_catalog_items(product_name, spec, locations)
            offers.extend(items)

        return sorted(offers, key=lambda i: i.price)

    def _get_instance_types(self) -> list[dict]:
        resp = self._request("GET", f"/projects/{self.project_id}/compute/vms/types")
        resp.raise_for_status()
        return resp.json()["items"]

    def _get_capacities(self) -> list[dict]:
        resp = self._request("GET", "/capacities")
        resp.raise_for_status()
        return resp.json()["items"]

    def _request(self, method: str, path: str, params: Optional[dict] = None) -> requests.Response:
        dt = str(datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0))
        dt = dt.replace(" ", "T")

        query_string = ""
        if params:
            query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))

        payload = f"{API_VERSION}{path}\n{query_string}\n{method}\n{dt}\n"

        decoded_secret = base64.urlsafe_b64decode(
            self.secret_key + "=" * (-len(self.secret_key) % 4)
        )
        sig = hmac.new(decoded_secret, msg=payload.encode("ascii"), digestmod=hashlib.sha256)
        encoded_sig = base64.urlsafe_b64encode(sig.digest()).decode("ascii").rstrip("=")

        headers = {
            "X-Crusoe-Timestamp": dt,
            "Authorization": f"Bearer {SIGNATURE_VERSION}:{self.access_key}:{encoded_sig}",
        }

        url = f"{API_URL}{API_VERSION}{path}"
        return requests.request(method, url, headers=headers, params=params, timeout=TIMEOUT)


def _get_available_type_locations(capacities: list[dict]) -> dict[str, list[str]]:
    best_qty: dict[tuple[str, str], int] = defaultdict(int)
    for cap in capacities:
        key = (cap["type"], cap["location"])
        best_qty[key] = max(best_qty[key], cap["quantity"])

    result: dict[str, list[str]] = defaultdict(list)
    for (instance_type, location), qty in best_qty.items():
        if qty > 0:
            result[instance_type].append(location)
    return dict(result)


def _make_catalog_items(
    product_name: str, spec: dict, locations: list[str]
) -> list[RawCatalogItem]:
    gpu_type = spec.get("gpu_type", "")
    num_gpu = spec.get("num_gpu", 0)

    if num_gpu > 0 and gpu_type:
        return _make_gpu_items(product_name, spec, gpu_type, locations)
    else:
        return _make_cpu_items(product_name, spec, locations)


def _make_gpu_items(
    product_name: str, spec: dict, gpu_type: str, locations: list[str]
) -> list[RawCatalogItem]:
    gpu_info = GPU_TYPE_MAP.get(gpu_type)
    if gpu_info is None:
        logger.warning("Unknown GPU type %s for %s, skipping", gpu_type, product_name)
        return []

    pricing = GPU_PRICING.get(gpu_type)
    if pricing is None:
        logger.warning("No pricing for GPU type %s (%s), skipping", gpu_type, product_name)
        return []

    gpu_name, gpu_vendor, gpu_memory = gpu_info
    on_demand_per_gpu, spot_per_gpu = pricing
    num_gpu = spec["num_gpu"]

    template = RawCatalogItem(
        instance_name=product_name,
        location=None,
        price=None,
        cpu=spec["cpu_cores"],
        memory=float(spec["memory_gb"]),
        gpu_vendor=gpu_vendor.value,
        gpu_count=num_gpu,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
        spot=None,
        disk_size=float(spec["disk_gb"]) if spec.get("disk_gb") else None,
        # disk_gb: ephemeral NVMe size in GB (0 = no ephemeral disk).
        # Used by dstack to decide whether to create a persistent data disk.
        provider_data={"disk_gb": spec.get("disk_gb", 0)},
    )

    items = []
    for location in locations:
        on_demand = copy.deepcopy(template)
        on_demand.location = location
        on_demand.spot = False
        on_demand.price = round(num_gpu * on_demand_per_gpu, 2)
        items.append(on_demand)

        # TODO: Enable spot offers once we confirm how to request spot billing
        # via the VM create API (POST /v1alpha5/projects/{pid}/compute/vms/instances).
        # The API schema doesn't have an obvious spot/billing_type field.

    return items


def _make_cpu_items(product_name: str, spec: dict, locations: list[str]) -> list[RawCatalogItem]:
    prefix = product_name.split(".")[0]
    per_vcpu = CPU_PRICING.get(prefix)
    if per_vcpu is None:
        logger.warning("No pricing for CPU prefix %s (%s), skipping", prefix, product_name)
        return []

    cpu_cores = spec["cpu_cores"]
    template = RawCatalogItem(
        instance_name=product_name,
        location=None,
        price=None,
        cpu=cpu_cores,
        memory=float(spec["memory_gb"]),
        gpu_vendor=None,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=None,
        spot=False,
        disk_size=float(spec["disk_gb"]) if spec.get("disk_gb") else None,
        provider_data={"disk_gb": spec.get("disk_gb", 0)},
    )

    items = []
    for location in locations:
        item = copy.deepcopy(template)
        item.location = location
        item.price = round(cpu_cores * per_vcpu, 2)
        items.append(item)

    return items
