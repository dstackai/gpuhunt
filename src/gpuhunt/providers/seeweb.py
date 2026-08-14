import logging
import re

import requests

from gpuhunt._internal.constraints import find_accelerators
from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, QueryFilter
from gpuhunt.providers.base import OnlineProvider, get_creds_env

logger = logging.getLogger(__name__)

API_URL = "https://api.seeweb.it/ecs/v2"
TIMEOUT = 30

# Maps a Seeweb plan's ``gpu_label`` (the ``gpu_label`` field of the ``/plans`` response) to a
# canonical GPU name and per-card memory in GiB. Keys are the exact ``gpu_label`` strings returned
# by the live ``/plans`` endpoint. Any GPU label not found here (and not resolvable by
# ``find_accelerators``) is skipped with a warning, which makes missing mappings easy to spot.
SEEWEB_GPU_MAP: dict[str, tuple[str, float]] = {
    "a30": ("A30", 24),
    "l4 24gb": ("L4", 24),
    "l40s 48gb": ("L40S", 48),
    "a100 80gb": ("A100", 80),
    "h100 80gb": ("H100", 80),
    "h200 141gb": ("H200", 141),
    "rtx a6000 48gb": ("A6000", 48),
    "rtx 6000 24gb": ("RTX6000", 24),
    "rtx pro 6000 96gb": ("RTXPRO6000", 96),
}

# Non-NVIDIA accelerators Seeweb offers that the NVIDIA-only MVP intentionally skips (quietly).
_NON_NVIDIA_MARKERS = ("MI300", "TENSTORRENT", "GRAYSKULL", "WORMHOLE")


class SeewebProvider(OnlineProvider):
    """Online provider for Seeweb Cloud Server GPU.

    Seeweb's plan/pricing endpoints require authentication, so this is an online provider queried
    live with a token. The dstack Seeweb backend builds a private ``gpuhunt.Catalog`` with this
    provider and the project's token (like the vastai/jarvislabs backends), so offers work without
    a published catalog. ``python -m gpuhunt seeweb`` can still dump a catalog for inspection.
    """

    NAME = "seeweb"

    def __init__(self, token: str):
        self.token = token

    @classmethod
    def from_env(cls) -> "SeewebProvider":
        return cls(token=get_creds_env("SEEWEB_API_TOKEN"))

    def get(
        self, query_filter: QueryFilter | None = None, balance_resources: bool = True
    ) -> list[CatalogItem]:
        response = requests.get(
            f"{API_URL}/plans",
            headers={"X-APITOKEN": self.token},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("plans"), list):
            raise ValueError("Unexpected response from Seeweb /plans endpoint")

        offers: list[CatalogItem] = []
        for plan in data["plans"]:
            if not isinstance(plan, dict):
                logger.warning("Skipping malformed Seeweb plan: %r", plan)
                continue
            offers.extend(_make_offers(plan))
        return sorted(offers, key=lambda item: item.price if item.price is not None else 0.0)


def _make_offers(plan: dict) -> list[CatalogItem]:
    plan_name = plan.get("name")
    gpu_label = plan.get("gpu_label")
    # In the /plans response, "available" means that the plan is active. It does not indicate
    # current capacity in any particular region.
    if plan.get("available") is False:
        return []
    if not plan_name or not gpu_label:
        # This includes CPU-only plans, which are outside the NVIDIA GPU provider scope.
        return []
    try:
        gpu_count = _parse_int(plan.get("gpu"))
    except (TypeError, ValueError):
        logger.warning("Skipping Seeweb plan %s: invalid GPU count %r", plan_name, plan.get("gpu"))
        return []
    if gpu_count <= 0:
        return []
    if any(marker in str(gpu_label).upper() for marker in _NON_NVIDIA_MARKERS):
        logger.debug("Skipping non-NVIDIA Seeweb plan %s (%s)", plan_name, gpu_label)
        return []

    resolved = _normalize_gpu(str(gpu_label))
    if resolved is None:
        logger.warning("Skipping Seeweb plan %s: unknown gpu_label %r", plan_name, gpu_label)
        return []
    gpu_name, gpu_memory = resolved

    try:
        cpu = _parse_int(plan.get("cpu"))
        memory = _parse_float(plan.get("ram")) / 1024
        disk_size = _parse_float(plan.get("disk"))
        price = _parse_float(plan.get("hourly_price"))
    except (TypeError, ValueError) as e:
        logger.warning("Skipping Seeweb plan %s: %s", plan_name, e)
        return []
    if cpu <= 0 or memory <= 0 or disk_size <= 0 or price < 0:
        logger.warning(
            "Skipping Seeweb plan %s: invalid non-positive resources or price", plan_name
        )
        return []

    offers: list[CatalogItem] = []
    seen_regions = set()
    # These are regions where the plan is active/compatible, not a real-time capacity signal.
    # Consumers that need current capacity should query /plans/availables separately.
    regions = plan.get("available_regions")
    if not isinstance(regions, list):
        logger.warning("Skipping Seeweb plan %s: invalid available_regions", plan_name)
        return []
    for region in regions:
        if not isinstance(region, dict) or not isinstance(region.get("location"), str):
            logger.warning("Skipping malformed region for Seeweb plan %s: %r", plan_name, region)
            continue
        location = region["location"].strip()
        if not location or location in seen_regions:
            continue
        seen_regions.add(location)
        offers.append(
            CatalogItem(
                provider=SeewebProvider.NAME,
                instance_name=plan_name,
                location=location,
                price=price,
                cpu=cpu,
                # Seeweb returns RAM in MiB; gpuhunt represents memory in GiB.
                memory=memory,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                gpu_memory=gpu_memory,
                gpu_vendor=AcceleratorVendor.NVIDIA,
                spot=False,
                disk_size=disk_size,
            )
        )
    return offers


def _normalize_gpu(gpu_label: str) -> tuple[str, float] | None:
    """Resolve a Seeweb ``gpu_label`` to a ``(canonical_name, memory_gib)`` pair."""
    label = " ".join(gpu_label.strip().split())
    # 1. Exact match against the curated map (with and without the NVIDIA prefix).
    for key in (
        label.casefold(),
        re.sub(r"^nvidia\s+", "", label, flags=re.IGNORECASE).casefold(),
    ):
        if key in SEEWEB_GPU_MAP:
            return SEEWEB_GPU_MAP[key]
    # 2. Fall back to gpuhunt's known-accelerator table.
    cleaned = re.sub(r"^NVIDIA\s+", "", label.upper()).strip()
    explicit_memory = re.search(r"(\d+)\s*GB", cleaned)
    candidate = re.sub(r"\d+\s*GB", "", cleaned).strip().replace(" ", "")
    if candidate:
        accelerators = find_accelerators(names=[candidate], vendors=[AcceleratorVendor.NVIDIA])
        if accelerators:
            info = accelerators[0]
            memory = float(explicit_memory.group(1)) if explicit_memory else float(info.memory)
            return info.name, memory
    return None


def _parse_int(value: object) -> int:
    match = re.search(r"\d+", str(value))
    if not match:
        raise ValueError(f"invalid integer value {value!r}")
    return int(match.group())


def _parse_float(value: object) -> float:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        raise ValueError(f"invalid numeric value {value!r}")
    return float(match.group())
