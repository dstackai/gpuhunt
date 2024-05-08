import datetime
import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Literal, Optional, TypedDict

import bs4
import jwt
import requests

from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
API_URL = "api.ai.nebius.cloud"
COMPUTE_SERVICE_ID = "bfa2pas77ftg9h3f2djj"
GPU_NAME_PLATFORM = {
    "A100": "gpu-standard-v3",
    "H100 PCIe": "standard-v3-h100-pcie",
    "Hopper H100 SXM (Type A)": "gpu-h100",
    "Hopper H100 SXM (Type B)": "gpu-h100-b",
    "L4": "standard-v3-l4",
    "L40": "standard-v3-l40",
    None: "standard-v2",
}
GPU_PLATFORMS = {
    "gpu-standard-v3": [
        [28, 119.0, 1, "A100", 80.0],
        [56, 238.0, 2, "A100", 80.0],
        [112, 476.0, 4, "A100", 80.0],
        [224, 952.0, 8, "A100", 80.0],
    ],
    "gpu-h100": [
        [20, 160.0, 1, "H100", 80.0],
        [40, 320.0, 2, "H100", 80.0],
        [80, 640.0, 4, "H100", 80.0],
        [160, 1280.0, 8, "H100", 80.0],
    ],
    "gpu-h100-b": [
        [20, 160.0, 1, "H100", 80.0],
        [40, 320.0, 2, "H100", 80.0],
        [80, 640.0, 4, "H100", 80.0],
        [160, 1280.0, 8, "H100", 80.0],
    ],
    "standard-v3-h100-pcie": [
        [24, 96.0, 1, "H100", 80.0],
        [48, 192.0, 2, "H100", 80.0],
    ],
    "standard-v3-l4": [
        [4, 16.0, 1, "L4", 24.0],
        [8, 32.0, 1, "L4", 24.0],
        [12, 48.0, 1, "L4", 24.0],
        [16, 64.0, 1, "L4", 24.0],
        [24, 96.0, 1, "L4", 24.0],
        [24, 96.0, 2, "L4", 24.0],
        [48, 192.0, 2, "L4", 24.0],
    ],
    "standard-v3-l40": [
        [8, 32.0, 1, "L40", 48.0],
        [12, 48.0, 1, "L40", 48.0],
        [16, 64.0, 1, "L40", 48.0],
        [24, 96.0, 1, "L40", 48.0],
        [48, 192.0, 2, "L40", 48.0],
    ],
}
CPU_PLATFORMS = {
    "standard-v2": {
        "cpus": [
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            44,
            48,
            52,
            56,
            60,
        ],
        "ratios": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
}


class NebiusProvider(AbstractProvider):
    NAME = "nebius"

    def __init__(self, service_account: "ServiceAccount"):
        self.api_client = NebiusAPIClient(service_account)

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        zone = self.api_client.compute_zones_list()[0]["id"]
        skus = []
        page_token = None
        logger.info("Fetching SKUs")
        while True:
            page = self.api_client.billing_skus_list(
                filter=f'serviceId="{COMPUTE_SERVICE_ID}"', page_token=page_token
            )
            skus += page["skus"]
            page_token = page.get("nextPageToken")
            if page_token is None:
                break
        platform_resources = self.aggregate_skus(skus)
        offers = self.get_gpu_platforms(zone, platform_resources)
        offers += self.get_cpu_platforms(zone, platform_resources)
        return sorted(offers, key=lambda i: i.price)

    @staticmethod
    def get_gpu_platforms(
        zone: str, platform_resources: "PlatformResourcePrice"
    ) -> List[RawCatalogItem]:
        items = []
        for platform, presets in GPU_PLATFORMS.items():
            prices = platform_resources[platform]
            for cpu, memory, gpu_count, gpu_name, gpu_memory in presets:
                if "cpu" not in prices:
                    continue
                items.append(
                    RawCatalogItem(
                        instance_name=platform,
                        location=zone,
                        price=round(
                            cpu * prices["cpu"]
                            + memory * prices["ram"]
                            + gpu_count * prices["gpu"],
                            5,
                        ),
                        cpu=cpu,
                        memory=memory,
                        gpu_count=gpu_count,
                        gpu_name=gpu_name,
                        gpu_memory=gpu_memory,
                        spot=False,
                        disk_size=None,
                    )
                )
        return items

    @staticmethod
    def get_cpu_platforms(
        zone: str, platform_resources: "PlatformResourcePrice"
    ) -> List[RawCatalogItem]:
        items = []
        for platform, limits in CPU_PLATFORMS.items():
            prices = platform_resources[platform]
            if "cpu" not in prices:
                continue
            for ratio in limits["ratios"]:
                for cpu in limits["cpus"]:
                    items.append(
                        RawCatalogItem(
                            instance_name=platform,
                            location=zone,
                            price=round(cpu * prices["cpu"] + cpu * ratio * prices["ram"], 5),
                            cpu=cpu,
                            memory=cpu * ratio,
                            gpu_count=0,
                            gpu_name=None,
                            gpu_memory=None,
                            spot=False,
                            disk_size=None,
                        )
                    )
        return items

    @staticmethod
    def parse_gpu_platforms(raw_html: str) -> Dict[str, List[List]]:
        """Parse GPU platforms from Nebius docs.

        Returns:
            Dict of platform name to a list of presets.
            Each preset contains: [cpu, memory, gpu_count, gpu_name, gpu_memory]
        """
        soup = bs4.BeautifulSoup(raw_html, "html.parser")
        configs = soup.find("h2", id="config").find_next_sibling("ul").find_all("li")
        platforms = {}
        for li in configs:
            platform = li.find("p").find("code").text
            gpu_name = re.search(r" ([A-Z]+\d+)[ \n]", li.find("p").text).group(1)
            items = []
            for tr in li.find("tbody").find_all("tr"):
                tds = tr.find_all("td")
                gpu_count = int(tds[0].text.strip(" *"))
                items.append(
                    [
                        int(tds[2].text),
                        float(tds[3].text),
                        gpu_count,
                        gpu_name,
                        int(tds[1].text) / gpu_count,
                    ]
                )
            platforms[platform] = items
        return platforms

    @staticmethod
    def parse_cpu_platforms(raw_html: str) -> Dict[str, Dict[str, List]]:
        """Parse CPU platforms from Nebius docs.

        Returns:
            Dict of platform name to Dict of resource name to a list of values.
        """
        soup = bs4.BeautifulSoup(raw_html, "html.parser")
        configs = (
            soup.find(
                "p",
                string=re.compile(
                    r"The computing resources may have the following configurations:"
                ),
            )
            .find_next_sibling("ul")
            .find_all("li")
        )
        platforms = {}
        for li in configs:
            platform = li.find("p").find("code").text
            tds = li.find("tbody").find("td", string="100%").find_next_siblings("td")
            platforms[platform] = {
                "cpus": [int(i) for i in tds[0].text.translate({"\n": "", " ": ""}).split(",")],
                "ratios": [
                    float(i) for i in tds[1].text.translate({"\n": "", " ": ""}).split(",")
                ],
            }
        return platforms

    def aggregate_skus(self, skus: List[dict]) -> "PlatformResourcePrice":
        vm_resources = {
            "GPU": "gpu",
            "RAM": "ram",
            "100% vCPU": "cpu",
        }
        vm_name_re = re.compile(
            r"((?:Intel|AMD) .+?)(?: with Nvidia (.+))?"
            rf"\. ({'|'.join(vm_resources)})(?: — (preemptible).*)?$"
        )
        platform_resources = defaultdict(dict)
        for sku in skus:
            if (r := vm_name_re.match(sku["name"])) is None:
                continue  # storage, images, snapshots, infiniband
            cpu_name, gpu_name, resource_name, spot = r.groups()
            if spot is not None:
                continue
            if gpu_name not in GPU_NAME_PLATFORM:
                logger.warning("Unknown GPU name: %s", gpu_name)
                continue
            platform_resources[GPU_NAME_PLATFORM[gpu_name]][vm_resources[resource_name]] = (
                self.get_sku_price(sku["pricingVersions"])
            )

        return platform_resources

    @staticmethod
    def get_sku_price(pricing_versions: List[dict]) -> Optional[float]:
        now = datetime.datetime.now(datetime.timezone.utc)
        price = None
        for version in sorted(pricing_versions, key=lambda p: p["effectiveTime"]):
            # I guess it's the price for on-demand instances
            if version["type"] != "STREET_PRICE":
                continue
            if datetime.datetime.fromisoformat(version["effectiveTime"]) > now:
                break
            # I guess we should take the first pricing expression
            price = float(version["pricingExpressions"][0]["rates"][0]["unitPrice"])
        return price


class NebiusAPIClient:
    # reference: https://nebius.ai/docs/api-design-guide/
    def __init__(self, service_account: "ServiceAccount"):
        self._service_account = service_account
        self._s = requests.Session()
        self._expires_at = 0

    def get_token(self):
        now = int(time.time())
        if now + 60 < self._expires_at:
            return
        logger.debug("Refreshing IAM token")
        expires_at = now + 3600
        payload = {
            "aud": self.url("iam", "/tokens"),
            "iss": self._service_account["service_account_id"],
            "iat": now,
            "exp": expires_at,
        }
        jwt_token = jwt.encode(
            payload,
            self._service_account["private_key"],
            algorithm="PS256",
            headers={"kid": self._service_account["id"]},
        )

        resp = requests.post(payload["aud"], json={"jwt": jwt_token})
        resp.raise_for_status()
        iam_token = resp.json()["iamToken"]
        self._s.headers["Authorization"] = f"Bearer {iam_token}"
        self._expires_at = expires_at

    def billing_skus_list(
        self,
        filter: Optional[str] = None,
        page_size: Optional[int] = 1000,
        page_token: Optional[str] = None,
    ) -> "BillingSkusListResponse":
        logger.debug("Fetching SKUs")
        params = {
            "currency": "USD",
            "pageSize": page_size,
        }
        if filter is not None:
            params["filter"] = filter
        if page_token is not None:
            params["pageToken"] = page_token
        self.get_token()
        resp = self._s.get(self.url("billing", "/skus"), params=params)
        resp.raise_for_status()
        return resp.json()

    def compute_zones_list(self) -> List[dict]:
        logger.debug("Fetching compute zones")
        self.get_token()
        resp = self._s.get(self.url("compute", "/zones"))
        resp.raise_for_status()
        return resp.json()["zones"]

    def url(self, service: str, path: str, version="v1") -> str:
        return f"https://{service}.{API_URL.rstrip('/')}/{service}/{version}/{path.lstrip('/')}"


class ServiceAccount(TypedDict):
    id: str
    service_account_id: str
    created_at: str
    key_algorithm: str
    public_key: str
    private_key: str


class BillingSkusListResponse(TypedDict):
    skus: List[dict]
    nextPageToken: Optional[str]


PlatformResourcePrice = Dict[str, Dict[Literal["cpu", "ram", "gpu"], float]]
