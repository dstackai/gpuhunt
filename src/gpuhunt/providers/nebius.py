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


class NebiusProvider(AbstractProvider):
    def __init__(self, service_account: "ServiceAccount"):
        self.api_client = NebiusAPIClient(service_account)

    def get(self, query_filter: Optional[QueryFilter] = None) -> List[RawCatalogItem]:
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
        return self.get_gpu_platforms(zone, platform_resources) + self.get_cpu_platforms(
            zone, platform_resources
        )

    def get_gpu_platforms(
        self, zone: str, platform_resources: "PlatformResourcePrice"
    ) -> List[RawCatalogItem]:
        logger.info("Fetching GPU platforms")
        resp = requests.get("https://nebius.ai/docs/compute/concepts/gpus")
        resp.raise_for_status()
        soup = bs4.BeautifulSoup(resp.text, "html.parser")
        configs = soup.find("h2", id="config").find_next_sibling("ul").find_all("li")
        items = []
        for li in configs:
            platform = li.find("p").find("code").text
            prices = platform_resources[platform]
            gpu_name = re.search(r" ([A-Z]+\d+) ", li.find("p").text).group(1)
            for tr in li.find("tbody").find_all("tr"):
                tds = tr.find_all("td")
                gpu_count = int(tds[0].text.strip(" *"))
                cpu = int(tds[2].text)
                memory = float(tds[3].text)
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
                        cpu=int(tds[2].text),
                        memory=float(tds[3].text),
                        gpu_count=gpu_count,
                        gpu_name=gpu_name,
                        gpu_memory=int(tds[1].text) / gpu_count,
                        spot=False,
                    )
                )
        return items

    def get_cpu_platforms(
        self, zone: str, platform_resources: "PlatformResourcePrice"
    ) -> List[RawCatalogItem]:
        logger.info("Fetching CPU platforms")
        resp = requests.get("https://nebius.ai/docs/compute/concepts/performance-levels")
        resp.raise_for_status()
        soup = bs4.BeautifulSoup(resp.text, "html.parser")
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
        items = []
        for li in configs:
            platform = li.find("p").find("code").text
            prices = platform_resources[platform]
            tds = li.find("tbody").find("td", string="100%").find_next_siblings("td")
            cpus = [int(i) for i in tds[0].text.translate({"\n": "", " ": ""}).split(",")]
            ratios = [float(i) for i in tds[1].text.translate({"\n": "", " ": ""}).split(",")]
            for ratio in ratios:
                for cpu in cpus:
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
                        )
                    )
        return items

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
            platform_resources[GPU_NAME_PLATFORM[gpu_name]][
                vm_resources[resource_name]
            ] = self.get_sku_price(sku["pricingVersions"])

        return platform_resources

    def get_sku_price(self, pricing_versions: List[dict]) -> Optional[float]:
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
