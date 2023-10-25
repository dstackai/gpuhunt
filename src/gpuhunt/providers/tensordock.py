import logging
from typing import Optional, List

import requests

from gpuhunt._internal.models import RawCatalogItem, QueryFilter
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

# https://documenter.getpostman.com/view/20973002/2s8YzMYRDc#2b4a3db0-c162-438c-aae4-6a88afc96fdb
marketplace_hostnodes_url = (
    "https://marketplace.tensordock.com/api/v0/client/deploy/hostnodes"
)
marketplace_gpus = {
    "a100-pcie-80gb": "A100",
    "geforcegtx1070-pcie-8gb": "GTX1070",
    "geforcertx3060-pcie-12gb": "RTX3060",
    "geforcertx3060ti-pcie-8gb": "RTX3060Ti",
    "geforcertx3060tilhr-pcie-8gb": "RTX3060TiLHR",
    "geforcertx3070-pcie-8gb": "RTX3070",
    "geforcertx3070ti-pcie-8gb": "RTX3070Ti",
    "geforcertx3080-pcie-10gb": "RTX3080",
    "geforcertx3080ti-pcie-12gb": "RTX3080Ti",
    "geforcertx3090-pcie-24gb": "RTX3090",
    "geforcertx4090-pcie-24gb": "RTX4090",
    "l40-pcie-48gb": "L40",
    "rtxa4000-pcie-16gb": "A4000",
    "rtxa5000-pcie-24gb": "A5000",
    "rtxa6000-pcie-48gb": "A6000",
    "v100-pcie-16gb": "V100",
}


class TensorDockProvider(AbstractProvider):
    NAME = "tensordock"

    def get(self, query_filter: Optional[QueryFilter] = None) -> List[RawCatalogItem]:
        logger.info("Fetching TensorDock offers")
        hostnodes = requests.get(marketplace_hostnodes_url).json()["hostnodes"]
        offers = []
        for hostnode, details in hostnodes.items():
            location = (
                "-".join(
                    [details["location"][key] for key in ["country", "region", "city"]]
                )
                .lower()
                .replace(" ", "")
            )
            cpu = details["specs"]["cpu"]["amount"]
            memory = details["specs"]["ram"]["amount"]
            base_price = (
                cpu * details["specs"]["cpu"]["price"]
                + memory * details["specs"]["ram"]["price"]
            )
            for gpu_name, gpu in details["specs"]["gpu"].items():
                gpu_count = gpu["amount"]
                if gpu_count == 0:
                    continue
                gpu_name = marketplace_gpus.get(gpu_name, gpu_name)
                # TODO use minimal required resources
                offer = RawCatalogItem(
                    instance_name=f"{gpu_name.lower()}_{gpu_count}_{hostnode}",
                    location=location,
                    price=round(gpu_count * gpu["price"] + base_price, 5),
                    cpu=cpu,
                    memory=memory,
                    gpu_count=gpu_count,
                    gpu_name=gpu_name,
                    gpu_memory=gpu["vram"],
                    spot=False,
                )
                offers.append(offer)
        return offers
