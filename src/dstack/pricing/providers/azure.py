import os
import json

from dstack.pricing.models import InstanceOffer
from dstack.pricing.providers import AbstractProvider


api_url = "https://prices.azure.com/api/retail/prices"
api_version = "2023-01-01-preview"


class AzureProvider(AbstractProvider):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def get_page(self, page: int) -> dict:
        with open(os.path.join(self.cache_dir, f"{page:04}.json")) as f:
            return json.load(f)

    def get(self) -> list[InstanceOffer]:
        page = 0
        offers = []
        while True:
            data = self.get_page(page)
            for item in data["Items"]:
                offer = InstanceOffer(
                    instance_name=item["armSkuName"],
                    location=item["armRegionName"],
                    price=item["retailPrice"],
                    spot="Spot" in item["meterName"],
                )
                offers.append(offer)
            if not data["NextPageLink"]:
                break
            page += 1
        return offers
