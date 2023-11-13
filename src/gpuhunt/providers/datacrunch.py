from typing import List, Optional

from datacrunch import DataCrunchClient

from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider


class DataCrunchProvider(AbstractProvider):
    NAME = "datacrunch"

    def __init__(self, client_id: str, client_secret: str) -> None:
        self.datacrunch_client = DataCrunchClient(client_id, client_secret)

    def get(self, query_filter: Optional[QueryFilter] = None) -> List[RawCatalogItem]:
        items = [
            RawCatalogItem(
                instance_name=instance.instance_type,
                location="FIN-01",
                price=instance.price_per_hour,
                cpu=instance.cpu["number_of_cores"],
                memory=instance.memory["size_in_gigabytes"],
                gpu_count=instance.gpu["number_of_gpus"],
                gpu_name=instance.gpu["description"],
                gpu_memory=instance.gpu_memory["size_in_gigabytes"],
                spot=False,
            )
            for instance in self.datacrunch_client.instance_types.get()
        ]

        return items
