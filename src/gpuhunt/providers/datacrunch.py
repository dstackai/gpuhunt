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
                gpu_name=gpu_name(instance.gpu["description"]),
                gpu_memory=instance.gpu_memory["size_in_gigabytes"],
                spot=False,
            )
            for instance in self.datacrunch_client.instance_types.get()
        ]

        return items


def gpu_name(name: str) -> str:
    gpu_map = {
        "1x H100 SXM5 80GB": "H100",
        "2x H100 SXM5 80GB": "H100",
        "4x H100 SXM5 80GB": "H100",
        "4x H100 SXM5 80GB": "H100",
        "8x H100 SXM5 80GB": "H100",
        "1x A100 SXM4 80GB": "A100",
        "2x A100 SXM4 80GB": "A100",
        "4x A100 SXM4 80GB": "A100",
        "8x A100 SXM4 80GB": "A100",
        "1x A100 SXM4 40GB": "A100",
        "2x A100 SXM4 40GB": "A100",
        "4x A100 SXM4 40GB": "A100",
        "8x A100 SXM4 40GB": "A100",
        "1x NVidia RTX6000 Ada 48GB": "RTX6000",
        "2x NVidia RTX6000 Ada 48GB": "RTX6000",
        "4x NVidia RTX6000 Ada 48GB": "RTX6000",
        "8x NVidia RTX6000 Ada 48GB": "RTX6000",
        "1x NVidia RTX A6000 48GB": "A6000",
        "2x NVidia RTX A6000 48GB": "A6000",
        "4x NVidia RTX A6000 48GB": "A6000",
        "8x NVidia RTX A6000 48GB": "A6000",
        "1x NVidia Tesla V100 16GB": "V100",
        "2x NVidia Tesla V100 16GB": "V100",
        "4x NVidia Tesla V100 16GB": "V100",
        "8x NVidia Tesla V100 16GB": "V100",
    }
    return gpu_map.get(name)
