from abc import ABC, abstractmethod
from typing import Optional, List

from gpuhunt._internal.models import RawCatalogItem, QueryFilter


class AbstractProvider(ABC):
    NAME: str = "abstract"

    @abstractmethod
    def get(self, query_filter: Optional[QueryFilter] = None) -> List[RawCatalogItem]:
        pass

    @classmethod
    def filter(cls, offers: List[RawCatalogItem]) -> List[RawCatalogItem]:
        return offers
