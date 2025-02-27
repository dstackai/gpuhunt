from abc import ABC, abstractmethod
from typing import Optional

from gpuhunt._internal.models import QueryFilter, RawCatalogItem


class AbstractProvider(ABC):
    """
    Abstract class for cloud provider implementations.

    Attributes:
        NAME: (class variable) The name of the provider.
    """

    NAME: str = "abstract"  # Override in subclasses

    @abstractmethod
    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        """
        Return a list of available instance offers. Offers should be ordered by priority. In most
        cases - by price, ascending.

        Args:
            query_filter: Set of filters requested by the user. Only used with online providers.
                Filters are safe to ignore, as they are also enforced by `gpuhunt` after calling
                `get`. However, they can be used to reduce the number or size of API requests if
                the provider's API supports filtering by GPU, RAM, region, etc.
            balance_resources: Whether the instance resources (CPU, RAM, disk) should be
                adjusted to better match the GPU. Only used with online providers. Only relevant
                to cloud providers that allow configuring instance CPU, RAM, and disk.
        """

        pass

    @classmethod
    def filter(cls, offers: list[RawCatalogItem]) -> list[RawCatalogItem]:
        """
        Return a subset of offers that should be stored in the catalog.

        Only used with offline providers. Only implement this method if there are reasons to omit
        some offers from the catalog.
        """

        return offers
