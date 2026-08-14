import os
from abc import ABC, abstractmethod

from typing_extensions import Self

from gpuhunt._internal.errors import MissingCredsError
from gpuhunt._internal.models import CatalogItem, QueryFilter


class AbstractProvider(ABC):
    """
    Abstract class for cloud provider implementations.

    Implement `OnlineProvider` or `OfflineProvider` rather than subclassing this directly.

    Attributes:
        NAME: (class variable) The name of the provider.
    """

    NAME: str = "abstract"  # Override in subclasses

    @abstractmethod
    def get(
        self, query_filter: QueryFilter | None = None, balance_resources: bool = True
    ) -> list[CatalogItem]:
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


class OnlineProvider(AbstractProvider, ABC):
    """
    A provider queried at request time, listed in `ONLINE_PROVIDERS`.

    Online providers are constructed by `default_catalog()` in the user's process, so they must
    be constructible from the environment alone.
    """

    @classmethod
    @abstractmethod
    def from_env(cls) -> Self:
        """
        Construct the provider from environment variables.

        Raises:
            MissingCredsError: If a required environment variable is not set. The provider is
                then skipped instead of failing the whole catalog.
        """

        pass


class OfflineProvider(AbstractProvider, ABC):
    """
    A provider collected into the published catalog, listed in `OFFLINE_PROVIDERS`.
    Credentials are supplied by the caller, so a missing one is an error.
    """

    @classmethod
    def filter(cls, offers: list[CatalogItem]) -> list[CatalogItem]:
        """
        Return a subset of offers that should be stored in the catalog.
        Implement this method if there are reasons to omit some offers from the catalog.
        """

        return offers


def get_creds_env(name: str) -> str:
    """
    Reads an environment variable required to construct a provider.

    Raises:
        MissingCredsError: If the variable is not set or empty.
    """

    value = os.getenv(name)
    if not value:
        raise MissingCredsError(f"Set the {name} environment variable")
    return value
