"""
Provider-specific models used within the catalog.

These models must be importable without provider dependencies installed,
so they reside here in a separate module instead of gpuhunt.providers.
"""

from typing import TypeVar

from pydantic import BaseModel, Extra

T = TypeVar("T")


class CatalogItemProviderData(BaseModel, extra=Extra.ignore):
    """
    Base class for provider-specific catalog item properties.

    Providers that have any provider-specific properties define subclasses.
    Other providers use this class as a concrete class without any properties.

    To access provider-specific properties, cast to the provider's subclass
    using its utility method. Example:

    ```
    item = CatalogItem(...)
    item.provider_data.gcp().is_dws_calendar_mode
    ```
    """

    def gcp(self) -> "GCPCatalogItemData":
        return self._cast(GCPCatalogItemData)

    def _cast(self, subclass_: type[T]) -> T:
        if isinstance(self, subclass_):
            return self
        raise TypeError(f"Cannot cast {type(self)} to {subclass_}")


class GCPCatalogItemData(CatalogItemProviderData):
    is_dws_calendar_mode: bool = False


# TODO: test all attributes have defaults
# TODO: test all subclasses are registered here
CATALOG_ITEM_PROVIDER_DATA_MODELS: dict[str, CatalogItemProviderData] = {
    "gcp": GCPCatalogItemData,
}
