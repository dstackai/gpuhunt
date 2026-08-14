import functools
import importlib
import logging
from collections.abc import Callable
from typing import Concatenate, TypeVar

from typing_extensions import ParamSpec

from gpuhunt._internal.catalog import Catalog
from gpuhunt._internal.errors import MissingCredsError
from gpuhunt.providers.base import OnlineProvider

logger = logging.getLogger(__name__)

# Every provider in `ONLINE_PROVIDERS` must be listed here to be queried by `default_catalog`.
ONLINE_PROVIDER_MODULES = [
    ("gpuhunt.providers.crusoe", "CrusoeProvider"),
    ("gpuhunt.providers.digitalocean", "DigitalOceanProvider"),
    ("gpuhunt.providers.hotaisle", "HotAisleProvider"),
    ("gpuhunt.providers.jarvislabs", "JarvisLabsProvider"),
    ("gpuhunt.providers.seeweb", "SeewebProvider"),
    ("gpuhunt.providers.vastai", "VastAIProvider"),
    ("gpuhunt.providers.vultr", "VultrProvider"),
]


@functools.lru_cache
def default_catalog() -> Catalog:
    """
    Returns:
        the latest catalog with all available providers loaded
    """
    catalog = Catalog()
    catalog.load()
    for module_name, class_name in ONLINE_PROVIDER_MODULES:
        try:
            module = importlib.import_module(module_name)
            provider_class: type[OnlineProvider] = getattr(module, class_name)
            catalog.add_provider(provider_class.from_env())
        except ImportError:
            logger.warning("Failed to import provider %s", class_name)
        except MissingCredsError as e:
            logger.warning("Skipping provider %s: %s", class_name, e)
    return catalog


P = ParamSpec("P")
R = TypeVar("R")
Method = Callable[P, R]
CatalogMethod = Callable[Concatenate[Catalog, P], R]


def with_signature(method: CatalogMethod[P, R]) -> Callable[[Method[P, R]], Method[P, R]]:
    """
    Returns:
        decorator to add the signature of the Catalog method to the decorated method
    """

    def decorator(func: Method) -> Method:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        return wrapper

    return decorator


@with_signature(Catalog.query)
def query(*args, **kwargs):
    """
    Query the `default_catalog`.
    See `Catalog.query` for more details on parameters

    Returns:
        (List[CatalogItem]): the result of the query
    """
    return default_catalog().query(*args, **kwargs)
