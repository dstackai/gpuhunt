from functools import lru_cache

from dstack.pricing._catalog import Catalog
from dstack.pricing._utils import print_table


@lru_cache()
def default_catalog() -> Catalog:
    catalog = Catalog()
    catalog.load()
    return catalog


def query() -> list:  # todo
    return default_catalog().query()
