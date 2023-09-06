from functools import lru_cache

from gpuhunt._catalog import Catalog


@lru_cache()
def default_catalog() -> Catalog:
    catalog = Catalog()
    catalog.load()
    return catalog


def query() -> list:  # todo
    return default_catalog().query()
