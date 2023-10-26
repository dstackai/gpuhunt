from typing import Callable, Optional


def empty_as_none(value: Optional[str], loader: Optional[Callable] = None):
    if value is None or value == "":
        return None
    if loader is not None:
        return loader(value)
    return value
