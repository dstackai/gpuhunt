from typing import Callable, Optional, Tuple, Union


def empty_as_none(value: Optional[str], loader: Optional[Callable] = None):
    if value is None or value == "":
        return None
    if loader is not None:
        return loader(value)
    return value


def parse_compute_capability(
    value: Optional[Union[str, Tuple[int, int]]]
) -> Optional[Tuple[int, int]]:
    if isinstance(value, str):
        major, minor = value.split(".")
        return int(major), int(minor)
    return value
