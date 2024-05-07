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


def to_camel_case(snake_case: str) -> str:
    words = snake_case.split("_")
    words = list(filter(None, words))
    words[1:] = [word[:1].upper() + word[1:] for word in words[1:]]
    return "".join(words)
