import logging
import sys
from typing import TypeVar


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(message)s",
    )


T = TypeVar("T")


def get_or_error(v: T | None, name: str = "value") -> T:
    """
    Unpacks an optional value. Used to denote that None is not possible in the current context.
    """
    if v is None:
        raise ValueError(f"Expected {name} to be set")
    return v


def parse_compute_capability(
    value: str | tuple[int, int] | None,
) -> tuple[int, int] | None:
    if isinstance(value, str):
        major, minor = value.split(".")
        return int(major), int(minor)
    return value


def to_camel_case(snake_case: str) -> str:
    words = snake_case.split("_")
    words = list(filter(None, words))
    words[1:] = [word[:1].upper() + word[1:] for word in words[1:]]
    return "".join(words)
