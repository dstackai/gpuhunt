import logging
import sys
from collections.abc import Callable
from typing import TypeVar, overload


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(message)s",
    )


T = TypeVar("T")


def get_or_error(v: T | None) -> T:
    """
    Unpacks an optional value. Used to denote that None is not possible in the current context.
    """
    if v is None:
        raise ValueError("Optional value is None")
    return v


R = TypeVar("R")


@overload
def load_optional(value: str | None, loader: Callable[[str], R]) -> R | None:
    pass


@overload
def load_optional(value: str | None, loader: None = None) -> str | None:
    pass


def load_optional(value: str | None, loader: Callable[[str], R] | None = None) -> str | R | None:
    if value is None or value == "":
        return None
    if loader is not None:
        return loader(value)
    return value


@overload
def load_required(value: str | None, loader: Callable[[str], R]) -> R:
    pass


@overload
def load_required(value: str | None, loader: None = None) -> str:
    pass


def load_required(value: str | None, loader: Callable[[str], R] | None = None) -> str | R:
    if value is None:
        raise ValueError("Required value is None")
    if value == "":
        raise ValueError("Required value is empty")
    if loader is not None:
        return loader(value)
    return value


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
