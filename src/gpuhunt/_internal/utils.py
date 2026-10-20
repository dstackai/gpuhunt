import logging
import sys
from collections.abc import Collection
from typing import TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def get_retrying_session(
    *,
    retries: int = 2,
    backoff_factor: float = 1,
    allowed_methods: Collection[str] = Retry.DEFAULT_ALLOWED_METHODS,
) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        max_retries=Retry(
            total=retries,
            backoff_factor=backoff_factor,
            allowed_methods=allowed_methods,
            status_forcelist=(429, 500, 502, 503, 504),
            raise_on_status=False,
        )
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


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
