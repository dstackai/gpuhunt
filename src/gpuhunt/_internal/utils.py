from typing import Callable, Optional, TypeVar, Union

T = TypeVar("T", bound=Union[int, float])


def is_between(value: T, left: Optional[T], right: Optional[T]) -> bool:
    if is_below(value, left) or is_above(value, right):
        return False
    return True


def is_below(value: T, limit: Optional[T]) -> bool:
    if limit is not None and value < limit:
        return True
    return False


def is_above(value: T, limit: Optional[T]) -> bool:
    if limit is not None and value > limit:
        return True
    return False


def empty_as_none(value: Optional[str], loader: Optional[Callable] = None):
    if value is None or value == "":
        return None
    if loader is not None:
        return loader(value)
    return value


def optimize(available: T, min_limit: T, max_limit: Optional[T]) -> Optional[T]:
    if is_above(available, max_limit):
        available = max_limit
    if is_below(available, min_limit):
        return None
    return min_limit
