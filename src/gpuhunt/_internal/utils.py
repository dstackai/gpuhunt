from typing import TypeVar, Union, Optional, Callable

T = TypeVar("T", bound=Union[int, float])


def is_between(value: T, left: Optional[T], right: Optional[T]) -> bool:
    if left is not None and value < left:
        return False
    if right is not None and value > right:
        return False
    return True


def empty_as_none(value: Optional[str], loader: Optional[Callable] = None):
    if value is None or value == "":
        return None
    if loader is not None:
        return loader(value)
    return value
