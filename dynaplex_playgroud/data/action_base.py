"""Shared base classes and type markers for dataclass-based MDPs."""
from __future__ import annotations

from abc import ABC
from typing import TypeVar, List

T = TypeVar("T")


class Action(ABC):
    """Marker base class for action dataclasses."""

    pass


class GlobalState(ABC):
    """Marker base class for global state dataclasses."""

    pass


class OrderedList(List[T]):
    """Marker type for lists where item order matters.

    Use as a type annotation on dataclass fields::

        @dataclass
        class MyAction(Action):
            steps: OrderedList[Step]    # order-sensitive → positional embeddings
            items: list[Item]           # order-invariant → no positional signal

    The ``SequenceBuilder`` detects this annotation and assigns incrementing
    ``intra_group_position`` values (1, 2, 3, …) to the list items.  Plain
    ``list`` fields get position 0 for all items (same as root), making the
    positional embedding a no-op.
    """
    pass

