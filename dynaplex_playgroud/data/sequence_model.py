"""Convert ActionSet dataclass trees into flat token sequences for Transformer processing.

Instead of building a HeteroData graph, this module walks the same dataclass tree
and produces a flat sequence of feature vectors, each tagged with a type ID.
The Transformer then applies all-to-all self-data over these tokens.

Action grouping
---------------
When an Action dataclass contains nested child objects (other dataclasses or
lists of dataclasses), those children become separate tokens.  All tokens that
belong to the same action sub-tree share an ``action_group_id``.  The root
Action token is additionally marked via ``is_group_root``.

For flat actions (no nesting) each action is its own group of size 1, which
preserves full backward compatibility with existing code.

Intra-group positional ordering
-------------------------------
Each token within an action group is assigned an ``intra_group_position``:
root = 0, then children are numbered sequentially (1, 2, 3, …) in the order
they appear in the dataclass fields (single nested objects first, then list
items preserving their list order).  This lets the policy net add positional
embeddings so that, e.g., ``[Stop_A, Stop_B]`` is distinguishable from
``[Stop_B, Stop_A]``.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Type, get_args, get_origin, get_type_hints
from enum import Enum

import torch

from dynaplex_playgroud.data.action_base import Action, OrderedList


# ============================================================================
# TOKEN TYPES
# ============================================================================

class TokenInfo:
    """Metadata about a single token in the sequence."""
    __slots__ = ("type_name", "features", "is_action", "source_obj",
                 "action_group_id", "is_group_root", "intra_group_position")

    def __init__(
        self,
        type_name: str,
        features: List[float],
        is_action: bool,
        source_obj: Any = None,
        action_group_id: int = -1,
        is_group_root: bool = False,
        intra_group_position: int = 0,
    ):
        self.type_name = type_name
        self.features = features
        self.is_action = is_action
        self.source_obj = source_obj
        self.action_group_id = action_group_id
        self.is_group_root = is_group_root
        self.intra_group_position = intra_group_position


# ============================================================================
# SEQUENCE BUILDER
# ============================================================================

class SequenceBuilder:
    """Walks an ActionSet dataclass tree and produces a flat token sequence.

    Each dataclass instance (GlobalState, Item, MoveAction, etc.) becomes one
    token.  The root ActionSet container is NOT included as a token (it has no
    features of its own).

    Output dict keys
    ~~~~~~~~~~~~~~~~
    - ``type_features``     : dict[str, Tensor]  — per-type raw features
    - ``type_ids``          : Tensor[seq_len]     — integer type id per position
    - ``action_mask``       : BoolTensor[seq_len] — True for every token that
                              belongs to *any* action group (root or child)
    - ``action_group_ids``  : LongTensor[seq_len] — group id (≥ 0) for action
                              sub-tree tokens, -1 for context tokens
    - ``is_group_root``     : BoolTensor[seq_len] — True only for the root
                              Action token of each group
    - ``intra_group_pos``   : LongTensor[seq_len] — position of this token
                              within its action group (0 for root, 1+ for
                              children in order).  0 for context tokens.
    - ``num_action_groups`` : int — number of distinct action groups
    - ``seq_len``           : int
    - ``type_names``        : list of unique type names (index = type_id)
    - ``token_type_names``  : list[str] per-position type name
    - ``indices_by_type``   : dict[str, list[int]] — global indices per type
    """

    def __init__(self, action_base: Type = Action, skip_root: bool = True):
        self.action_base = action_base
        self.skip_root = skip_root
        # Discovered during first call, then frozen
        self._type_name_to_id: Dict[str, int] | None = None
        self._type_names: List[str] | None = None
        self._raw_dims: Dict[str, int] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, root: Any) -> Dict[str, Any]:
        """Convert a root dataclass (e.g. ActionSet) into a flat token dict."""
        tokens: List[TokenInfo] = []
        counter = [0]  # mutable counter for action group IDs
        self._walk(root, tokens, is_root=True,
                   action_group_id=-1, counter=counter)

        # Discover / validate type registry
        if self._type_name_to_id is None:
            self._discover_types(tokens)

        # Build tensors
        seq_len = len(tokens)
        type_ids = torch.zeros(seq_len, dtype=torch.long)
        action_group_ids = torch.full((seq_len,), -1, dtype=torch.long)
        is_group_root = torch.zeros(seq_len, dtype=torch.bool)
        intra_group_pos = torch.zeros(seq_len, dtype=torch.long)
        token_type_names: List[str] = []

        # Group features by type so we can pad to the max dim per type
        features_by_type: Dict[str, List[List[float]]] = {}
        indices_by_type: Dict[str, List[int]] = {}

        for i, tok in enumerate(tokens):
            # Late-discover types not seen in the first sample (e.g. a list
            # that was empty in the first call but non-empty now).
            if tok.type_name not in self._type_name_to_id:
                new_id = len(self._type_name_to_id)
                self._type_name_to_id[tok.type_name] = new_id
                self._type_names.append(tok.type_name)
                self._raw_dims[tok.type_name] = len(tok.features)
                import warnings
                warnings.warn(
                    f"SequenceBuilder: type '{tok.type_name}' was not seen in "
                    f"the first sample — the network may not have an input "
                    f"projection for it (tokens will get zero embeddings).",
                    stacklevel=2,
                )
            tid = self._type_name_to_id[tok.type_name]
            type_ids[i] = tid
            action_group_ids[i] = tok.action_group_id
            is_group_root[i] = tok.is_group_root
            intra_group_pos[i] = tok.intra_group_position
            token_type_names.append(tok.type_name)
            features_by_type.setdefault(tok.type_name, []).append(tok.features)
            indices_by_type.setdefault(tok.type_name, []).append(i)

        # Derive action_mask from action_group_ids (backward compatible)
        action_mask = action_group_ids >= 0

        num_action_groups = counter[0]

        # Create per-type feature tensors (each type may have different raw dim)
        type_features: Dict[str, torch.Tensor] = {}
        for tname, feat_lists in features_by_type.items():
            raw_dim = self._raw_dims[tname]
            tensor = torch.zeros(len(feat_lists), raw_dim, dtype=torch.float32)
            for j, feats in enumerate(feat_lists):
                for k, v in enumerate(feats[:raw_dim]):
                    tensor[j, k] = v
            type_features[tname] = tensor

        return {
            "type_features": type_features,
            "type_ids": type_ids,
            "action_mask": action_mask,
            "action_group_ids": action_group_ids,
            "is_group_root": is_group_root,
            "intra_group_pos": intra_group_pos,
            "num_action_groups": num_action_groups,
            "seq_len": seq_len,
            "type_names": list(self._type_names),
            "token_type_names": token_type_names,
            "indices_by_type": indices_by_type,
        }

    @property
    def type_name_to_id(self) -> Dict[str, int]:
        assert self._type_name_to_id is not None, "Call build() first"
        return dict(self._type_name_to_id)

    @property
    def type_names(self) -> List[str]:
        assert self._type_names is not None, "Call build() first"
        return list(self._type_names)

    @property
    def raw_dims(self) -> Dict[str, int]:
        assert self._raw_dims is not None, "Call build() first"
        return dict(self._raw_dims)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _walk(
        self,
        obj: Any,
        tokens: List[TokenInfo],
        is_root: bool = False,
        action_group_id: int = -1,
        counter: List[int] | None = None,
        ordered_pos_counter: List[int] | None = None,
    ) -> None:
        """Recursively walk the dataclass tree, creating one token per object.

        Parameters
        ----------
        action_group_id:
            -1 means "not inside an action sub-tree".  ≥ 0 means this token
            belongs to the action group with that id (propagated to children).
        counter:
            A mutable ``[int]`` used to allocate fresh group ids.
        ordered_pos_counter:
            A mutable ``[int]`` tracking the next intra-group position for
            items originating from ``OrderedList`` fields.  Shared across
            an action group so that positions are globally sequential.
            ``None`` when not inside an action group.
        """
        if not is_dataclass(obj):
            return

        type_hints = get_type_hints(type(obj))

        # Detect whether *this* object is an Action root
        is_action_root = isinstance(obj, self.action_base) and action_group_id == -1

        # If we just entered an action sub-tree, allocate a new group id
        # and a fresh ordered-position counter (starts at 1; 0 = root)
        if is_action_root:
            action_group_id = counter[0]
            counter[0] += 1
            ordered_pos_counter = [1]  # next available position for ordered items

        # Collect primitive features for this object
        primitives: List[float] = []
        nested_objs: List[Any] = []
        # Each entry: (list_items, is_ordered)
        nested_lists: List[tuple[List[Any], bool]] = []

        for f in fields(obj):
            ftype = type_hints.get(f.name, f.type)
            ftype = _unwrap_optional(ftype)
            value = getattr(obj, f.name)

            if _is_primitive(ftype):
                primitives.append(_to_float(value, ftype))
            elif _is_list_of_dataclass(ftype):
                if value is not None:
                    is_ordered = _is_ordered_list(ftype)
                    nested_lists.append((value, is_ordered))
            elif is_dataclass(ftype):
                if value is not None:
                    nested_objs.append(value)
            # else: skip

        # Add this object as a token (skip root container if requested)
        # Root Action token and non-ordered items get position 0
        if not (is_root and self.skip_root):
            if primitives:  # Only add nodes that have features
                tokens.append(TokenInfo(
                    type_name=type(obj).__name__,
                    features=primitives,
                    is_action=(action_group_id >= 0),
                    source_obj=obj,
                    action_group_id=action_group_id,
                    is_group_root=is_action_root,
                    intra_group_position=0,  # root and non-list items: 0
                ))

        # Recurse into nested single objects (position 0, no ordering)
        for nested in nested_objs:
            self._walk(nested, tokens,
                       action_group_id=action_group_id, counter=counter,
                       ordered_pos_counter=ordered_pos_counter)

        # Recurse into nested lists
        for lst, is_ordered in nested_lists:
            for item in lst:
                if item is None:
                    continue
                if is_ordered and action_group_id >= 0 and ordered_pos_counter is not None:
                    # Ordered list item: assign incrementing position
                    pos = ordered_pos_counter[0]
                    ordered_pos_counter[0] += 1
                    self._walk_with_position(
                        item, tokens, action_group_id, counter,
                        ordered_pos_counter, position=pos,
                    )
                else:
                    # Unordered list item: position 0
                    self._walk(item, tokens,
                               action_group_id=action_group_id, counter=counter,
                               ordered_pos_counter=ordered_pos_counter)

    def _walk_with_position(
        self,
        obj: Any,
        tokens: List[TokenInfo],
        action_group_id: int,
        counter: List[int],
        ordered_pos_counter: List[int],
        position: int,
    ) -> None:
        """Walk a single object, forcing a specific intra-group position.

        Used for items from ``OrderedList`` fields so they receive their
        sequential position.  Any further nested children of this item
        are walked normally (position 0).
        """
        if not is_dataclass(obj):
            return

        type_hints = get_type_hints(type(obj))
        primitives: List[float] = []
        nested_objs: List[Any] = []
        nested_lists: List[tuple[List[Any], bool]] = []

        for f in fields(obj):
            ftype = type_hints.get(f.name, f.type)
            ftype = _unwrap_optional(ftype)
            value = getattr(obj, f.name)

            if _is_primitive(ftype):
                primitives.append(_to_float(value, ftype))
            elif _is_list_of_dataclass(ftype):
                if value is not None:
                    nested_lists.append((value, _is_ordered_list(ftype)))
            elif is_dataclass(ftype):
                if value is not None:
                    nested_objs.append(value)

        if primitives:
            tokens.append(TokenInfo(
                type_name=type(obj).__name__,
                features=primitives,
                is_action=(action_group_id >= 0),
                source_obj=obj,
                action_group_id=action_group_id,
                is_group_root=False,
                intra_group_position=position,
            ))

        # Children of an ordered-list item are walked normally
        for nested in nested_objs:
            self._walk(nested, tokens,
                       action_group_id=action_group_id, counter=counter,
                       ordered_pos_counter=ordered_pos_counter)
        for lst, is_ordered in nested_lists:
            for item in lst:
                if item is not None:
                    self._walk(item, tokens,
                               action_group_id=action_group_id, counter=counter,
                               ordered_pos_counter=ordered_pos_counter)

    def _discover_types(self, tokens: List[TokenInfo]) -> None:
        """Build the type registry from the first sample."""
        seen: Dict[str, int] = {}
        dims: Dict[str, int] = {}
        for tok in tokens:
            if tok.type_name not in seen:
                seen[tok.type_name] = len(seen)
                dims[tok.type_name] = len(tok.features)
            else:
                # Validate consistent feature dim
                assert dims[tok.type_name] == len(tok.features), (
                    f"Inconsistent feature dim for {tok.type_name}: "
                    f"expected {dims[tok.type_name]}, got {len(tok.features)}"
                )
        self._type_name_to_id = seen
        self._type_names = list(seen.keys())
        self._raw_dims = dims


# ============================================================================
# HELPERS
# ============================================================================

def _unwrap_optional(tp: Type) -> Type:
    from typing import Union
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _is_primitive(tp: Type) -> bool:
    if tp in (int, float, bool, str):
        return True
    try:
        if issubclass(tp, Enum):
            return True
    except TypeError:
        print("Unknown type found in dataclass fields:", tp)
        pass
    return False


def _is_list_of_dataclass(tp: Type) -> bool:
    origin = get_origin(tp)
    if origin in (list, List):
        args = get_args(tp)
        if len(args) == 1 and is_dataclass(args[0]):
            return True
    # Also match OrderedList[T] — it's a subclass of list
    if isinstance(tp, type) and issubclass(tp, OrderedList):
        # bare OrderedList without subscript — shouldn't happen in practice
        return False
    # Check if it's OrderedList[SomeDataclass]
    if hasattr(tp, "__origin__") and isinstance(tp.__origin__, type) and issubclass(tp.__origin__, OrderedList):
        args = get_args(tp)
        if len(args) == 1 and is_dataclass(args[0]):
            return True
    return False


def _is_ordered_list(tp: Type) -> bool:
    """Return True if ``tp`` is ``OrderedList[T]`` (order-sensitive)."""
    if hasattr(tp, "__origin__") and isinstance(tp.__origin__, type) and issubclass(tp.__origin__, OrderedList):
        return True
    if isinstance(tp, type) and issubclass(tp, OrderedList):
        return True
    return False


def _to_float(value: Any, tp: Type) -> float:
    if isinstance(value, Enum):
        return float(list(type(value)).index(value))
    if tp == bool:
        return float(value)
    return float(value)

