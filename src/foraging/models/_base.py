from typing import Any
from itertools import islice

class HashableDict(dict):
    """
    A dictionary that is hashable.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hash = hash(frozenset(self.items()))
        self._frozen = True

    def __copy__(self):
        # HashableDict is immutable after construction; copying can safely return self.
        return self

    def __deepcopy__(self, memo):
        # HashableDict is immutable after construction; deep copying can safely return self.
        return self

    def __hash__(self):
        return self._hash

    def update(self, key, value) -> "HashableDict":
        mutable = dict(self)
        mutable[key] = value
        return HashableDict(mutable)

    def pop(self, key) -> tuple["HashableDict", Any]:
        mutable = dict(self)
        value = mutable.pop(key)
        return HashableDict(mutable), value

    def popitem(self) -> tuple["HashableDict", Any, Any]:
        mutable = dict(self)
        key, value = mutable.popitem()
        return HashableDict(mutable), key, value

    def setdefault(self, key, value) -> tuple["HashableDict", Any]:
        mutable = dict(self)
        value = mutable.setdefault(key, value)
        return HashableDict(mutable), value

    def clear(self) -> "HashableDict":
        mutable = dict(self)
        mutable.clear()
        return HashableDict(mutable)

    # block mutation directly on object after construction
    def _blocked(self, *a, **k): raise TypeError("HashableDict is immutable")
    __setitem__ = __delitem__ = _blocked    

class SuperDict(dict):
    """
    A dictionary with some convenience functionality. Supports attribute access.
    """

    def index(self, i: int) -> tuple[Any, Any]:
        if i < 0:
            i = len(self) + i
        return next(islice(self.items(), i, i + 1))

    def sort(self):
        """Recreate dict with sorted items"""
        items_sorted = dict(sorted(self.items()))
        self.clear()
        super().update(items_sorted)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
