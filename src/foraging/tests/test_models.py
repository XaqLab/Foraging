# tests/test_models_base.py
import pytest

from foraging.models import HashableDict, SuperDict


def test_hashabledict_hash_and_immutability():
    hd = HashableDict(a=1, b=2)

    # hashable
    assert isinstance(hash(hd), int)

    # equal to plain dict by contents
    assert hd == {"a": 1, "b": 2}

    # usable as a dict key
    d = {hd: "ok"}
    assert d[hd] == "ok"

    # immutable after construction
    with pytest.raises(TypeError):
        hd["a"] = 3

    # "functional" ops return a NEW HashableDict, leaving the original unchanged
    hd2 = hd.update("a", 3)
    assert isinstance(hd2, HashableDict)
    assert hd2 is not hd
    assert hd == {"a": 1, "b": 2}
    assert hd2 == {"a": 3, "b": 2}

    hd3, popped = hd.pop("a")
    assert isinstance(hd3, HashableDict)
    assert hd3 is not hd
    assert popped == 1
    assert hd == {"a": 1, "b": 2}
    assert hd3 == {"b": 2}

    hd4, k, v = hd.popitem()
    assert isinstance(hd4, HashableDict)
    assert hd4 is not hd
    assert (k, v) in {("a", 1), ("b", 2)}
    assert hd == {"a": 1, "b": 2}
    assert len(hd4) == 1

    hd5, val = hd.setdefault("c", 10)
    assert isinstance(hd5, HashableDict)
    assert hd5 is not hd
    assert val == 10
    assert hd == {"a": 1, "b": 2}
    assert hd5 == {"a": 1, "b": 2, "c": 10}

    hd6 = hd.clear()
    assert isinstance(hd6, HashableDict)
    assert hd6 is not hd
    assert hd == {"a": 1, "b": 2}
    assert hd6 == {}


def test_hashabledict_order_independence():
    hd1 = HashableDict({"a": 1, "b": 2})
    hd2 = HashableDict({"b": 2, "a": 1})
    assert hd1 == hd2
    assert hash(hd1) == hash(hd2)


def test_superdict_index():
    sd = SuperDict()
    sd["a"] = 1
    sd["b"] = 2
    sd["c"] = 3
    assert sd.index(0) == ("a", 1)
    assert sd.index(1) == ("b", 2)
    assert sd.index(-1) == ("c", 3)


def test_superdict_sort():
    sd = SuperDict({"b": 2, "a": 1, "c": 3})
    sd.sort()
    assert list(sd.keys()) == ["a", "b", "c"]
    assert list(sd.values()) == [1, 2, 3]
