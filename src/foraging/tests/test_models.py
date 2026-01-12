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
    with pytest.raises(TypeError):
        hd.update({"c": 3})
    with pytest.raises(TypeError):
        hd.clear()

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