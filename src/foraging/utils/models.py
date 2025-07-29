import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from itertools import islice, permutations, product
from typing import Any, Optional, Type

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike
from scipy.stats import gamma

from foraging.utils import INDEX, MIN_INDEX


## ABSTRACT CLASSES
class AbstractBelief(ABC):

    @abstractmethod
    def prior(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def support(self, *args, **kwargs):
        pass

    @abstractmethod
    def normalize(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def query(self, *args, **kwargs):
        pass

    @abstractmethod
    def features(self, *args, **kwargs):
        pass


class AbstractID(ABC):

    @abstractmethod
    def fields(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def values(self, *args, **kwargs) -> Any:
        pass


class AbstractRecord(ABC):

    @abstractmethod
    def id(self, *args, **kwargs) -> AbstractID:
        pass

    @abstractmethod
    def content(self, *args, **kwargs) -> Any:
        pass


class AbstractRecordKeeper(AbstractRecord):

    @abstractmethod
    def records(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def add(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def update_record(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def delete(self, *args, **kwargs) -> Any:
        pass


# class HistoryManager:
#     def __init__(self):
#         self.history = []

#     def add(self, record):
#         self.history.append(record)

#     def get(self, index: int):
#         return self.history[index]

#     def get_all(self):
#         return self.history

#     def clear(self):
#         self.history = []


### IMPLEMENTATIONS
class ArrayBelief(AbstractBelief):

    def __init__(self, support: np.ndarray, probabilities: np.ndarray = None):
        self._support = support
        if probabilities is None:
            probabilities = np.ones(len(support)) / len(support)
        self._prior = probabilities
        self._features = probabilities

    @property
    def support(self) -> np.ndarray:
        return self._support

    @support.setter
    def support(self, support: np.ndarray):
        self._support = support

    @property
    def features(self) -> np.ndarray:
        return self._features

    @features.setter
    def features(self, features: np.ndarray):
        self._features = features
        self.normalize()

    @property
    def prior(self) -> np.ndarray:
        return self._prior

    @prior.setter
    def prior(self, prior: np.ndarray) -> Any:
        self._prior = prior

    def normalize(self, inplace: bool = True):
        if inplace:
            self._features /= self._features.sum()
            return self._features
        else:
            return self._features / self._features.sum()

    def query(self, i: int, **kwargs):
        return self.features[i]

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def likelihood(self, *args, **kwargs) -> Any:
        pass


class GammaBoxBelief(ArrayBelief):

    def __init__(
        self, shape: int = 1, schedules: np.ndarray = None, prior: ArrayBelief = None
    ):
        if prior:
            super().__init__(prior.support, prior.features)
        else:
            super().__init__(schedules)
        self.shape = shape

    def likelihood(self, obs: tuple[bool, float], schedule: np.ndarray) -> np.ndarray:
        # Extract the reward availability and push interval.
        is_avail = obs[0]
        t = obs[1]

        # Calculate the probability of reward being available/unavailable after t time has passed under the given latents.
        p_t = gamma.cdf(t, self.shape, scale=schedule / self.shape)

        # The last element is the probability of being in the last state, i.e., reward being available.
        if (
            not is_avail
        ):  # If reward is not available, return the complementary probability.
            return 1.0 - p_t
        return p_t

    def update(self, obs: tuple[bool, float], **kwargs):
        self.features = (
            self.likelihood(obs, self.support) * self.features
        )  # this automatically normalizes due to setter property!!


class IndependentBoxesBelief(AbstractBelief):
    def __init__(self, n_boxes: int, belief_cls: Type[AbstractBelief], *args, **kwargs):
        self.n_boxes = n_boxes
        self.belief_cls = belief_cls
        self.beliefs = [belief_cls(*args, **kwargs) for _ in range(n_boxes)]

    def update(self, box: int, *args, **kwargs):
        self.beliefs[box].update(*args, **kwargs)

    def query(self, i: int, *args, **kwargs):
        return self.beliefs[i].query(*args, **kwargs)

    def normalize(self, *args, **kwargs):
        for belief in self.beliefs:
            belief.normalize(*args, **kwargs)

    @property
    def prior(self):
        return [self.beliefs[i].prior for i in range(self.n_boxes)]

    @prior.setter
    def prior(self, prior: AbstractBelief):
        for belief in self.beliefs:
            belief.prior = prior

    @property
    def support(self):
        return [self.beliefs[i].support for i in range(self.n_boxes)]

    @support.setter
    def support(self, support: ArrayLike):
        for belief in self.beliefs:
            belief.support = support

    @property
    def features(self):
        return [self.beliefs[i].features for i in range(self.n_boxes)]

    @features.setter
    def features(self, features: ArrayLike):
        for i, belief in enumerate(self.beliefs):
            belief.features = features[i]


class PermutationBelief(ArrayBelief):
    def __init__(self, params: list):
        super().__init__(list(permutations(params)))


class GammaBoxPermutationBelief(PermutationBelief):
    def __init__(self, schedules: list, shape: int = 1):
        super().__init__(schedules)
        self.shape = shape

    def likelihood(self, obs: tuple[bool, float, int], perm: list):
        # Extract the reward availability and push interval.
        is_avail = obs[0]
        t = obs[1]
        schedule = perm[obs[2]]

        # Calculate the probability of reward being available/unavailable after t time has passed under the given latents.
        p_t = gamma.cdf(t, self.shape, scale=schedule / self.shape)

        # The last element is the probability of being in the last state, i.e., reward being available.
        if (
            not is_avail
        ):  # If reward is not available, return the complementary probability.
            return 1.0 - p_t
        return p_t

    def update(self, obs: tuple[bool, float, int]):
        self.features = (
            self.likelihood(obs) * self.features
        )  # this automatically normalizes due to setter property!!


class RealID(AbstractID):
    def __init__(self, values: tuple, fields: tuple[str]):
        self._values = values
        self._fields = fields

    @property
    def fields(self) -> tuple[str]:
        return self._fields

    @fields.setter
    def fields(self, fields: tuple[str]):
        self._fields = fields

    @property
    def values(self) -> tuple:
        return self._values

    @values.setter
    def values(self, values: tuple):
        self._values = values

    def __eq__(self, other: AbstractID) -> bool:
        return self.values == other.values and all(
            x == y for x, y in zip(self.fields, other.fields)
        )

    def __hash__(self) -> int:
        return hash((self.values, self.fields))


class MockID(RealID):
    def __init__(self, values: int):
        super().__init__((values,), ("id",))


class EventID(RealID):

    def __init__(self, values: tuple, minimal: bool = False):
        if minimal:
            super().__init__(values, INDEX[:MIN_INDEX])
        else:
            super().__init__(values, INDEX)


class Record(AbstractRecord):

    def __init__(self, id: AbstractID, record: Any):
        self._id = id
        self._record = record

    @property
    def id(self) -> AbstractID:
        return self._id

    @id.setter
    def id(self, id: AbstractID):
        self._id = id

    @property
    def content(self) -> Any:
        return self._record

    @content.setter
    def content(self, record: Any):
        self._record = record


class RecordKeeper(AbstractRecordKeeper):

    def __init__(self, init_id: AbstractID = None, init_record: Any = None):
        self._records = {}
        if init_record and init_id:
            record = Record(init_id, init_record)
            self._records[init_id] = record

    def id(self, i: int) -> AbstractID:
        if i < 0:
            i = len(self) + i
        return list(islice(self.records.values(), i, i + 1))[0].id

    def content(self, i: int = None, id: AbstractID = None) -> Any:
        id = self.id(i) if id is None else id
        return self._records[id]

    @property
    def records(self) -> dict[AbstractID, Record]:
        return self._records

    @records.setter
    def records(self, records: dict[AbstractID, Record]):
        self._records = records

    def add(self, id: AbstractID, record: Any):
        self._records[id] = Record(id, record)

    def update_record(self, id: AbstractID, record: Any) -> Any:
        old_record = self.records[id].content
        self.records[id].content = record
        return old_record

    def delete(self, id: AbstractID) -> Any:
        return self.records.pop(id, None).content

    def __len__(self):
        return len(self.records)


class BeliefModule(RecordKeeper, AbstractBelief):

    def __init__(self, init_id: AbstractID, init_belief: AbstractBelief):
        super().__init__(init_id, init_belief)

    @property
    def prior(self) -> AbstractBelief:
        return self.content(i=0).content

    @prior.setter
    def prior(self, prior: AbstractBelief):
        self.update_record(self.content(i=0).id, prior)

    @property
    def support(self):
        return self.prior.support

    @support.setter
    def support(self, support):
        # Change the support for all beliefs
        for _, record in self.records.items():
            record.content.support = support

    @property
    def features(self):
        return [self.records[id].content.features for id in self.records.keys()]

    @features.setter
    def features(self, features: dict[AbstractID, ArrayLike]):
        for id, record in self.records.items():
            record.content.features = features[id]

    def update(self, new_id: AbstractID, *args, **kwargs):

        # Step 1: Create a deep copy of the last record
        last_record = self.content(i=-1)
        new_record = deepcopy(last_record)

        # Step 2: Invoke the update method on the contents of this copy
        new_record.content.update(*args, **kwargs)

        # Step 3: Use the user-supplied AbstractID to augment the internal RecordKeeper structure
        self.add(new_id, new_record.content)

    def normalize(self, *args, **kwargs):
        # Normalize all beliefs
        for id, record in self.records.items():
            record.content.normalize(*args, **kwargs)

    def query(self, i: int = None, id: AbstractID = None):
        return self.content(i, id)
