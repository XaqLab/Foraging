import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice, permutations, product
from typing import Any, Callable, NamedTuple, Optional, Protocol, Type, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gamma

from foraging.utils import INDEX, MIN_INDEX, flatten

## TYPES
O = TypeVar("O")
X = TypeVar("X")
ID = TypeVar("ID")
R = TypeVar("R")


@dataclass
class RewardObservation:
    is_available: bool
    time: float


@dataclass
class GammaParameters:
    shape: float
    schedule: float


@dataclass
class PossibleSchedules:
    """
    A convenience class that behaves like an array of schedules but also has structured fields.
    When converted with np.asarray(), it becomes an array of schedules.
    When passed to GammaLikelihood, it provides schedule and shape fields.
    """

    shape: float
    schedule: Iterable[float]

    def __array__(self, dtype=None):
        """Make this class convertible to NumPy array."""
        return np.asarray(self.schedule, dtype=dtype)

    def __len__(self):
        return len(self.schedule)

    def __iter__(self):
        return iter(self.schedule)


## INTERFACES AND ABSTRACTIONS


# tests
class Uniform:
    def __init__(self, support: Iterable[X]):
        self.support = support
        self._support = np.asarray(support)  # for indexing
        self._representation = np.ones(len(support)) / len(support)

    def representation(self) -> ArrayLike:
        return self._representation

    def sample(self, n: int = 1) -> Iterable[X]:
        return np.random.choice(self._support, size=n, p=self.representation())

    def query(self, x: X) -> float:
        return self.representation()[np.argwhere(self._support == x)[0][0]]


class Belief(Protocol[X]):
    """
    Interface for a belief.
    Functionally, a belief supports sampling and querying.
    Structurally, a belief has a representation ie. probabilities, sufficient statistics, etc.
    """

    @property
    def representation(self) -> Any: ...

    def sample(self, n: int = 1) -> Iterable[X]: ...

    def query(self, x: X) -> float: ...


class BeliefCollection(Protocol[X]):
    """Convenience class for a collection of beliefs."""

    def __init__(self, n: int, belief_cls: Type[Belief[X]], *args, **kwargs):
        self.belief_cls = belief_cls
        self.beliefs = [belief_cls(*args, **kwargs) for _ in range(n)]

    def __getitem__(self, i: int) -> Belief[X]:
        return self.beliefs[i]

    def __setitem__(self, i: int, belief: Belief[X]):
        self.beliefs[i] = belief

    @property
    def representation(self) -> Iterable[Any]:
        return [belief.representation() for belief in self.beliefs]

    def sample(self, n: int = 1) -> Iterable[X]:
        return [belief.sample(n) for belief in self.beliefs]

    def query(self, x: Iterable[X]) -> Iterable[float]:
        return [belief.query(_x) for belief, _x in zip(self.beliefs, x)]

    def query_by_index(self, i: int, x: X) -> float:
        return self.beliefs[i].query(x)


class Likelihood(Protocol[X, O]):
    """
    Interface for a likelihood.
    """

    def __call__(self, o: O, x: X) -> float | Iterable[float]: ...


class BeliefUpdate(Protocol[X, O]):
    """
    Encapsulates the belief update rule ie. Bayesian, online variational methods, neural networks, etc.
    """

    def __call__(self, prior: Belief[X], o: O) -> Belief[X]: ...


class UpdatesByBox(Protocol[O, ID]):
    """Convenience protocol for updating beliefs with a box-specific observation."""

    def update(self, box_position: int, id: ID, o: O): ...


## IMPLEMENTATIONS
class Probabilities:
    """
    A belief that has finite and discrete probabilities as the representation.
    """

    def __init__(self, support: Iterable[X], probabilities: Iterable[float]):
        self.support = support
        self._support = np.asarray(support)  # for indexing
        self._representation = np.asarray(probabilities)

    @property
    def representation(self) -> ArrayLike:
        return self._representation

    @representation.setter
    def representation(self, value: Iterable[float]):
        value = np.asarray(value)
        assert np.all(value >= 0), "Probabilities must be non-negative"
        assert np.isclose(np.sum(value), 1.0), "Probabilities must sum to 1"
        self._representation = value

    def sample(self, n: int = 1) -> Iterable[X]:
        return np.random.choice(self._support, size=n, p=self.representation)

    def query(self, x: X) -> float:
        return self.representation[np.argwhere(self._support == x)[0][0]]

    def query_by_index(self, i: int) -> float:
        return self.representation[i]

    def __len__(self) -> int:
        return len(self.support)


class ExactBayesianUpdateOnProbabilities:
    """
    Exact Bayesian update on Probabilities.
    """

    def __init__(self, likelihood: Likelihood[X, O], vectorized: bool = False):
        """Set vectorized to True to vectorize likelihood over the parameters."""
        self.likelihood = likelihood
        self.vectorized = vectorized

    def __call__(self, prior: Probabilities, data: O) -> Probabilities:
        # Bayes rule: posterior ∝ likelihood × prior
        if self.vectorized:
            posterior_probs = (
                self.likelihood(data, prior.support) * prior.representation
            )
        else:
            posterior_probs = np.array(
                [
                    self.likelihood(data, x) * prior.representation[i]
                    for i, x in enumerate(prior.support)
                ]
            )

        # Normalize
        posterior_probs = posterior_probs / np.sum(posterior_probs)
        return Probabilities(prior.support, posterior_probs)


class RecordKeeper(Protocol[ID, R]):

    def __init__(self, init_id: ID = None, init_record: R = None):
        self._records = {}
        if init_record and init_id:
            self._records[init_id] = init_record

    def id(self, i: int) -> ID:
        if i < 0:
            i = len(self) + i
        return list(self._records.keys())[i]

    def __getitem__(self, id: ID) -> R:
        return self._records[id]

    def __setitem__(self, id: ID, record: R):
        self._records[id] = record

    @property
    def records(self) -> dict[ID, R]:
        return self._records

    @records.setter
    def records(self, records: dict[ID, R]):
        self._records = records

    def delete(self, id: ID) -> R:
        return self.records.pop(id, None)

    def __len__(self):
        return len(self.records)

    def sort(self):
        self.records = dict(sorted(self.records.items()))


class Posterior(RecordKeeper[ID, Belief[X]]):
    def __init__(self, init_id: ID, prior: Belief[X], update: BeliefUpdate[X, O]):
        super().__init__(init_id, prior)
        self.update = update

    def prior(self) -> Belief[X]:
        return self[self.id(0)]

    def representation(self):
        return [self[id].representation() for id in self.records.keys()]

    def update(self, id: ID, o: O):
        self[id] = self.update(self[self.id(-1)], o)

    def query(self, x: X) -> float:
        return self[self.id(-1)].query(x)

    def sample(self, n: int = 1) -> Iterable[X]:
        return self[self.id(-1)].sample(n)


class FactorizedPosterior(Posterior[ID, Belief[X]]):
    """
    A factorized posterior that inherits from Posterior and implements UpdatesByBox.
    Each factor is managed as a separate Posterior instance, allowing independent updates.
    """

    def __init__(
        self,
        n_factors: int,
        belief_cls: Type[Belief[X]],
        update: BeliefUpdate[X, O],
        factor_ids: list[ID] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize a factorized posterior with n_factors independent posteriors.

        Args:
            n_factors: Number of independent factors
            belief_cls: Class to instantiate for each factor's prior
            update: Belief update function to apply to individual factors
            factor_ids: Optional list of IDs for each factor (defaults to [0, 1, 2, ...])
            *args, **kwargs: Arguments passed to belief_cls constructor
        """
        # Create factor IDs if not provided
        if factor_ids is None:
            factor_ids = list(range(n_factors))

        # Initialize the first factor as the main posterior
        first_prior = belief_cls(*args, **kwargs)
        super().__init__(factor_ids[0], first_prior, update)

        # Store the update function and factor information
        self._update_func = update
        self._factor_ids = factor_ids
        self._belief_cls = belief_cls
        self._belief_args = args
        self._belief_kwargs = kwargs

        # Initialize remaining factors
        for i in range(1, n_factors):
            prior = belief_cls(*args, **kwargs)
            self[factor_ids[i]] = prior

    def update(self, box_position: int, id: ID, o: O):
        """
        Update a specific factor based on box position and observation.
        This method satisfies the UpdatesByBox protocol.

        Args:
            box_position: Which factor to update (0 to n_factors-1)
            id: Identifier (unused, kept for protocol compatibility)
            o: Observation containing the data for updating
        """
        if box_position < 0 or box_position >= len(self._factor_ids):
            raise ValueError(
                f"Box position {box_position} out of range [0, {len(self._factor_ids)-1}]"
            )

        # Get the factor ID for this box position
        factor_id = self._factor_ids[box_position]

        # Update the specific factor using the parent Posterior's update mechanism
        self[factor_id] = self._update_func(self[factor_id], o)

    def query_factor(self, factor_idx: int, x: X) -> float:
        """
        Query a specific factor's belief.

        Args:
            factor_idx: Index of the factor to query
            x: Value to query

        Returns:
            Probability of x under the specified factor's belief
        """
        if factor_idx < 0 or factor_idx >= len(self._factor_ids):
            raise ValueError(
                f"Factor index {factor_idx} out of range [0, {len(self._factor_ids)-1}]"
            )

        factor_id = self._factor_ids[factor_idx]
        return self[factor_id].query(x)

    def sample_factor(self, factor_idx: int, n: int = 1) -> Iterable[X]:
        """
        Sample from a specific factor's belief.

        Args:
            factor_idx: Index of the factor to sample from
            n: Number of samples to draw

        Returns:
            Samples from the specified factor's belief
        """
        if factor_idx < 0 or factor_idx >= len(self._factor_ids):
            raise ValueError(
                f"Factor index {factor_idx} out of range [0, {len(self._factor_ids)-1}]"
            )

        factor_id = self._factor_ids[factor_idx]
        return self[factor_id].sample(n)

    def get_factor_representation(self, factor_idx: int) -> Any:
        """
        Get the representation of a specific factor.

        Args:
            factor_idx: Index of the factor

        Returns:
            Representation of the specified factor's belief
        """
        if factor_idx < 0 or factor_idx >= len(self._factor_ids):
            raise ValueError(
                f"Factor index {factor_idx} out of range [0, {len(self._factor_ids)-1}]"
            )

        factor_id = self._factor_ids[factor_idx]
        return self[factor_id].representation

    def get_factor_id(self, factor_idx: int) -> ID:
        """
        Get the ID of a specific factor.

        Args:
            factor_idx: Index of the factor

        Returns:
            The ID of the specified factor
        """
        if factor_idx < 0 or factor_idx >= len(self._factor_ids):
            raise ValueError(
                f"Factor index {factor_idx} out of range [0, {len(self._factor_ids)-1}]"
            )

        return self._factor_ids[factor_idx]

    @property
    def factor_ids(self) -> list[ID]:
        """Get the list of factor IDs."""
        return self._factor_ids.copy()

    def __len__(self) -> int:
        return len(self._factor_ids)


class GammaLikelihood:
    def __call__(
        self, o: RewardObservation, x: GammaParameters
    ) -> float | Iterable[float]:
        # Calculate the probability of reward being available/unavailable after t time has passed under the given latents.
        p_t = gamma.cdf(o.time, x.shape, scale=x.schedule / x.shape)

        # The last element is the probability of being in the last state, i.e., reward being available.
        if (
            not o.is_available
        ):  # If reward is not available, return the complementary probability.
            return 1.0 - p_t
        return p_t


## ABSTRACT CLASSES
class AbstractBelief(ABC):
    """
    Abstract class for a belief.
    Structurally, at a minimum, a belief has a prior (which is also a belief), support, and features ie. probabilities, parameters, arbitrary representations, etc.
    Functionally, at a minimum, a belief supports updating after receiving an observation, normalizing after updating, querying the probability of a given value, and sampling from the belief (if computable).
    """

    @abstractmethod
    def prior(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def support(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def features(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def normalize(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def query(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def sample(self, *args, **kwargs) -> Any:
        pass


class BoxesBeliefContainer(ABC):
    """
    A convenience abstraction for representing a set of beliefs about boxes, regardless of the structure of the beliefs ie. 3 independent beliefs for each box vs 1 belief over permutations of fixed schedules.
    Requires update method to take a box-related variable as an argument.
    """

    @abstractmethod
    def update(self, box: int = None, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class IndexedBelief(AbstractBelief):
    """
    A belief that is indexed by an integer.
    This is useful for representing a belief that is indexed by a box number, for example.
    The update method is then called with the box number as the first argument.
    The query method is then called with the box number as the first argument.
    """

    @abstractmethod
    def update(self, i: int, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def query(self, i: int, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


### IMPLEMENTATIONS
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

    def __lt__(self, other: AbstractID) -> bool:
        return self.values < other.values


class IntID(RealID):
    def __init__(self, values: int):
        super().__init__((values,), ("id",))


class EventID(RealID):

    def __init__(self, values: tuple, minimal: bool = False):
        if minimal:
            super().__init__(values, INDEX[:MIN_INDEX])
        else:
            super().__init__(values, INDEX)


class RecordKeeper(AbstractRecordKeeper):

    def __init__(self, init_id: AbstractID = None, init_record: Any = None):
        self._records = {}
        if init_record and init_id:
            self._records[init_id] = init_record

    def id(self, i: int) -> AbstractID:
        if i < 0:
            i = len(self) + i
        return list(self._records.keys())[i]

    def __getitem__(self, id: AbstractID) -> Any:
        return self._records[id]

    def __setitem__(self, id: AbstractID, record: Any) -> Any:
        self._records[id] = record

    @property
    def records(self) -> dict[AbstractID, Any]:
        return self._records

    @records.setter
    def records(self, records: dict[AbstractID, Any]):
        self._records = records

    def delete(self, id: AbstractID) -> Any:
        return self.records.pop(id, None)

    def __len__(self):
        return len(self.records)

    def sort(self):
        self.records = dict(sorted(self.records.items()))


class ArrayBelief(AbstractBelief):
    """
    A belief that is represented as an array of probabilities as the features.
    """

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

    def sample(self, n: int = 1, **kwargs) -> np.ndarray:
        return np.random.choice(self.support, size=n, p=self.features)

    def __len__(self) -> int:
        return len(self.support)


class GammaBoxBelief(ArrayBelief):
    """
    An implementation of the beliefs of a box where reward intervals are gamma distributed and observations are reward outcomes.
    """

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


class IndependentBoxesBelief(IndexedBelief):
    """
    A belief that is represented as a collection of independent beliefs, one for each box.
    """

    def __init__(self, n_boxes: int, belief_cls: Type[AbstractBelief], *args, **kwargs):
        self.n_boxes = n_boxes
        self.belief_cls = belief_cls
        self.beliefs = [belief_cls(*args, **kwargs) for _ in range(n_boxes)]

    def update(self, i: int, *args, **kwargs):
        self.beliefs[i].update(*args, **kwargs)

    def query(self, i: int, *args, **kwargs):
        return self.beliefs[i]

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

    def __len__(self):
        return self.n_boxes


class IndependentGammaBoxesBelief(IndependentBoxesBelief):
    def __init__(self, n_boxes: int, *args, **kwargs):
        super().__init__(n_boxes, GammaBoxBelief, *args, **kwargs)


class PermutationBelief(ArrayBelief):
    def __init__(self, params: list):
        super().__init__(list(permutations(params)))


class PermutationGammaBoxesBelief(PermutationBelief):
    def __init__(self, schedules: list, shape: int = 1):
        super().__init__(schedules)
        self.shape = shape

    def likelihood(self, i: int, obs: tuple[bool, float], perm: list):
        # Extract the reward availability and push interval.
        is_avail = obs[0]
        t = obs[1]
        schedule = perm[i]

        # Calculate the probability of reward being available/unavailable after t time has passed under the given latents.
        p_t = gamma.cdf(t, self.shape, scale=schedule / self.shape)

        # The last element is the probability of being in the last state, i.e., reward being available.
        if (
            not is_avail
        ):  # If reward is not available, return the complementary probability.
            return 1.0 - p_t
        return p_t

    def update(self, i: int, obs: tuple[bool, float]):
        new_features = np.zeros(len(self.support))
        for j, perm in enumerate(self.support):
            new_features[j] = self.likelihood(i, obs, perm) * self.features[j]
        self.features = new_features


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

    def __lt__(self, other: AbstractID) -> bool:
        return self.values < other.values


class IntID(RealID):
    def __init__(self, values: int):
        super().__init__((values,), ("id",))


class EventID(RealID):

    def __init__(self, values: tuple, minimal: bool = False):
        if minimal:
            super().__init__(values, INDEX[:MIN_INDEX])
        else:
            super().__init__(values, INDEX)


class RecordKeeper(AbstractRecordKeeper):

    def __init__(self, init_id: AbstractID = None, init_record: Any = None):
        self._records = {}
        if init_record and init_id:
            self._records[init_id] = init_record

    def id(self, i: int) -> AbstractID:
        if i < 0:
            i = len(self) + i
        return list(self._records.keys())[i]

    def __getitem__(self, id: AbstractID) -> Any:
        return self._records[id]

    def __setitem__(self, id: AbstractID, record: Any) -> Any:
        self._records[id] = record

    @property
    def records(self) -> dict[AbstractID, Any]:
        return self._records

    @records.setter
    def records(self, records: dict[AbstractID, Any]):
        self._records = records

    def delete(self, id: AbstractID) -> Any:
        return self.records.pop(id, None)

    def __len__(self):
        return len(self.records)

    def sort(self):
        self.records = dict(sorted(self.records.items()))


class Posterior(RecordKeeper, AbstractBelief):

    def __init__(self, init_id: AbstractID, init_belief: AbstractBelief):
        super().__init__(init_id, init_belief)

    @property
    def prior(self) -> AbstractBelief:
        return self[self.id(0)]

    @prior.setter
    def prior(self, prior: AbstractBelief):
        self[self.id(0)] = prior

    @property
    def support(self):
        return self.prior.support

    @support.setter
    def support(self, support):
        # Change the support for all beliefs
        for _, record in self.records.items():
            record.support = support

    @property
    def features(self):
        return [self[id].features for id in self.records.keys()]

    @features.setter
    def features(self, features: dict[AbstractID, Any]):
        for id, record in self.records.items():
            record.features = features[id]

    def query(self, *args, **kwargs) -> Any:
        self[self.id(-1)].query(*args, **kwargs)

    def sample(self, *args, **kwargs) -> Any:
        self[self.id(-1)].sample(*args, **kwargs)

    def update(self, new_id: AbstractID, *args, **kwargs):

        # Create a deep copy of the last belief
        last_belief = self[self.id(-1)]
        new_belief = deepcopy(last_belief)

        # Invoke the update method on the contents of this copy
        new_belief.update(*args, **kwargs)

        # Add new belief
        self[new_id] = new_belief

    def normalize(self, *args, **kwargs):
        # Normalize all beliefs
        for id, record in self.records.items():
            record.normalize(*args, **kwargs)


class BeliefCollection(RecordKeeper):
    """
    A collection of beliefs.
    """

    def __init__(self, n: int, belief_cls: Type[AbstractBelief], *args, **kwargs):
        super().__init__()
        for i in range(n):
            self[IntID(i)] = belief_cls(*args, **kwargs)

    def update(self, i: int, *args, **kwargs):
        self[self.id(i)].update(*args, **kwargs)
