import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gamma

from foraging import SEED
from foraging.models import SuperDict

O = TypeVar("O")
X = TypeVar("X")
RNG = TypeVar("RNG")


## Abstractions
class Belief[X]:
    """
    Interface for a belief.
    Functionally, a belief supports sampling and querying.
    Structurally, a belief has a representation ie. probabilities, sufficient statistics, etc.
    """

    @property
    def representation(self) -> Any: ...

    def sample(self, n: int = 1, rng: RNG = None) -> Iterable[X]: ...

    def query(self, x: X) -> float: ...


class Likelihood[X, O]:
    """
    Interface for a likelihood.
    """

    def __call__(self, o: O, x: X) -> float | Iterable[float]: ...


class BeliefUpdate[X, O]:
    """
    Encapsulates the belief update rule ie. Bayesian, online variational methods, neural networks, etc.
    """

    def __call__(self, prior: Belief[X], o: O) -> Belief[X]: ...


## Implementations
class Probabilities(Belief[X]):
    """
    A belief that has finite and discrete probabilities as the representation.
    """

    def __init__(self, support: Iterable[X], probabilities: Iterable[float]):
        self.support = support
        self._support = np.asarray(support)  # for indexing
        probabilities = np.asarray(probabilities)
        assert np.all(probabilities >= 0), "Probabilities must be non-negative"
        assert np.isclose(np.sum(probabilities), 1.0), "Probabilities must sum to 1"
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

    def sample(self, n: int = 1, rng=None) -> Iterable[X]:
        if rng is None:
            rng = np.random.default_rng(SEED)
        return rng.choice(self._support, size=n, p=self.representation)

    def query(self, x: X) -> float:
        return self.representation[np.argwhere(self._support == x)[0][0]]

    def query_by_index(self, i: int) -> float:
        return self.representation[i]

    def __len__(self) -> int:
        return len(self.support)


class FactorizedBelief(Belief[Iterable[X]]):
    """
    A factorized belief is a belief that is factorized into n_factors beliefs.
    This is useful for representing beliefs over jointly independent variables.
    """

    def __init__(self, n_factors: int, prior: Belief[X]):
        self.beliefs = [deepcopy(prior) for _ in range(n_factors)]

    def __getitem__(self, i: int) -> Belief[X]:
        return self.beliefs[i]

    def __setitem__(self, i: int, belief: Belief[X]):
        self.beliefs[i] = belief

    def __len__(self) -> int:
        return len(self.beliefs)

    @property
    def representation(self) -> Iterable[Any]:
        return [belief.representation for belief in self.beliefs]

    def query(self, x: Iterable[X]) -> float:
        return math.prod([self.beliefs[i].query(_x) for i, _x in enumerate(x)])

    def sample(self, n: int = 1, rng: RNG = None) -> Iterable[X]:
        return [belief.sample(n, rng) for belief in self.beliefs]


class ExactBayesianUpdateOnProbabilities(BeliefUpdate[X, O]):
    """
    Exact Bayesian update on Probabilities.
    """

    def __init__(self, likelihood: "Likelihood[X, O]", vectorize: bool = False):
        """Set vectorize to True to vectorize likelihood over the parameters."""
        self.likelihood = likelihood
        self.vectorize = vectorize

    def __call__(self, prior: "Probabilities", data: O) -> "Probabilities":
        # Bayes rule: posterior ∝ likelihood × prior
        if self.vectorize:
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


class RewardOutcomeLikelihood(Likelihood["GammaParameters", "RewardOutcome"]):
    """
    A likelihood for a single gamma distributed variable.
    """

    def __call__(
        self, o: "RewardOutcome", x: "GammaParameters"
    ) -> float | Iterable[float]:
        # Calculate the probability of reward being available/unavailable after t time has passed under the given latents.
        p_t = gamma.cdf(o.time, x.shape, scale=x.schedule / x.shape)

        # The last element is the probability of being in the last state, i.e., reward being available.
        if (
            not o.is_available
        ):  # If reward is not available, return the complementary probability.
            return 1.0 - p_t
        return p_t


class RewardIntervalLikelihood(Likelihood["GammaParameters", "RewardInterval"]):
    """
    A likelihood for a single gamma distributed variable.
    """

    def __call__(
        self, o: "RewardInterval", x: "GammaParameters"
    ) -> float | Iterable[float]:
        # Calculate the probability of the reward interval under the given latents.
        return gamma.pdf(o.time, x.shape, scale=x.schedule / x.shape)


class PermutationLikelihood(Likelihood["IndexedObservation", "Permutation"]):
    """
    A likelihood for a permutation of schedules.
    """

    def __call__(self, o: "IndexedObservation", x: "Permutation"):
        # Extract the reward availability and push interval.
        i = o.i
        obs = o.observation
        schedule = x.permutation[i]

        # Calculate the probability of reward being available/unavailable after t time has passed under the given latents.
        p_t = gamma.cdf(obs.time, x.shape, scale=schedule / x.shape)

        # The last element is the probability of being in the last state, i.e., reward being available.
        if (
            not obs.is_available
        ):  # If reward is not available, return the complementary probability.
            return 1.0 - p_t
        return p_t


class Posterior[X, O, RNG](Belief[X]):
    """
    Evolving beliefs over a variable X, given observations O.
    """

    def __init__(self, init_id: Any, prior: "Belief[X]", update: "BeliefUpdate[X, O]"):
        self.data = SuperDict({init_id: prior})
        self._update = update

    @property
    def prior(self) -> "Belief[X]":
        return self.data.index(0)[1]

    @property
    def representation(self) -> Iterable[Any]:
        return [self[key].representation for key in self.data.keys()]

    def update(self, key: Any, o: O):
        old_belief = deepcopy(self.head)
        self[key] = self._update(old_belief, o)

    def query(self, x: X) -> float:
        return self.head.query(x)

    def sample(self, n: int = 1, rng: RNG = None) -> Iterable[X]:
        return self.head.sample(n, rng)

    @property
    def head(self) -> "Belief[X]":
        return self.data.index(-1)[1]

    def __getitem__(self, key: Any) -> "Belief[X]":
        return self.data[key]

    def __setitem__(self, key: Any, value: "Belief[X]"):
        self.data[key] = value

    def __delitem__(self, key: Any):
        del self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: Any) -> bool:
        return key in self.data


class FactorizedPosterior[X, O, RNG](Posterior):
    """
    A factorized posterior.
    Each factor is managed as a separate Posterior instance, allowing independent updates.
    """

    def __init__(
        self,
        n_factors: int,
        init_id: Any,
        update: "BeliefUpdate[X, O]",
        prior: "Belief[X]",
    ):
        """
        Initialize a factorized posterior with n_factors independent posteriors.

        Args:
            n_factors: Number of independent factors
            init_id: Initial ID
            update: Belief update function to apply to individual factors
            prior: Prior belief
        """

        # Initialize the first factor as the main posterior
        first_belief = FactorizedBelief(n_factors, prior)
        super().__init__(init_id, first_belief, update)

    def update(self, key: Any, o: "IndexedObservation"):
        i = o.i
        old_belief = deepcopy(self.head)
        old_belief[i] = self._update(old_belief[i], o.observation)
        self[key] = old_belief


## Convenience Data-structures
@dataclass
class RewardOutcome:
    """A reward observation is a boolean indicating whether the reward is available and a float indicating the time the reward observation occurred."""

    is_available: bool
    time: float


@dataclass
class RewardInterval:
    """An observation of the reward interval."""

    time: float


@dataclass
class IndexedObservation[O]:
    """An indexed observation is an integer index and an observation."""

    i: int
    observation: O


@dataclass
class GammaParameters:
    """Gamma distribution parameters."""

    shape: float
    schedule: float


@dataclass
class Permutation:
    """A permutation of schedules and a shape parameter that act like GammaParameters."""

    permutation: list
    shape: float


@dataclass
class PossibleSchedules:
    """
    A convenience class that behaves like an array of schedules but also has structured fields.
    When converted with np.asarray(), it becomes an array of schedules.
    When passed to a Likelihood, it acts as an instance of GammaParameters.
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

    def __getitem__(self, i):
        return self.schedule[i]


class SupportsUpdatesByBox(Protocol):
    """Convenience protocol for updating beliefs with a box-specific observation."""

    def update(self, key: Any, o: IndexedObservation): ...


class Permutation2SchedulesWrapper(Probabilities):
    """A wrapper that converts a Permutation to a list of schedules."""

    def sample(self, n: int = 1, rng: RNG = None) -> Iterable[Iterable[float]]:
        samples = super().sample(n, rng)
        return np.array([sample.permutation for sample in samples]).T
